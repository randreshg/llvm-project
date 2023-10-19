
#include "llvm/Transforms/ARTS/ARTSTransform.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/MemoryDependenceAnalysis.h"
#include "llvm/Analysis/MemorySSA.h"
#include "llvm/Analysis/MemorySSAUpdater.h"
#include "llvm/Analysis/RegionInfo.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/Frontend/ARTS/ARTSIRBuilder.h"

#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/Support/Debug.h"
#include "llvm/Transforms/IPO/Attributor.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/CodeExtractor.h"

#include <optional>
using namespace llvm;

/// DEBUG
#define DEBUG_TYPE "arts-transform"
#if !defined(NDEBUG)
static constexpr auto TAG = "[" DEBUG_TYPE "] ";
#endif

/// COMMAND LINE OPTIONS
static cl::opt<bool>
    EnableARTSTransformation("arts-enable",
                             cl::desc("Disable transformation to ARTS."),
                             cl::Hidden, cl::init(false));

static cl::opt<bool> PrintModuleBeforeOptimizations(
    "arts-print-module-before",
    cl::desc("Print the module before the ARTSTransformer Module Pass"),
    cl::Hidden, cl::init(false));

static cl::opt<bool> PrintModuleAfterOptimizations(
    "arts-print-module-after",
    cl::desc("Print the module after the ARTSTransformer Module Pass"),
    cl::Hidden, cl::init(false));

/// ABSTRACT ATTRIBUTES
/// ARTSInformationCache It stores ARTS related
/// information that the attributor can use
struct ARTSInformationCache : public InformationCache {
  ARTSInformationCache(Module &M, AnalysisGetter &AG,
                       BumpPtrAllocator &Allocator,
                       SetVector<Function *> *Functions)
      : InformationCache(M, AG, Allocator, Functions), AG(AG), ARTSBuilder(M),
        ARTSTransform(M) {

    ARTSBuilder.initialize();
  }

  /// Getters for analysis.
  AnalysisGetter &AG;
  /// The ARTSIRBuilder instance.
  ARTSIRBuilder ARTSBuilder;
  /// ARTSTransformer pointer
  ARTSTransformer ARTSTransform;
};

/// AADataEnv
/// This AbstractAttribute is used to obtain the data environment of a code
/// region, which is the set of variables that are private, shared, firstprivate
/// and lastprivate.
struct AADataEnv : public StateWrapper<BooleanState, AbstractAttribute> {
  using Base = StateWrapper<BooleanState, AbstractAttribute>;

  AADataEnv(const IRPosition &IRP, Attributor &A) : Base(IRP) {}

  /// Statistics are tracked as part of manifest for now.
  void trackStatistics() const override {}

  static AADataEnv &createForPosition(const IRPosition &IRP, Attributor &A);

  /// See AbstractAttribute::getName()
  const std::string getName() const override { return "AADataEnv"; }

  /// See AbstractAttribute::getIdAddr()
  const char *getIdAddr() const override { return &ID; }

  /// This function should return true if the type of the \p AA is
  /// AADataEnv
  static bool classof(const AbstractAttribute *AA) {
    return (AA->getIdAddr() == &ID);
  }

  static const char ID;
};

struct AADataEnvCallSite : AADataEnv {
  AADataEnvCallSite(const IRPosition &IRP, Attributor &A) : AADataEnv(IRP, A) {}

  const std::string getAsStr(Attributor *A) const override {
    if (!isValidState())
      return "<invalid>";

    std::string Str("AADataEnvCallSite: ");
    return Str + "OK";
  }

  ChangeStatus handleParallelRegion(CallBase &CB, Attributor &A) {
    /// Get the number of arguments
    unsigned int NumArgs = CB.data_operands_size();
    /// Get private and shared variables
    for (unsigned int ArgItr = 3; ArgItr < NumArgs; ArgItr++) {
      Value *Arg = CB.getArgOperand(ArgItr);
      Type *ArgType = Arg->getType();
      // LLVM_DEBUG(dbgs() << "Argument: " << *Arg << "\n");
      // LLVM_DEBUG(dbgs() << "Argument type: " << *ArgType << "\n");

      /// For now assume that if it is a pointer, it is a shared variable
      if (PointerType *PT = dyn_cast<PointerType>(ArgType))
        DE.SharedVars.push_back(Arg);
      /// If not, it is a first private variable
      else
        DE.FirstprivateVars.push_back(Arg);

      /// NOTES: The outline function or the variable should have attributes
      /// that provide more information about the lifetime of the variable. It
      /// is also important to consider the underlying type of the variable.
      /// There may be cases, where there is a firstprivate variable that is a
      /// pointer. For this case, the pointer is private, but the data it points
      /// to is shared.
      ///
      /// Do we need to consider private variables?
    }
    OutlinedFunction = dyn_cast<Function>(CB.getArgOperand(2));
    /// For now, indicate optimistic fixpoint
    indicateOptimisticFixpoint();
    LLVM_DEBUG(dbgs() << "[AADataEnvCallSite] Parallel region - Data "
                         "environment filled out\n");
    return ChangeStatus::CHANGED;
  }

  ChangeStatus handleTaskRegion(CallBase &CB, Attributor &A) {
    // For the shared variables we are interested, in all stores that are done
    // to the shareds field of the kmp_task_t struct. For the firstprivate
    // variables we are interested in all stores that are done to the privates
    // field of the kmp_task_t_with_privates struct.
    //
    // The AAPointerInfo attributor is used to obtain the access (read/write,
    // offsets and size) of a Val. This attributor is run on the returned Val
    // of the taskalloc function. The returned Val is a pointer to the
    // kmp_task_t_with_privates struct.
    // struct kmp_task_t_with_privates {
    //    kmp_task_t task_data;
    //    kmp_privates_t privates;
    // };
    // typedef struct kmp_task {
    //   void *shareds;
    //   kmp_routine_entry_t routine;
    //   kmp_int32 part_id;
    //   kmp_cmplrdata_t data1;
    //   kmp_cmplrdata_t data2;
    // } kmp_task_t;
    //
    // - For shared variables, the access of the shareds field is obtained by
    // obtaining stores done to offset 0 of the returned Val of the taskalloc.
    // -For firstprivate variables, the access of the privates field is obtained
    // by obtaining stores done to offset 8 of the returned Val of the
    // kmp_task_t_with_privates struct.
    //
    // If there is a load instruction that uses the returned Val of the
    // taskalloc function, we need to run the AAPointerInfo attributor on it.
    //
    // The function returns ChangeStatus::CHANGED if the data environment is
    // updated, ChangeStatus::UNCHANGED otherwise.

    ChangeStatus Changed = ChangeStatus::UNCHANGED;
    LLVM_DEBUG(dbgs() << "[AADataEnvCallSite] Task alloc - FOUND\n");
    DataEnv AuxDE(DE);
    /// Get context and module
    LLVMContext &Ctx = CB.getContext();
    Module *M = CB.getModule();
    /// Get the size of the kmp_task_t struct
    auto *kmp_task_t = dyn_cast<StructType>(CB.getType());
    auto kmp_task_t_size = M->getDataLayout().getTypeAllocSize(
        kmp_task_t->getTypeByName(Ctx, "struct.kmp_task_t"));
    /// Run AAPointerInfo on the callbase
    auto *PI = A.getAAFor<AAPointerInfo>(
        *this, IRPosition::callsite_returned(CB), DepClassTy::OPTIONAL);
    if (!PI->getState().isValidState()) {
      indicatePessimisticFixpoint();
      LLVM_DEBUG(dbgs() << "[AADataEnvCallSite] AAPointerInfo is invalid\n");
      return Changed;
    }
    /// Is it a fixpoint?
    if (!PI->getState().isAtFixpoint()) {
      LLVM_DEBUG(
          dbgs() << "[AADataEnvCallSite] AAPointerInfo is not at fixpoint\n");
      return Changed;
    }
    LLVM_DEBUG(dbgs() << "[AADataEnvCallSite] AAPointerInfo is valid with: "
                      << PI->getOffsetBinsSize() << " bins\n");
    /// Analyze result
    /// This CB checks the access of each offset bin of PI
    const std::function<bool(const AA::RangeTy &, const SmallSet<unsigned, 4> &,
                             const AAPointerInfo *AAPI)>
        AccessCB = [&](const AA::RangeTy &Range,
                       const SmallSet<unsigned, 4> &AccIndex,
                       const AAPointerInfo *AAPI) {
          // LLVM_DEBUG(dbgs()
          //             << "[" << Range.Offset << "-" << Range.Offset +
          //             Range.Size
          //             << "] : " << AccIndex.size() << "\n");
          for (auto AI : AccIndex) {
            auto &Acc = AAPI->getAccess(AI);
            auto AccKind = Acc.getKind();
            auto *Inst = Acc.getLocalInst();
            // LLVM_DEBUG(dbgs() << "     - " << AccKind << " - " << *Inst <<
            // "\n");

            if (Acc.isWrittenValueYetUndetermined()) {
              // LLVM_DEBUG(dbgs()
              //             << "       - c: Value written is not known yet
              //             \n");
              return false;
            }
            /// Store instruction.
            if (AccKind == AAPointerInfo::AccessKind::AK_MUST_WRITE) {
              // LLVM_DEBUG(dbgs() << "       -- Store instruction\n");
              Value *Val = Acc.getWrittenValue();
              if (!Val)
                return false;
              /// Check if it is a shared variable
              if (Range.Offset == 0) {
                // LLVM_DEBUG(dbgs() << "       - c: shared variable " << *Val
                // << "\n");
                AuxDE.SharedVars.push_back(Val);
              }
              /// Check if it is a firstprivate variable
              else if ((uint64_t)Range.Offset >= (uint64_t)kmp_task_t_size) {
                // LLVM_DEBUG(dbgs() << "       - c: firstprivate variable " <<
                // *Val << "\n");
                AuxDE.FirstprivateVars.push_back(Val);
              }
              // else
              //   LLVM_DEBUG(dbgs() << "       - c: other: " << *Val << "\n");
            }
            /// Read instruction
            else if (AccKind == AAPointerInfo::AccessKind::AK_MUST_READ) {
              // LLVM_DEBUG(dbgs() << "       -- Read instruction\n");
              /// Check if it a load instruction
              if (auto *LI = dyn_cast<LoadInst>(Inst)) {
                // LLVM_DEBUG(dbgs() << "       - Load instruction. Running AA
                // on this.\n");
                auto *PIL = A.getAAFor<AAPointerInfo>(
                    *this, IRPosition::value(*LI), DepClassTy::OPTIONAL);
                if (!PIL->getState().isValidState()) {
                  indicatePessimisticFixpoint();
                  // LLVM_DEBUG(dbgs() <<"[AADataEnvCallSite] PIL is
                  // invalid\n");
                  return false;
                }
                /// Is it a fixpoint?
                if (!PIL->getState().isAtFixpoint()) {
                  // LLVM_DEBUG(dbgs() <<"[AADataEnvCallSite] PIL is not at
                  // fixpoint\n");
                  return false;
                }
                /// For all offset bins in the PI, run AccessCB
                if (!PIL->forallOffsetBins(AccessCB)) {
                  // LLVM_DEBUG(dbgs() << "PI forallOffsetBins failed\n");
                  indicatePessimisticFixpoint();
                  return false;
                }
              }
            }
          }
          return true;
        };

    /// For all offset bins in the PI, run AccessCB
    if (!PI->forallOffsetBins(AccessCB)) {
      LLVM_DEBUG(dbgs() << "PI forallOffsetBins failed\n");
      /// We dont indicate fixpoint here. The analysis could've failed because
      /// PI is not at fixpoint.
      return Changed;
    }
    /// If we reach this point, it means that the PI is valid and at fixpoint.
    indicateOptimisticFixpoint();
    /// Append the data environment
    DE.append(AuxDE);
    OutlinedFunction = dyn_cast<Function>(CB.getArgOperand(5));
    return ChangeStatus::CHANGED;
  }

  ChangeStatus handleTaskwait() { return ChangeStatus::UNCHANGED; }

  void initialize(Attributor &A) override {
    CallBase &CB = cast<CallBase>(getAssociatedValue());
    Function *Callee = getAssociatedFunction();
    LLVM_DEBUG(dbgs() << "[AADataEnvCallSite] initialize: " << Callee->getName()
                      << "\n");
    switch (OMPInfo::getRTFunction(Callee)) {
    case OMPInfo::SET_NUM_THREADS: {
      auto *Arg = CB.getArgOperand(0);
      if (auto *NumThreads = dyn_cast<ConstantInt>(Arg)) {
        LLVM_DEBUG(dbgs() << "Number of threads: " << NumThreads->getZExtValue()
                          << "\n");
      }
      indicateOptimisticFixpoint();
    } break;
    case OMPInfo::PARALLEL:
      /// Handle it in the updateImpl function
      break;
    case OMPInfo::TASKALLOC: {
      /// The task is created by the taskalloc function, and it returns a
      /// pointer to the task (check handleTaskRegion documentation ).
      /// We need to obtain its data environment. Since this pointer is used in
      /// functions that are not analyzable, and for purposes of this analysis
      /// we are not interested on what happens inside those functions, we will
      /// simply remove it. A pointer is captured by the call if it
      /// makes a copy of any part of the pointer that outlives the call.
      for (auto &U : CB.uses()) {
        auto *User = U.getUser();
        // LLVM_DEBUG(dbgs() << "--- " << *U << "\n");
        LLVM_DEBUG(dbgs() << "------ " << *User << "\n");
        /// Convert to instruction
        if (!isa<Instruction>(User))
          continue;
        // LLVM_DEBUG(dbgs() << "------ " << *(U.get()) << "\n");
        if (auto *CBI = dyn_cast<CallInst>(User)) {
          // auto *Callee = CBI->getCalledFunction();
          // LLVM_DEBUG(dbgs() << "------ " << Callee->getName() << "\n");
          /// Add the attribute nocapture to the returned Val of the taskalloc
          if (OMPInfo::isTaskFunction(CBI->getCalledFunction())) {
            CBI->eraseFromParent();
          }
              // CBI->addParamAttr(U.getOperandNo(), Attribute::NoCapture);
        }
      }
      /// Lets handle the task in the updateImpl function
    } break;
    case OMPInfo::OTHER: {
      LLVM_DEBUG(dbgs() << "Other instruction - FOUND\n");
      /// Unknown caller or declarations are not analyzable, we give up.
      if (!Callee || !A.isFunctionIPOAmendable(*Callee)) {
        indicatePessimisticFixpoint();
        LLVM_DEBUG(dbgs() << "Unknown caller or declarations are not "
                             "analyzable, we give up.\n");
        return;
      }
    } break;
    /// Default
    default: {
      LLVM_DEBUG(dbgs() << "Other instruction - FOUND\n");
    } break;
    }
  }

  ChangeStatus updateImpl(Attributor &A) override {
    ChangeStatus Changed = ChangeStatus::UNCHANGED;
    CallBase &CB = cast<CallBase>(getAssociatedValue());
    Function *Callee = getAssociatedFunction();
    LLVM_DEBUG(dbgs() << "[AADataEnvCallSite] updateImpl: " << Callee->getName()
                      << "\n");

    switch (OMPInfo::getRTFunction(Callee)) {
    case OMPInfo::PARALLEL:
      Changed |= handleParallelRegion(CB, A);
      break;
    case OMPInfo::TASKALLOC:
      Changed |= handleTaskRegion(CB, A);
      break;
    default: {
      LLVM_DEBUG(dbgs() << "Other instruction - FOUND\n");
    } break;
    }

    if (Changed == ChangeStatus::CHANGED) {
      /// If the data environment is updated, update info in the cache
      auto &AIC = static_cast<ARTSInformationCache &>(A.getInfoCache());
      auto *EB = AIC.ARTSTransform.getEDTBlock(CB.getParent());
      auto &OMP = EB->OMP;
      OMP.DE.append(DE);
      OMP.F = OutlinedFunction;
    }
    return Changed;
  }

  ChangeStatus manifest(Attributor &A) override {
    ChangeStatus Changed = ChangeStatus::UNCHANGED;
    LLVM_DEBUG(dbgs() << "[AADataEnvCallSite] Manifest\n");
    return Changed;
  }

  /// Attributes
  DataEnv DE;
  Function *OutlinedFunction = nullptr;
};

AADataEnv &AADataEnv::createForPosition(const IRPosition &IRP, Attributor &A) {
  AADataEnv *AA = nullptr;
  switch (IRP.getPositionKind()) {
  case IRPosition::IRP_INVALID:
  case IRPosition::IRP_ARGUMENT:
  case IRPosition::IRP_CALL_SITE_ARGUMENT:
  case IRPosition::IRP_RETURNED:
  case IRPosition::IRP_CALL_SITE_RETURNED:
  case IRPosition::IRP_FLOAT:
  case IRPosition::IRP_FUNCTION:
    llvm_unreachable(
        "AAToARTS can only be created for float/callsite position!");
  case IRPosition::IRP_CALL_SITE:
    AA = new (A.Allocator) AADataEnvCallSite(IRP, A);
    break;
  }
  return *AA;
}

const char AADataEnv::ID = 0;

/// AAToARTS
/// This AbstractAttribute analyzes a given function and tries to determine
/// if it can be transformed to ARTS.
using BlockSequence = SmallVector<BasicBlock *, 0>;
struct AAToARTS : public StateWrapper<BooleanState, AbstractAttribute> {
  using Base = StateWrapper<BooleanState, AbstractAttribute>;

  AAToARTS(const IRPosition &IRP, Attributor &A) : Base(IRP) {}

  /// Statistics are tracked as part of manifest for now.
  void trackStatistics() const override {}

  static AAToARTS &createForPosition(const IRPosition &IRP, Attributor &A);

  /// See AbstractAttribute::getName()
  const std::string getName() const override { return "AAToARTS"; }

  /// See AbstractAttribute::getIdAddr()
  const char *getIdAddr() const override { return &ID; }

  /// This function should return true if the type of the \p AA is
  /// AAToARTS
  static bool classof(const AbstractAttribute *AA) {
    return (AA->getIdAddr() == &ID);
  }

  static const char ID;
};

struct AAToARTSFunction : AAToARTS {
  AAToARTSFunction(const IRPosition &IRP, Attributor &A) : AAToARTS(IRP, A) {}

  /// See AbstractAttribute::getAsStr(Attributor *A)
  const std::string getAsStr(Attributor *A) const override {
    if (!isValidState())
      return "<invalid>";
    std::string Str("AAToARTSFunction: ");
    return Str + "OK";
  }

  void visitRegion(Region &R) {
    // LLVM_DEBUG(dbgs() << TAG << "Visiting region: " << R << "\n");
    printRegion(R);
    for(auto &Itr: R) {
      Region *SubRegion = Itr.get();
      visitRegion(*SubRegion);
    }
  }

  void printRegion(Region &R) {
    LLVM_DEBUG(dbgs() << "----------------------\n");
    LLVM_DEBUG(dbgs() << "Region: " << R.getNameStr() << "\n");
    LLVM_DEBUG(dbgs() << "Top level? " << R.isTopLevelRegion() << "\n");
    LLVM_DEBUG(dbgs() << "Depth: " << R.getDepth() << "\n");
    LLVM_DEBUG(dbgs() << "Entry: " << R.getEntry()->getName() << "\n");
    for (auto *BB : R.blocks()) {
      LLVM_DEBUG(dbgs() << "BB: " << BB->getName() << "\n");
    }
    const BasicBlock *ExitBB = R.getExit();
    if(ExitBB)
      LLVM_DEBUG(dbgs() << "Exit: " << ExitBB->getName() << "\n");
    else
      LLVM_DEBUG(dbgs() << "Exit: <Function Return>\n");

  }

  bool identifyRegions(Function &F, AnalysisGetter &AG) {
    // auto *LI = AG.getAnalysis<LoopAnalysis>(F);
    // auto *DT =  AG.getAnalysis<DominatorTreeAnalysis>(F);
    auto *RI = AG.getAnalysis<RegionInfoAnalysis>(F);
    auto *Region = RI->getTopLevelRegion();
    visitRegion(*Region);
    return true;
  }

  void dominatedBBs(BasicBlock *FromBB, DominatorTree &DT, BlockSequence &DominatedBlocks) {
    /// This function finds the BBs that are dominated by FromBB and add
    /// them to the DominatedBlocks vector.
    Function &F = *FromBB->getParent();
    // LLVM_DEBUG(dbgs() << "BB " << FromBB->getName()<< ", dominates:\n");
    for (auto &ToBB : F) {
        if (DT.dominates(FromBB, &ToBB))
          DominatedBlocks.push_back(&ToBB);
    }
    /// Then we create an EDT
    // auto *EDT = AT->insertEDT(&F);
    /// Insert all dominated BBs to the EDT
  }

  /// Given a basic block, this function creates a function that contains
  /// the instructions of the BB. The function is then inserted to the
  /// module.
  Function *createFunction(DominatorTree *DT, BasicBlock *FromBB, 
                           bool DTAnalysis = false,
                           SmallVector<Value *, 0> *ExcludeArgsFromAggregate = nullptr) {
    Function &F = *FromBB->getParent();
    AssumptionCache *AC = AG->getAnalysis<AssumptionAnalysis>(F, true);
    CodeExtractorAnalysisCache CEAC(F);
    /// Collect blocks
    BlockSequence Region;
    /// Get all BBs that are dominated by FromBB
    if(DTAnalysis)
      dominatedBBs(FromBB, *DT, Region);
    else
      Region.push_back(FromBB);
    LLVM_DEBUG(dbgs() << TAG << "Creating function for: " 
                      << FromBB->getName() << " - " << Region.size() << "\n");
    /// Extract code from the region
    auto FunctionName = "edt." + FromBB->getName().str();
    CodeExtractor CE(Region, DT, /* AggregateArgs */ false, /* BFI */ nullptr,
                   /* BPI */ nullptr, AC, /* AllowVarArgs */ true,
                   /* AllowAlloca */ true, /* AllocaBlock */ nullptr,
                   /* Suffix */ FunctionName);
    /// Debug info
    // SetVector<Value *> Inputs, Outputs, Sinks;
    // CE.findInputsOutputs(Inputs, Outputs, Sinks);
    // LLVM_DEBUG(dbgs() << "Inputs\n");
    // for (auto *I : Inputs) {
    //   LLVM_DEBUG(dbgs() << *I << "\n");
    // }
    // LLVM_DEBUG(dbgs() << "Outputs\n");
    // for (auto *O : Outputs) {
    //   LLVM_DEBUG(dbgs() << *O << "\n");
    // }
    // LLVM_DEBUG(dbgs() << "Sinks\n");
    // for (auto *S : Sinks) {
    //   LLVM_DEBUG(dbgs() << *S << "\n");
    // }
    assert(CE.isEligible() &&
           "Expected Region outlining to be possible!");
    

    if(ExcludeArgsFromAggregate)
      for (auto *V : *ExcludeArgsFromAggregate)
        CE.excludeArgFromAggregate(V);
    /// Generate function
    Function *OutF = CE.extractCodeRegion(CEAC);
    LLVM_DEBUG(dbgs() << "Extracted function: " << *OutF << "\n");
    return OutF;
  }

  void analyzeOutlinedRegion(Function &F, SmallVector<Value *, 0> *ArgsToRemove = nullptr) {
    if (ArgsToRemove) {
      for (auto *Arg : *ArgsToRemove) {
        // LLVM_DEBUG(dbgs() << "Removing argument: " << *Arg << "\n");
        /// Check all uses of the argument
        for (auto &U : Arg->uses()) {
          // LLVM_DEBUG(dbgs() << "  -Use: " << *U.getUser() << "\n");
          /// Convert to instruction
          auto *I = dyn_cast<Instruction>(U.getUser());
          if (!I)
            continue;
          /// If the argument is used by a call instruction, replace it with a nullptr
          if (auto *CBI = dyn_cast<CallBase>(I)) {
            CBI->setArgOperand(U.getOperandNo(), UndefValue::get(Arg->getType()));
            continue;
          }
          /// If it is a load instruction, replace it with 0
          if (auto *LI = dyn_cast<LoadInst>(I)) {
            LI->replaceAllUsesWith(ConstantInt::get(LI->getType(), 0));
            I->eraseFromParent();
            break;
          }
          /// Remove instruction
          I->replaceAllUsesWith(UndefValue::get(Arg->getType()));
          I->eraseFromParent();
        }
      }
    }
  }

  bool identifyEDTs(Function &F, Attributor &A) {
    /// This function identifies the regions that will be transformed to
    /// have and EDT initializer.
    /// If a basic block contains a call to the following functions, it
    /// most likely can be transformed to an EDT:
    /// - __kmpc_fork_call.
    /// - __kmpc_single
    /// - __kmpc_omp_taskwait or __kmpc_omp_taskwait_deps
    /// - function with a pointer return type (not void) that is used
    ///   after the call.
    ///
    /// The BB that meet the criteria are split into two BBs. Check example:
    /// BB1:
    ///   %0 = alloca i32, align 4
    ///   %1 = call i32 @__kmpc_fork_call(i32 1, i32 (i32, i8*)*
    ///   @__omp_offloading_fib, i8* %0)
    ///   ...
    ///   ret i32 0
    /// It is transformed to:
    /// BB1:
    ///   %0 = alloca i32, align 4
    ///   %1 = call i32 @__kmpc_fork_call(i32 1, i32 (i32, i8*)*
    ///   @__omp_offloading_fib, i8* %0) br label %edt.done.0
    /// edt.done.0:
    ///   ...
    ///   ret i32 0
    ///
    /// For the example above, the BB that contains the call to __kmpc_fork_call
    /// will be then replaced by the EDT initializer (GUID reserve + call to EDT
    /// allocator) and the parallel outlined function will be the EDT. We then
    /// have to analyze whether the par.done BB needs to be transformed to an
    /// EDT. This is done by analyzing the instructions of the BB to obtain its
    /// data environment. If any of the instructions in the BB uses a shared
    /// variable, the BB is transformed to an EDT.
    ///
    /// By the end of this function, we should have a map that contains the BBs
    /// that can be transformed to EDTs. The key of the map is the BB that
    /// contains the call to the function that creates the region (e.g.
    /// __kmpc_fork_call) and the Val is the struct EDTInfo. Aux variables

    /// Aux variables
    LoopInfo *LI = nullptr;
    DominatorTree *DT =  AG->getAnalysis<DominatorTreeAnalysis>(F);

    /// Region counter
    unsigned int ParallelRegion = 0;
    unsigned int TaskRegion = 0;

    LLVM_DEBUG(dbgs() << TAG << "Identifying EDT regions\n");

    /// If function is not IPO amendable, we give up
    if (!A.isFunctionIPOAmendable(F))
      return false;

    /// Get entry block
    BasicBlock *CurrentBB = &(F.getEntryBlock());
    BasicBlock *NextBB = nullptr;
    do {
      // LLVM_DEBUG(dbgs() << TAG << "CurrentBB: " << *CurrentBB << "\n");
      NextBB = CurrentBB->getNextNode();
      /// Get first instruction of the function
      Instruction *CurrentI = &(CurrentBB->front());
      do {
        // LLVM_DEBUG(dbgs() << TAG << "CurrentI: " << *CurrentI << "\n");
        /// We are only interested in call instructions
        auto *CB = dyn_cast<CallBase>(CurrentI);
        if (!CB)
          continue;
        /// Get the callee
        Function *Callee = CB->getCalledFunction();
        OMPInfo::RTFType RTF = OMPInfo::getRTFunction(Callee);
        switch (RTF) {
        case OMPInfo::PARALLEL: {
          OMPInfo OI(OMPInfo::PARALLEL, CB);

          /// Split block at __kmpc_parallel
          BasicBlock *ParallelBB =
              SplitBlock(CurrentI->getParent(), CurrentI, DT, LI, nullptr,
                         "par.region." + std::to_string(ParallelRegion));
          /// Split block at the next instruction
          CurrentI = CurrentI->getNextNonDebugInstruction();
          BasicBlock *ParallelDone =
              SplitBlock(ParallelBB, CurrentI, DT, LI, nullptr,
                         "par.done." + std::to_string(ParallelRegion));
          NextBB = ParallelDone;
          ParallelRegion++;

          /// Analyze outlined region
          Function &OutlinedFunction = *dyn_cast<Function>(CB->getArgOperand(2));
          SmallVector<Value *, 0> ArgsToRemove = {
            OutlinedFunction.getArg(0), 
            OutlinedFunction.getArg(1)};
          analyzeOutlinedRegion(OutlinedFunction, &ArgsToRemove);
          identifyEDTs(OutlinedFunction, A);
          LLVM_DEBUG(dbgs() << "Outlined function: " << OutlinedFunction<< "\n");
          createFunction(DT, &OutlinedFunction.getEntryBlock(), true);

          /// Extract regions and outline them into functions
          Function *ParallelFunction = createFunction(DT, ParallelBB);
          /// For the Done function run the analysis again
          Function *DoneFunction = createFunction(DT, ParallelDone, true);
          identifyEDTs(*DoneFunction, A);
        } break;
        case OMPInfo::TASKALLOC: {
          OMPInfo OI(OMPInfo::TASK, CB);
          /// Split block at __kmpc_omp_task_alloc
          BasicBlock *TaskBB =
              SplitBlock(CurrentI->getParent(), CurrentI, DT, LI, nullptr,
                         "task.region." + std::to_string(TaskRegion));
          /// Find the task call
          while ((CurrentI = CurrentI->getNextNonDebugInstruction())) {
            auto *TCB = dyn_cast<CallBase>(CurrentI);
            if (TCB && OMPInfo::getRTFunction(TCB->getCalledFunction()) ==
                           OMPInfo::TASK)
              break;
          }
          assert(CurrentI && "Task RT call not found");
          /// Split block again at the next instruction
          CurrentI = CurrentI->getNextNonDebugInstruction();
          BasicBlock *TaskDone =
              SplitBlock(TaskBB, CurrentI, DT, LI, nullptr,
                         "task.done." + std::to_string(TaskRegion));
          NextBB = TaskDone;
          TaskRegion++;
          /// Analyzed outlined region
          Function &OutlinedFunction = *dyn_cast<Function>(CB->getArgOperand(5));
          identifyEDTs(OutlinedFunction, A);
          /// Extract regions and create aux functions
          createFunction(DT, TaskBB);
          /// For the Done function run the analysis again
          Function *DoneFunction = createFunction(DT, TaskDone, true);
          identifyEDTs(*DoneFunction, A);
        } break;
        case OMPInfo::TASKWAIT: {
          /// \Note A taskwait requires an event.
          /// Split block at __kmpc_omp_taskwait
          BasicBlock *TaskWaitBB =
              SplitBlock(CurrentI->getParent(), CurrentI, DT, LI, nullptr,
                         "taskwait.region." + std::to_string(TaskRegion));
          /// Split block again at the next instruction
          CurrentI = CurrentI->getNextNonDebugInstruction();
          BasicBlock *TaskWaitDone =
              SplitBlock(TaskWaitBB, CurrentI, DT, LI, nullptr,
                         "taskwait.done." + std::to_string(TaskRegion));
          NextBB = TaskWaitDone;
          /// Add the taskwait region to the map
          // AT.insertEDTBlock(TaskWaitBB, RTF, CB);
          TaskRegion++;
        } break;
        case OMPInfo::OTHER: {
          // if (!Callee || !A.isFunctionIPOAmendable(*Callee))
          //   continue;
          // /// Get return type of the callee
          // Type *RetTy = Callee->getReturnType();
          // /// If the return type is void, we are not interested
          // if(RetTy->isVoidTy())
          //   continue;
          /// If the return type is a pointer, it is because we probably would
          /// use the returned Val. For this case, we need to create an EDT.
        } break;
        default:
          continue;
          break;
        }
      } while ((CurrentI = CurrentI->getNextNonDebugInstruction()));
    } while ((CurrentBB = NextBB));
    
    // /// Fill out set of read/write instructions
    // bool UsedAssumedInformationInCheckRWInst = true;
    // if (!A.checkForAllReadWriteInstructions(
    //         [&](Instruction &I) {
    //           if (I.isLifetimeStartOrEnd())
    //             return true;
    //           if(!AT.getEDTBlock(&I)->isInit()) {
    //             AT.insertRWInst(&F, &I);
    //             return true;
    //           }
    //           /// Only add main instruction for init block. The "main" instruction correspond
    //           /// to the function that contains the call to the outlined function.
    //           OMPInfo::RTFType RTF = OMPInfo::getRTFunction(&I);
    //           if(RTF == OMPInfo::PARALLEL || RTF == OMPInfo::TASKALLOC)
    //             AT.insertRWInst(&F, &I);
    //           return true;
    //         },
    //         *this, UsedAssumedInformationInCheckRWInst)) {
    //   LLVM_DEBUG(dbgs() << "  - checkForAllReadWriteInstructions failed\n");
    //   indicatePessimisticFixpoint();
    //   // return llvm::ChangeStatus::CHANGED;
    // }
    // LLVM_DEBUG(dbgs() << "  - Number of RW instructions: "
    //                   << AT.RWInsts[&F].size() << "\n");

    // /// Fill out set of external instructions
    // for (auto *I : AT.RWInsts[&F]) {
    //   LLVM_DEBUG(dbgs() << "    - Inst: " << *I << "\n");
    //   /// Get the EDT block of the instruction
    //   EDTBlock *CurrentEB = AT.getEDTBlock(I);
    //   assert(CurrentEB && "EDT block not found");
    //   /// Handle EDT init block
    //   /// Obtain the data environment of the EDT Init blocks (OpenMP regions)
    //   if (CurrentEB->isInit()) {
    //     LLVM_DEBUG(dbgs() << "[AADataEnvFunction] EDT init block: "
    //                     << CurrentEB->OMP.getCBName() << "\n");
    //     LLVM_DEBUG(dbgs() << "- Instruction: " << *I << "\n");
        
    //     auto *DEAA = A.getAAFor<AADataEnv>(
    //         *this, IRPosition::callsite_function(*CurrentEB->OMP.CB),
    //         DepClassTy::OPTIONAL);
    //     /// Check if it is at fixpoint
    //     if (!DEAA->getState().isAtFixpoint()) {
    //       LLVM_DEBUG(dbgs() << "[AADataEnvFunction] DEAA is not at fixpoint\n");
    //       // return llvm::ChangeStatus::CHANGED;
    //       return false;
    //     }
    //     /// Check if it is valid
    //     if (!DEAA->getState().isValidState()) {
    //       LLVM_DEBUG(dbgs() << "[AADataEnvFunction] DEAA is invalid\n");
    //       indicatePessimisticFixpoint();
    //       // return llvm::ChangeStatus::CHANGED;
    //       return false;
    //     }
    //     /// This will be handled later
    //     continue;
    //   }
    //   /// Iterate Instruction operands
    //   for (Use &Op : I->operands()) {
    //     /// We are only concerned with instructions
    //     Instruction *OpInst = dyn_cast<Instruction>(Op.get());
    //     if (!OpInst)
    //       continue;
    //     /// If Operand is in the same EDT ignore it
    //     auto &OpEB = *AT.getEDTBlock(OpInst);
    //     if (CurrentEB->isInSameEDT(OpEB))
    //       continue;
    //     /// If the Val is a function, ignore it
    //     if (isa<Function>(Op))
    //       continue;
    //     CurrentEB->EDT->ExtInsts.insert(OpInst);
    //     LLVM_DEBUG(dbgs() << "      - External Val for EDT #"
    //                       << CurrentEB->EDT->ID << ": " << *Op << "\n");
    //   }
    // }
    // identifyRegions(F, AG);
    // LLVM_DEBUG(dbgs() << "  - EDT regions identified: "
    //                   << AT.EDTsFromFunction[&F].size() << "\n");
    return false;
  }

  void initialize(Attributor &A) override {
    Function *F = getAnchorScope();
    LLVM_DEBUG(dbgs() << "\n[AAToARTSFunction] initialize: " << F->getName()
                      << "\n");
    /// Local attributes
    AIT = &static_cast<ARTSInformationCache &>(A.getInfoCache());
    AT = &AIT->ARTSTransform;
    AG = &AIT->AG;
    /// Identify
    if (!identifyEDTs(*F, A)) {
      indicatePessimisticFixpoint();
      return;
    }
    LLVM_DEBUG(dbgs() << *F << "\n");
    /// The rest is handled in the updateImpl function
  }

  ChangeStatus updateImpl(Attributor &A) override {
    Function *F = getAnchorScope();
    LLVM_DEBUG(dbgs() << "[AAToARTSFunction] updateImpl: " << F->getName()
                      << "\n");
    auto &ARTSInfoCache = static_cast<ARTSInformationCache &>(A.getInfoCache());
    auto &AT = ARTSInfoCache.ARTSTransform;

    

    // /// Iterate through the EDTs to analyze the EDT with empty number of Blocks
    // LLVM_DEBUG(dbgs() << "\n" << TAG << "EDTs with empty number of Blocks:\n");
    // for (auto *EDT : AT.EDTsFromFunction[F]) {
    //   if (EDT->Blocks.size() != 0)
    //     continue;
    //   LLVM_DEBUG(dbgs() << "EDT #" << EDT->ID << "\n");
    //   auto *EDTInit = EDT->Init;
    //   assert((EDTInit && EDTInit->HasOMP) && "EDTInit not found");
    //   auto &OMP = EDTInit->OMP;
    //   LLVM_DEBUG(dbgs() << "  - OMP: " << OMP.F->getName() << "\n");

    //   /// Add it to the EDTForFunction map
    //   AT.insertEDTForFunction(OMP.F, EDT);
    //   /// Run AAToARTS on the EDTInit function
    //   auto *OMPFAA = A.getAAFor<AAToARTS>(*this, IRPosition::function(*OMP.F),
    //                                       DepClassTy::OPTIONAL);
    //   if (!OMPFAA->getState().isValidState()) {
    //     LLVM_DEBUG(dbgs() << "[AAToARTSFunction] OMPFAA is invalid\n");
    //     indicatePessimisticFixpoint();
    //     return ChangeStatus::CHANGED;
    //   }
    //   if (!OMPFAA->getState().isAtFixpoint()) {
    //     LLVM_DEBUG(dbgs() << "[AAToARTSFunction] OMPFAA is not at fixpoint\n");
    //     return ChangeStatus::UNCHANGED;
    //   }
    // }

    // /// If this is the main function
    // if (F->getName() == "main") {
    //   LLVM_DEBUG(dbgs() << TAG
    //                     << "----------- PROCESS HAS FINISHED -----------\n");
    //   EDTInfo *MainEDT = AT.getEDTForFunction(F);
    //   assert(MainEDT && "Main EDT not found");
    //   auto &ARTSBuilder = ARTSInfoCache.ARTSBuilder;

    //   LLVM_DEBUG(dbgs() << TAG << "----------- EDTs INFORMATION -----------\n");
    //   LLVM_DEBUG(dbgs() << *MainEDT << "\n");
    //   Function *MainEDTFunction = ARTSBuilder.createEDT("main.edt");
    //   MainEDT->setF(MainEDTFunction);
    //   /// Iterate through the EDT blocks
    //   for(auto *EB : MainEDT->Blocks) {
    //     /// Entry and Other blocks are inmmediately added to the main EDT Function
    //     if(EB->isEntry() || EB->isOther()) {
    //       ARTSBuilder.insertEDTBlock(EB, MainEDTFunction);
    //       continue;
    //     }
    //     /// If it is an init block, we need to create a new EDT function for the 
    //     /// outlined function
    //     if(EB->isInit()) {
    //       auto &OMP = EB->OMP;
    //       assert(OMP.F && "OMP.F not found");
    //       ARTSBuilder.insertEDTBlock(EB, MainEDTFunction);
    //       Function *EDTFunction = ARTSBuilder.createEDT(OMP.F->getName());
    //       /// Is there a init alloca EDT block?
    //       if(!MainEDT->InitAlloca) {
    //         // ARTSBuilder.insertEDTBlock(EB->InitAlloca, EDTFunction);
    //       }
    //       continue;
    //     }
    //   }

    // }
    indicateOptimisticFixpoint();
    return ChangeStatus::CHANGED;
  }

  ChangeStatus manifest(Attributor &A) override {
    ChangeStatus Changed = ChangeStatus::UNCHANGED;
    LLVM_DEBUG(dbgs() << "[AAToARTSFunction] Manifest\n");
    return Changed;
  }

  /// Attributes
  ARTSInformationCache *AIT;
  ARTSTransformer *AT;
  AnalysisGetter *AG;
};

AAToARTS &AAToARTS::createForPosition(const IRPosition &IRP, Attributor &A) {
  AAToARTS *AA = nullptr;
  switch (IRP.getPositionKind()) {
  case IRPosition::IRP_INVALID:
  case IRPosition::IRP_ARGUMENT:
  case IRPosition::IRP_CALL_SITE_ARGUMENT:
  case IRPosition::IRP_RETURNED:
  case IRPosition::IRP_CALL_SITE_RETURNED:
  case IRPosition::IRP_CALL_SITE:
  case IRPosition::IRP_FLOAT:
    llvm_unreachable(
        "AAToARTS can only be created for function/float position!");
    break;
  case IRPosition::IRP_FUNCTION:
    AA = new (A.Allocator) AAToARTSFunction(IRP, A);
    break;
  }
  return *AA;
}

const char AAToARTS::ID = 0;

/// ARTS TRANSFORM
bool ARTSTransformer::run(Attributor &A) {
  bool Changed = false;
  /// The process start in the main function, we which will converted to an EDT.
  Function *MainF = M.getFunction("main");
  if (!MainF) {
    LLVM_DEBUG(dbgs() << TAG << "Main function not found\n");
    return Changed;
  }
  /// Run AAToARTS on the main function
  LLVM_DEBUG(dbgs() << TAG
                    << "Initializing AAToARTS attributor for main function\n");
  A.getOrCreateAAFor<AAToARTS>(IRPosition::function(*MainF),
                               /* QueryingAA */ nullptr, DepClassTy::NONE,
                               /* ForceUpdate */ false,
                               /* UpdateAfterInit */ true);
  Changed |= runAttributor(A);

  LLVM_DEBUG(dbgs() << TAG << "\nEDTs INFORMATION\n");
  LLVM_DEBUG(dbgs() << TAG << EDTs << "\n");
  return Changed;
}

bool ARTSTransformer::runAttributor(Attributor &A) {
  LLVM_DEBUG(dbgs() << TAG << "[Attributor] Process started\n");
  ChangeStatus Changed = A.run();
  LLVM_DEBUG(dbgs() << TAG << "[Attributor] Done, result: " << Changed
                    << ".\n");

  return Changed == ChangeStatus::CHANGED;
}

/// ARTS TRANSFORMATION PASS
PreservedAnalyses ARTSTransformPass::run(Module &M, ModuleAnalysisManager &AM) {
  /// Command line options
  if (!EnableARTSTransformation)
    return PreservedAnalyses::all();
  if (PrintModuleBeforeOptimizations)
    LLVM_DEBUG(dbgs() << TAG << "Module before ARTSTransformer Module Pass:\n"
                      << M);

  /// Run the ARTSTransformer Module Pass
  LLVM_DEBUG(dbgs() << TAG << "Run the ARTSTransformer Module Pass\n");
  bool Changed = false;

  /// Get the set of functions in the module
  SetVector<Function *> Functions;
  for (Function &F : M) {
    if (F.isDeclaration() && !F.hasLocalLinkage())
      continue;
    Functions.insert(&F);
  }

  /// If there are no functions, we are done.
  if (Functions.empty())
    return PreservedAnalyses::none();

  /// Create attributor
  FunctionAnalysisManager &FAM =
      AM.getResult<FunctionAnalysisManagerModuleProxy>(M).getManager();
  AnalysisGetter AG(FAM);
  CallGraphUpdater CGUpdater;
  BumpPtrAllocator Allocator;
  ARTSInformationCache InfoCache(M, AG, Allocator, &Functions);
  AttributorConfig AC(CGUpdater);
  AC.IsModulePass = true;
  AC.DefaultInitializeLiveInternals = false;
  AC.DeleteFns = false;
  AC.RewriteSignatures = false;
  Attributor A(Functions, InfoCache, AC);

  /// Run ARTSTransform
  Changed |= InfoCache.ARTSTransform.run(A);

  LLVM_DEBUG(dbgs() << TAG << "Module after ARTSTransformer Module Pass:\n"
                    << M);
  if (Changed)
    return PreservedAnalyses::none();
  return PreservedAnalyses::all();
}

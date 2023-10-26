
#include "llvm/Transforms/ARTS/ARTSTransform.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/CallGraphSCCPass.h"
#include "llvm/Analysis/MemoryDependenceAnalysis.h"
#include "llvm/Analysis/MemorySSA.h"
#include "llvm/Analysis/MemorySSAUpdater.h"
#include "llvm/Analysis/RegionInfo.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/Frontend/ARTS/ARTSIRBuilder.h"

#include "llvm/IR/Attributes.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/Debug.h"
#include "llvm/Transforms/IPO/Attributor.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/CallGraphUpdater.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/CodeExtractor.h"

#include <cstdint>
#include <optional>
using namespace llvm;

/// DEBUG
#define DEBUG_TYPE "arts-transform"
#if !defined(NDEBUG)
static constexpr auto TAG = "[" DEBUG_TYPE "] ";
#endif

/// USING DIRECTIVES
using BlockSequence = SmallVector<BasicBlock *, 0>;

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
                       CallGraphUpdater &CGUpdater, BumpPtrAllocator &Allocator,
                       SetVector<Function *> *Functions, ARTSTransformer &ARTSTransform)
      : InformationCache(M, AG, Allocator, Functions), AG(AG),
        CGUpdater(CGUpdater), Functions(*Functions), ARTSTransform(ARTSTransform),
        ARTSBuilder(M) {

    ARTSBuilder.initialize();
  }

  /// Getters for analysis.
  AnalysisGetter &AG;
  /// Call Graph
  CallGraphUpdater &CGUpdater;
  /// Module functions
  SetVector<Function *> &Functions;
  /// ARTSTransformer pointer
  ARTSTransformer &ARTSTransform;
  /// The ARTSIRBuilder instance.
  ARTSIRBuilder ARTSBuilder;
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
        DE.Shareds.insert(Arg);
      /// If not, it is a first private variable
      else
        DE.FirstPrivates.insert(Arg);

      /// NOTES: The outline function or the variable should have attributes
      /// that provide more information about the lifetime of the variable. It
      /// is also important to consider the underlying type of the variable.
      /// There may be cases, where there is a firstprivate variable that is a
      /// pointer. For this case, the pointer is private, but the data it points
      /// to is shared.
      ///
      /// Do we need to consider private variables?
    }
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
    /// Get context and module
    LLVMContext &Ctx = CB.getContext();
    Module *M = CB.getModule();
    /// Get the size of the kmp_task_t struct
    auto *TaskStruct = dyn_cast<StructType>(CB.getType());
    auto SharedsSize = M->getDataLayout().getTypeAllocSize(
        TaskStruct->getTypeByName(Ctx, "struct.kmp_task_t"));
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
          LLVM_DEBUG(dbgs()
                      << "[" << Range.Offset << "-" << Range.Offset +
                      Range.Size
                      << "] : " << AccIndex.size() << "\n");
          for (auto AI : AccIndex) {
            auto &Acc = AAPI->getAccess(AI);
            auto AccKind = Acc.getKind();
            auto *Inst = Acc.getLocalInst();

            if (Acc.isWrittenValueYetUndetermined()) {
              return false;
            }
            /// Store instruction.
            if (AccKind == AAPointerInfo::AccessKind::AK_MUST_WRITE) {
              Value *Val = Acc.getWrittenValue();
              LLVM_DEBUG(dbgs() << "    - Written value: " << *Val << "\n");
              if (!Val)
                return false;
              /// Check if it is a shared variable
              if (Range.Offset == 0) {
                DE.Shareds.insert(Val);
              }
              /// Check if it is a firstprivate variable
              else if ((uint64_t)Range.Offset >= (uint64_t)SharedsSize) {
                DE.FirstPrivates.insert(Val);
              }
            }
            /// Read instruction
            else if (AccKind == AAPointerInfo::AccessKind::AK_MUST_READ) {
              if (auto *LI = dyn_cast<LoadInst>(Inst)) {
                Value *Val = LI->getPointerOperand();
                LLVM_DEBUG(dbgs() << "    - Read value: " << *Val << "\n");
                auto *PIL = A.getAAFor<AAPointerInfo>(
                    *this, IRPosition::value(*LI), DepClassTy::OPTIONAL);
                if (!PIL->getState().isValidState()) {
                  indicatePessimisticFixpoint();
                  return false;
                }
                /// Is it a fixpoint?
                if (!PIL->getState().isAtFixpoint()) {
                  return false;
                }
                /// For all offset bins in the PI, run AccessCB
                if (!PIL->forallOffsetBins(AccessCB)) {
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
    LLVM_DEBUG(dbgs() << "-- " << CB.getCalledFunction()->getName()
                      << ": \n" << DE);
    /// If we reach this point, it means that the PI is valid and at fixpoint.
    indicateOptimisticFixpoint();
    return ChangeStatus::CHANGED;
  }

  ChangeStatus handleTaskwait() { return ChangeStatus::UNCHANGED; }

  void initialize(Attributor &A) override {
    CallBase &CB = cast<CallBase>(getAssociatedValue());
    Function *Callee = getAssociatedFunction();
    LLVM_DEBUG(dbgs() << "[AADataEnvCallSite] initialize: " << Callee->getName()
                      << "\n");
    /// Handle in the updateImpl function
    // switch (OMPInfo::getRTFunction(Callee)) {
    // case OMPInfo::SET_NUM_THREADS: {
    //   auto *Arg = CB.getArgOperand(0);
    //   if (auto *NumThreads = dyn_cast<ConstantInt>(Arg)) {
    //     LLVM_DEBUG(dbgs() << "Number of threads: " << NumThreads->getZExtValue()
    //                       << "\n");
    //   }
    //   indicateOptimisticFixpoint();
    // } break;
    // case OMPInfo::PARALLEL:
    //   /// Handle it in the updateImpl function
    //   break;
    // case OMPInfo::TASKALLOC: {
    //   /// The task is created by the taskalloc function, and it returns a
    //   /// pointer to the task (check handleTaskRegion documentation ).
    //   /// We need to obtain its data environment. Since this pointer is used in
    //   /// functions that are not analyzable, and for purposes of this analysis
    //   /// we are not interested on what happens inside those functions, we will
    //   /// simply remove it. A pointer is captured by the call if it
    //   /// makes a copy of any part of the pointer that outlives the call.
    //   for (auto &U : CB.uses()) {
    //     auto *User = U.getUser();
    //     if (!isa<Instruction>(User))
    //       continue;
    //     if (auto *CBI = dyn_cast<CallInst>(User)) {
    //       /// Remove the call instruction, we dont need it
    //       if (OMPInfo::isTaskFunction(CBI->getCalledFunction())) {
    //         CBI->eraseFromParent();
    //       }
    //     }
    //   }
    //   /// Lets handle the task in the updateImpl function
    // } break;
    // case OMPInfo::OTHER: {
    //   LLVM_DEBUG(dbgs() << "Other instruction - FOUND\n");
    //   /// Unknown caller or declarations are not analyzable, we give up.
    //   if (!Callee || !A.isFunctionIPOAmendable(*Callee)) {
    //     indicatePessimisticFixpoint();
    //     LLVM_DEBUG(dbgs() << "Unknown caller or declarations are not "
    //                          "analyzable, we give up.\n");
    //     return;
    //   }
    // } break;
    // /// Default
    // default: {
    //   LLVM_DEBUG(dbgs() << "Other instruction - FOUND\n");
    // } break;
    // }
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

    // if (Changed == ChangeStatus::CHANGED) {
    //   /// If the data environment is updated, update info in the cache
    //   auto &AIC = static_cast<ARTSInformationCache &>(A.getInfoCache());
    //   auto *EB = AIC.ARTSTransform.getEDTBlock(CB.getParent());
    //   // auto &OMP = EB->OMP;
    //   // OMP.DE.append(DE);
    //   // OMP.F = OutlinedFunction;
    // }
    return Changed;
  }

  ChangeStatus manifest(Attributor &A) override {
    ChangeStatus Changed = ChangeStatus::UNCHANGED;
    LLVM_DEBUG(dbgs() << "[AADataEnvCallSite] Manifest\n");
    return Changed;
  }

  /// Data environment of the Region
  DataEnv DE;
  /// Given a value it maps it to its access (read/write, offsets and size)
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

  void initialize(Attributor &A) override {
    Function *F = getAnchorScope();
    LLVM_DEBUG(dbgs() << "\n[AAToARTSFunction] initialize: " << F->getName()
                      << "\n");
    // /// Local attributes
    // AIT = &static_cast<ARTSInformationCache &>(A.getInfoCache());
    // AT = &AIT->ARTSTransform;
    // AG = &AIT->AG;

    // /// Identify
    // if (!identifyEDTs(*F, A)) {
    //   indicatePessimisticFixpoint();
    //   return;
    // }
    // LLVM_DEBUG(dbgs() << *F << "\n");
    /// The rest is handled in the updateImpl function
  }

  ChangeStatus updateImpl(Attributor &A) override {
    Function *F = getAnchorScope();
    LLVM_DEBUG(dbgs() << "[AAToARTSFunction] updateImpl: " << F->getName()
                      << "\n");
    // auto &ARTSInfoCache = static_cast<ARTSInformationCache &>(A.getInfoCache());
    // auto &AT = ARTSInfoCache.ARTSTransform;

    // /// Iterate through the EDTs to analyze the EDT with empty number of
    // Blocks LLVM_DEBUG(dbgs() << "\n" << TAG << "EDTs with empty number of
    // Blocks:\n"); for (auto *EDT : AT.EDTsFromFunction[F]) {
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
    //   auto *OMPFAA = A.getAAFor<AAToARTS>(*this,
    //   IRPosition::function(*OMP.F),
    //                                       DepClassTy::OPTIONAL);
    //   if (!OMPFAA->getState().isValidState()) {
    //     LLVM_DEBUG(dbgs() << "[AAToARTSFunction] OMPFAA is invalid\n");
    //     indicatePessimisticFixpoint();
    //     return ChangeStatus::CHANGED;
    //   }
    //   if (!OMPFAA->getState().isAtFixpoint()) {
    //     LLVM_DEBUG(dbgs() << "[AAToARTSFunction] OMPFAA is not at
    //     fixpoint\n"); return ChangeStatus::UNCHANGED;
    //   }
    // }

    // /// If this is the main function
    // if (F->getName() == "main") {
    //   LLVM_DEBUG(dbgs() << TAG
    //                     << "----------- PROCESS HAS FINISHED -----------\n");
    //   EDTInfo *MainEDT = AT.getEDTForFunction(F);
    //   assert(MainEDT && "Main EDT not found");
    //   auto &ARTSBuilder = ARTSInfoCache.ARTSBuilder;

    //   LLVM_DEBUG(dbgs() << TAG << "----------- EDTs INFORMATION
    //   -----------\n"); LLVM_DEBUG(dbgs() << *MainEDT << "\n"); Function
    //   *MainEDTFunction = ARTSBuilder.createEDT("main.edt");
    //   MainEDT->setF(MainEDTFunction);
    //   /// Iterate through the EDT blocks
    //   for(auto *EB : MainEDT->Blocks) {
    //     /// Entry and Other blocks are inmmediately added to the main EDT
    //     Function if(EB->isEntry() || EB->isOther()) {
    //       ARTSBuilder.insertEDTBlock(EB, MainEDTFunction);
    //       continue;
    //     }
    //     /// If it is an init block, we need to create a new EDT function for
    //     the
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

struct ARTSAnalyzer {
  ARTSAnalyzer(ARTSTransformer &AT, FunctionAnalysisManager &FAM, AnalysisGetter &AG)
      : AT(AT), FAM(FAM), AG(AG) {}

  //// Region functions
  void visitRegion(Region &R) {
    LLVM_DEBUG(dbgs() << TAG << "Visiting region: " << R << "\n");
    printRegion(R);
    for (auto &Itr : R) {
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
    if (ExitBB)
      LLVM_DEBUG(dbgs() << "Exit: " << ExitBB->getName() << "\n");
    else
      LLVM_DEBUG(dbgs() << "Exit: <Function Return>\n");
  }

  bool identifyRegions(Function &F) {
    auto *RI = AG.getAnalysis<RegionInfoAnalysis>(F);
    auto *Region = RI->getTopLevelRegion();
    visitRegion(*Region);
    return true;
  }

  /// This function finds the inputs and outputs of a region.
  void getInputsOutputs(BlockSequence Region, DominatorTree *DT,
                        SetVector<Value *> &Inputs,
                        SetVector<Value *> &Outputs,
                        SetVector<Value *> &Sinks) {
    Function &F = *Region.front()->getParent();
    AssumptionCache *AC = AG.getAnalysis<AssumptionAnalysis>(F, true);
    CodeExtractorAnalysisCache CEAC(F);
    CodeExtractor CE(Region, DT, /* AggregateArgs */ false, /* BFI */ nullptr,
                     /* BPI */ nullptr, AC, /* AllowVarArgs */ false,
                     /* AllowAlloca */ true, /* AllocaBlock */ nullptr,
                     /* Suffix */ "");

    // SetVector<Value *> Inputs, Outputs, Sinks;
    CE.findInputsOutputs(Inputs, Outputs, Sinks);
    LLVM_DEBUG(dbgs() << "  - Inputs: \n");
    for (auto *I : Inputs) {
      LLVM_DEBUG(dbgs() << "    -- " <<  *I << "\n");
    }
    LLVM_DEBUG(dbgs() << "  - Outputs: \n");
    for (auto *O : Outputs) {
      LLVM_DEBUG(dbgs() << "  -- " << *O << "\n");
    }
    LLVM_DEBUG(dbgs() << "  - Sinks: \n");
    for (auto *S : Sinks) {
      LLVM_DEBUG(dbgs() << "    -- " << *S << "\n");
    }
  }

  /// It finds the BBs that are dominated by FromBB and add
  /// them to the DominatedBlocks vector.
  void dominatedBBs(BasicBlock *FromBB, DominatorTree &DT,
                    BlockSequence &DominatedBlocks) {
    Function &F = *FromBB->getParent();
    for (auto &ToBB : F) {
      if (DT.dominates(FromBB, &ToBB))
        DominatedBlocks.push_back(&ToBB);
    }
  }

  /// Remove values interface
  void removeValue(Value *V) {
    if (isa<UndefValue>(V))
      return;
    /// Instructions
    if(auto *I = dyn_cast<Instruction>(V)) {
        if (auto *CBI = dyn_cast<CallBase>(I)) {
          LLVM_DEBUG(dbgs() << "   - Removing call instruction: " << *CBI << "\n");
          /// Iterate through the arguments and replace them with undef using int itr
          for (uint32_t ArgItr = 0; ArgItr < CBI->data_operands_size(); ArgItr++) {
            Value *Arg = CBI->getArgOperand(ArgItr);
            if (!isa<PointerType>(Arg->getType()))
              continue;

            LLVM_DEBUG(dbgs() << "    - Arg: " << *Arg << "\n");
            // replaceValueWithUndef(Arg, false);
            removeValue(Arg);
          }
        }
      LLVM_DEBUG(dbgs() << "   - Removing instruction: " << *I << "\n");
      replaceValueWithUndef(I, false);
      I->eraseFromParent();
      LLVM_DEBUG(dbgs() << "    - Instruction removed\n");
      return;
    }

    replaceValueWithUndef(V, false);
    /// Global variables are not instructions, but we still need to remove them
    if (GlobalVariable *GV = dyn_cast<GlobalVariable>(V)) {
      LLVM_DEBUG(dbgs() << "   - Removing global variable: " << *GV << "\n");
      GV->eraseFromParent();
      return;
    }

    /// Function
    if (Function *F = dyn_cast<Function>(V)) {
      LLVM_DEBUG(dbgs() << "   - Removing function: " << *F << "\n");
      F->eraseFromParent();
      return;
    }
  }

  void removeValues() {
    LLVM_DEBUG(dbgs() << "\n" << TAG << "Removing values\n");
    for (auto *V : ValuesToRemove) {
      removeValue(V);
    }
  }
  /// Function to iteratively replace uses of a Value with UndefValue
  /// and remove instructions if requested.
  void replaceValueWithUndef(Value *V, bool RemoveInsts = false) {
    /// If the value is undef, we dont need to do anything
    if (isa<UndefValue>(V)) {
      // LLVM_DEBUG(dbgs() << "Value is undef\n");
      return;
    }
    // Create a worklist to keep track of instructions to be removed
    SmallVector<Instruction *, 16> Worklist;
    // Initialize the worklist with all uses of the argument
    for (auto &Use : V->uses()) {
      if (Instruction *UserInst = dyn_cast<Instruction>(Use.getUser())) {
        LLVM_DEBUG(dbgs() << "    - Use: " << *UserInst << "\n");
        Worklist.push_back(UserInst);
      }
    }
    V->replaceAllUsesWith(UndefValue::get(V->getType()));
    // Replace uses with UndefValue and mark instructions for removal
    while (!Worklist.empty()) {
      Instruction *Inst = Worklist.pop_back_val();
      // LLVM_DEBUG(dbgs() << ". Replacing: " << *Inst << "\n");
      // Add users of this instruction to the worklist for further processing
      for (auto &Use : Inst->uses()) {
        if (Instruction *UserInst = dyn_cast<Instruction>(Use.getUser())) {
          Worklist.push_back(UserInst);
        }
      }
      // Replace uses of the argument with UndefValue
      Value *Undef = UndefValue::get(Inst->getType());
      Inst->replaceAllUsesWith(Undef);
      // Mark the instruction for removal
      if(RemoveInsts)
        Inst->eraseFromParent();
        // removeValue(Inst);
    }

    /// Remove the value itself
    // if(RemoveInsts)
    //   removeValue(V);
  }

  /// Given a basic block, this function creates a function that contains
  /// the instructions of the BB. The function is then inserted to the
  /// module.
  Function *
  createFunction(DominatorTree *DT, BasicBlock *FromBB, bool DTAnalysis = false,
                 SmallVector<Value *, 0> *ExcludeArgsFromAggregate = nullptr) {
    Function &F = *FromBB->getParent();
    AssumptionCache *AC = AG.getAnalysis<AssumptionAnalysis>(F, true);
    CodeExtractorAnalysisCache CEAC(F);
    /// Collect blocks
    BlockSequence Region;
    /// Get all BBs that are dominated by FromBB
    if (DTAnalysis)
      dominatedBBs(FromBB, *DT, Region);
    else
      Region.push_back(FromBB);
    LLVM_DEBUG(dbgs() << TAG << "Creating function for: " << FromBB->getName()
                      << " - " << Region.size() << "\n");
    /// Extract code from the region
    CodeExtractor CE(Region, DT, /* AggregateArgs */ false, /* BFI */ nullptr,
                      /* BPI */ nullptr, AC, /* AllowVarArgs */ false,
                      /* AllowAlloca */ true, /* AllocaBlock */ nullptr,
                      /* Suffix */ "edt");
    assert(CE.isEligible() && "Expected Region outlining to be possible!");

    if (ExcludeArgsFromAggregate)
      for (auto *V : *ExcludeArgsFromAggregate)
        CE.excludeArgFromAggregate(V);
    /// Generate function
    Function *OutF = CE.extractCodeRegion(CEAC);
    /// Add function to the attribute cache and update call graph
    // AIT->Functions.insert(OutF);
    // AIT->CGUpdater.registerOutlinedFunction(F, *OutF);
    // AIT->CGUpdater.reanalyzeFunction(F);
    return OutF;
  }

  /// Analyzes outlined region, replaces the RT call with a call to the
  /// outlined function, which is also modified to remove the arguments that
  /// are not needed.
  Function *analyzeOutlinedRegion(CallBase *RTCall,
                                  uint32_t OutlinedFunctionPos,
                                  uint32_t KeepArgsFrom = 0,
                                  uint32_t KeepCallArgsFrom = 0) {
    Function *OutlinedFn =
        dyn_cast<Function>(RTCall->getArgOperand(OutlinedFunctionPos));
    if (KeepArgsFrom > 0) {
      /// Get private and shared variables
      for (unsigned int ArgItr = 0; ArgItr < KeepArgsFrom; ArgItr++) {
        Value *Arg = OutlinedFn->args().begin() + ArgItr;
        // LLVM_DEBUG(dbgs() << "Removing argument: " << *Arg << "\n");
        /// Check all uses of the argument
        for (auto &U : Arg->uses()) {
          // LLVM_DEBUG(dbgs() << "  -Use: " << *U.getUser() << "\n");
          /// Convert to instruction
          auto *I = dyn_cast<Instruction>(U.getUser());
          if (!I)
            continue;
          /// If the argument is used by a call instruction, replace it with a
          /// nullptr
          if (auto *CBI = dyn_cast<CallBase>(I)) {
            CBI->setArgOperand(U.getOperandNo(),
                               UndefValue::get(Arg->getType()));
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

    auto *OldFn = OutlinedFn;

    SmallVector<Type *, 16> NewArgumentTypes;
    SmallVector<AttributeSet, 16> NewArgumentAttributes;

    // Collect replacement argument types and copy over existing attributes.
    AttributeList OldFnAttributeList = OldFn->getAttributes();
    for (auto ArgItr = KeepArgsFrom; ArgItr < OldFn->arg_size(); ArgItr++) {
      Value *Arg = OldFn->args().begin() + ArgItr;
      NewArgumentTypes.push_back(Arg->getType());
      NewArgumentAttributes.push_back(OldFnAttributeList.getParamAttrs(ArgItr));
    }

    FunctionType *OldFnTy = OldFn->getFunctionType();
    Type *RetTy = OldFnTy->getReturnType();

    // Construct the new function type using the new arguments types.
    FunctionType *NewFnTy =
        FunctionType::get(RetTy, NewArgumentTypes, OldFnTy->isVarArg());
    LLVM_DEBUG(dbgs() << " - Function rewrite '" << OldFn->getName()
                      << "' from " << *OldFn->getFunctionType() << " to "
                      << *NewFnTy << "\n");

    // Create the new function body and insert it into the module.
    Function *NewFn = Function::Create(NewFnTy, OldFn->getLinkage(),
                                       OldFn->getAddressSpace(), "");
    OldFn->getParent()->getFunctionList().insert(OldFn->getIterator(), NewFn);
    NewFn->takeName(OldFn);
    NewFn->setName("parallel.edt");
    // NewFn->copyAttributesFrom(OldFn);
    OldFn->setSubprogram(nullptr);
    OldFn->setMetadata("dbg", nullptr);

    LLVMContext &Ctx = OldFn->getContext();
    // NewFn->setAttributes(AttributeList::get(
    //     Ctx, OldFnAttributeList.getFnAttrs(), OldFnAttributeList.getRetAttrs(),
    //     NewArgumentAttributes));
    // NewFn->setMemoryEffects(MemoryEffects::argMemOnly());

    // Since we have now created the new function, splice the body of the old
    // function right into the new function, leaving the old rotting hulk of the
    // function empty.
    NewFn->splice(NewFn->begin(), OldFn);
    // LLVM_DEBUG(dbgs() << "[ARTS] New function: \n" << *NewFn);

    // Fixup block addresses to reference new function.
    // SmallVector<BlockAddress *, 8u> BlockAddresses;
    // for (User *U : OldFn->users()) {
    //   if (auto *BA = dyn_cast<BlockAddress>(U))
    //     BlockAddresses.push_back(BA);
    // }
    // for (auto *BA : BlockAddresses)
    //   BA->replaceAllUsesWith(BlockAddress::get(NewFn, BA->getBasicBlock()));

    /// Replace callsite
    CallBase *OldCB = dyn_cast<CallBase>(RTCall);
    const AttributeList &OldCallAttributeList = OldCB->getAttributes();

    // Collect the new argument operands for the replacement call site.
    SmallVector<Value *, 0> NewCallArgs;
    SmallVector<Value *, 16> NewArgOperands;
    SmallVector<AttributeSet, 16> NewArgOperandAttributes;
    for (unsigned OldArgNum = KeepCallArgsFrom;
         OldArgNum < OldCB->data_operands_size(); ++OldArgNum) {
      NewArgOperands.push_back(OldCB->getArgOperand(OldArgNum));
      NewArgOperandAttributes.push_back(
          OldCallAttributeList.getParamAttrs(OldArgNum));
    }

    // Create a new call or invoke instruction to replace the old one.
    auto *NewCI =
        CallInst::Create(NewFn, NewArgOperands, std::nullopt, "", OldCB);
    NewCI->setTailCallKind(cast<CallInst>(OldCB)->getTailCallKind());

    // Copy over various properties and the new attributes.
    CallBase *NewCB = NewCI;
    NewCB->setCallingConv(OldCB->getCallingConv());
    NewCB->takeName(OldCB);
    NewCB->setAttributes(AttributeList::get(
        Ctx, OldCallAttributeList.getFnAttrs(),
        OldCallAttributeList.getRetAttrs(), NewArgOperandAttributes));
    // LLVM_DEBUG(dbgs() << "- New call: " << *NewCB << "\n");

    // Rewire the function arguments.
    Argument *OldFnArgIt = OldFn->arg_begin() + KeepArgsFrom;
    Argument *NewFnArgIt = NewFn->arg_begin();
    for (unsigned OldArgNum = KeepArgsFrom; OldArgNum < OldFn->arg_size();
         ++OldArgNum, ++OldFnArgIt) {
      NewFnArgIt->takeName(&*OldFnArgIt);
      OldFnArgIt->replaceAllUsesWith(&*NewFnArgIt);
      ++NewFnArgIt;
    }

    // Eliminate the instructions *after* we visited all of them.
    OldCB->replaceAllUsesWith(NewCB);
    OldCB->eraseFromParent();
    ValuesToRemove.push_back(OldFn);

    assert(NewFn->isDeclaration() == false && "New function is a declaration");
    return NewFn;
  }

  bool handleParallelOutlinedRegion(CallBase *CB) {
    // AT.insertCB(CB);
    /// Analyze outlined region
    const uint32_t ParallelOutlinedFunctionPos = 2;
    const uint32_t KeepArgsFrom = 2;
    const uint32_t KeepCallArgsFrom = 3;
    Function *OutlinedFunction = analyzeOutlinedRegion(
        CB, ParallelOutlinedFunctionPos, KeepArgsFrom, KeepCallArgsFrom);
    if(!identifyEDTs(*OutlinedFunction))
      return false;
    return true;
  }

  bool handleTaskOutlinedRegion(CallBase *CB) {
    AT.insertCB(CB);
    /// Analyze outlined region
    const uint32_t TaskOutlinedFunctionPos = 5;

    /// Analyze return pointer
    // For the shared variables we are interested in all stores that are done
    // to the shareds field of the kmp_task_t struct. For the firstprivate
    // variables we are interested in all stores that are done to the privates
    // field of the kmp_task_t_with_privates struct.
    //
    // The returned Val is a pointer to the
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
    // The function returns ChangeStatus::CHANGED if the data environment is
    // updated, ChangeStatus::UNCHANGED otherwise.

    OMPInfo OI(OMPInfo::TASK, CB);
    /// Maps a value to an offset in the task data
    DenseMap<Value *, int64_t> ValueToOffsetTD;
    /// Maps an offset to a value in the Outlined Function
    DenseMap<int64_t, Value *> OffsetToValueOF;
  
    /// Get context and module
    const DataLayout &DL = CB->getModule()->getDataLayout();
    LLVMContext &Ctx = CB->getContext();

    /// Get the size of the kmp_task_t struct
    const auto *TaskStruct = dyn_cast<StructType>(CB->getType());
    const auto TaskDataSize = static_cast<int64_t>(DL.getTypeAllocSize(
        TaskStruct->getTypeByName(Ctx, "struct.kmp_task_t")));

    /// Analyze Task Data
    /// This analysis assumes we only have stores to the task struct
    BasicBlock *BB = CB->getParent();
    // LLVM_DEBUG(dbgs() << "TASKDATA ANALYSIS\n");
    for (Instruction &I : *BB) {
      if (&I == CB)
        continue;

      if (!isa<StoreInst>(&I))
        continue;

      int64_t Offset = -1;
      auto *S = cast<StoreInst>(&I);
      GetPointerBaseWithConstantOffset(S->getPointerOperand(), Offset, DL);
      auto *Val = S->getValueOperand();
      ValueToOffsetTD[Val] = Offset;

      // LLVM_DEBUG({
      //   dbgs() << "  - Store: " << *S << "\n";
      //   dbgs() << "    - ValOperand: " << *(S->getValueOperand()) << "\n";
      //   dbgs() << "    - Offset: " << Offset << "\n";
      // });

      /// Private variables
      if(Offset > TaskDataSize) {
        OI.DE.Privates.insert(Val);
        continue;
      }
      /// Shared variables
      OI.DE.Shareds.insert(Val);
    }

    /// Analyze Outlined Function
    Function *OutlinedFn =
      dyn_cast<Function>(CB->getArgOperand(TaskOutlinedFunctionPos));
    Argument *TaskData = dyn_cast<Argument>(OutlinedFn->arg_begin() + 1);
    /// This assumes the 'desaggregation' happens in the first basic block
    BasicBlock *EntryBB = &OutlinedFn->getEntryBlock();
    BasicBlock::iterator Itr = EntryBB->begin();
    auto *TaskDataPtr = &*Itr;
    // LLVM_DEBUG({
    //   dbgs() << "OUTLINED FUNCTION ANALYSIS\n";
    //   dbgs() << "  - TaskData: " << *TaskData << "\n";
    //   dbgs() << "  - TaskDataPtr: " << *TaskDataPtr << "\n";
    // });
    // Iterate from the second instruction onwards
    ++Itr;
    for (; Itr != BB->end(); ++Itr) {
      Instruction &I = *Itr;
      /// We are only concerned about load instructions from the task data
      if(!isa<LoadInst>(&I))
        continue;
      auto *L = cast<LoadInst>(&I);
      auto *Val = L->getPointerOperand();
      int64_t Offset = -1;
      auto *BasePointer = GetPointerBaseWithConstantOffset(Val, Offset, DL);
      // GetPointerBaseWithConstantOffset(Val, Offset, DL);
      if(Offset == -1)
        continue;
      /// Private variables
      auto Cond = (Offset >= TaskDataSize && BasePointer == TaskData) || 
                        (Offset <  TaskDataSize && BasePointer == TaskDataPtr);
      if(!Cond) 
        continue;
      // LLVM_DEBUG({
      //   dbgs() << "  - Load: " << *L << "\n";
      //   dbgs() << "    - ValOperand: " << *Val << "\n";
      //   dbgs() << "    - BasePointer: " << *BasePointer << "\n";
      //   dbgs() << "    - Offset: " << Offset << "\n";
      // });
      OffsetToValueOF[Offset] = L;
      if(OffsetToValueOF.size() == ValueToOffsetTD.size())
        break;
    }

    /// Assert size of value to offset map
    assert(ValueToOffsetTD.size() == OffsetToValueOF.size() &&
            "ValueToOffsetTD and ValueToOffsetOF have different sizes");

    /// Debug ValueToOffsetOF and OffsetToValueTD
    // LLVM_DEBUG({
    //   dbgs() << "ValueToOffsetOF\n";
    //   for(auto Itr : OffsetToValueOF) {
    //     dbgs() << "  - " << Itr.first << " - " << *(Itr.second) << " - Ty: " << *(Itr.second->getType()) << "\n";
    //   }
    //   dbgs() << "OffsetToValueTD\n";
    //   for(auto Itr : ValueToOffsetTD) {
    //     dbgs() << "  - " << Itr.second << " - " << *(Itr.first) << " - Ty: " << *(Itr.first->getType()) << "\n";
    //   }
    // });

    /// Generate new function signature using OffsetToValueTD
    SmallVector<Type *, 16> NewArgumentTypes;
    for(auto Itr : ValueToOffsetTD) {
      auto *V = Itr.first;
      NewArgumentTypes.push_back(V->getType());
    }
    FunctionType *NewFnTy =
        FunctionType::get(Type::getVoidTy(Ctx), NewArgumentTypes, false);
    LLVM_DEBUG(dbgs() << " - Function rewrite '" << OutlinedFn->getName()
                      << "' from " << *OutlinedFn->getFunctionType() << " to "
                      << *NewFnTy << "\n");
    // Create the new function body and insert it into the module.
    Function *NewFn = Function::Create(NewFnTy, OutlinedFn->getLinkage(),
                                       OutlinedFn->getAddressSpace());
    OutlinedFn->getParent()->getFunctionList().insert(
      OutlinedFn->getIterator(), NewFn);
    NewFn->takeName(OutlinedFn);
    NewFn->setName("task.edt");
    OutlinedFn->setSubprogram(nullptr);
    OutlinedFn->setMetadata("dbg", nullptr);

    /// Move BB to new function
    NewFn->splice(NewFn->begin(), OutlinedFn);
    /// Rewire the function arguments.
    Argument *NewFnArgItr = NewFn->arg_begin();
    for (auto TDItr : ValueToOffsetTD) {
      Value *V = OffsetToValueOF[TDItr.second];
      // LLVM_DEBUG(dbgs() << " - Rewiring argument: " << *V << " to " << *NewFnArgItr << "\n");
      V->replaceAllUsesWith(NewFnArgItr);
      ++NewFnArgItr;
    }

    /// Generate CallInst to NewFn
    SmallVector<Value *, 0> NewCallArgs;
    for (auto TDItr : ValueToOffsetTD) {
      NewCallArgs.push_back(TDItr.first);
    }

    auto *LastInstruction = BB->getTerminator();
    auto *NewCI =
        CallInst::Create(NewFn, NewCallArgs, std::nullopt, "", LastInstruction);
    NewCI->setTailCallKind(cast<CallInst>(CB)->getTailCallKind());

    /// Remove Argument 0 and 1 from Original Task Outlined Function
    replaceValueWithUndef(OutlinedFn->getArg(0), true);
    replaceValueWithUndef(OutlinedFn->getArg(1), true);

    /// Remove Values
    ValuesToRemove.push_back(CB);
    // ValuesToRemove.push_back(OutlinedFn);

    // Iterate through the basic blocks and check/replace terminators
    for (auto &NewBB : *NewFn) {
      auto *Terminator = NewBB.getTerminator();
      if (isa<ReturnInst>(Terminator)) {
        IRBuilder<> Builder(Terminator);
        Builder.CreateRetVoid();
        Terminator->eraseFromParent();
      }
    }

    // LLVM_DEBUG(dbgs() << " - New function: \n" << *NewFn);

    // if(!identifyEDTs(*NewFn))
      return false;
    return true;
  }

  /// This function identifies the EDTs in the function
  bool identifyEDTs(Function &F) {
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
    DominatorTree *DT = AG.getAnalysis<DominatorTreeAnalysis>(F);

    /// Region counter
    unsigned int ParallelRegion = 0;
    unsigned int TaskRegion = 0;

    LLVM_DEBUG(dbgs() << TAG << "Identifying EDT regions for: " << F.getName() << "\n");

    /// If function is not IPO amendable, we give up
    if (F.isDeclaration() && !F.hasLocalLinkage())
      return false;

    /// Get entry block
    BasicBlock *CurrentBB = &(F.getEntryBlock());
    BasicBlock *NextBB = nullptr;
    do {
      NextBB = CurrentBB->getNextNode();
      /// Get first instruction of the function
      Instruction *CurrentI = &(CurrentBB->front());
      do {
        /// We are only interested in call instructions
        auto *CB = dyn_cast<CallBase>(CurrentI);
        if (!CB)
          continue;
        /// Get the callee
        Function *Callee = CB->getCalledFunction();
        OMPInfo::RTFType RTF = OMPInfo::getRTFunction(Callee);
        switch (RTF) {
        case OMPInfo::PARALLEL: {
          LLVM_DEBUG(dbgs() << TAG << "Parallel Region Found: " << "\n  "
                            <<  *CB << "\n");
          /// Split block at __kmpc_parallel
          auto ParallelName = "par.region." + std::to_string(ParallelRegion);
          BasicBlock *ParallelBB =
              SplitBlock(CurrentI->getParent(), CurrentI, DT, LI, nullptr,
                          ParallelName);
          /// Split block at the next instruction
          CurrentI = CurrentI->getNextNonDebugInstruction();
          auto ParallelDoneName =
              "par.done." + std::to_string(ParallelRegion);
          BasicBlock *ParallelDone =
              SplitBlock(ParallelBB, CurrentI, DT, LI, nullptr);
          NextBB = ParallelDone;
          ParallelRegion++;

          handleParallelOutlinedRegion(CB);

          /// Extract regions and outline them into functions
          // Function *ParallelFunction = createFunction(DT, ParallelBB);
          /// For the Done function run the analysis again
          Function *DoneFunction = createFunction(DT, ParallelDone, true);
          DoneFunction->setName("parallel.edt.done");
          /// Get caller of the Done function, and rename BB to par.done
          CallBase *DoneCB = dyn_cast<CallBase>(DoneFunction->user_back());
          BasicBlock *DoneBB = DoneCB->getParent();
          DoneBB->setName(ParallelDoneName);
          /// Analyze the instructions of the Done function
          identifyEDTs(*DoneFunction);
        } break;
        case OMPInfo::TASKALLOC: {
          LLVM_DEBUG(dbgs() << TAG << "Task Region Found: " << "\n  "
                            <<  *CB << "\n");
          /// Split block at __kmpc_omp_task_alloc
          auto TaskName = "task.region." + std::to_string(TaskRegion);
          BasicBlock *TaskBB =
              SplitBlock(CurrentI->getParent(), CurrentI, DT, LI, nullptr,
                         TaskName);
          /// Find the task call
          while ((CurrentI = CurrentI->getNextNonDebugInstruction())) {
            auto *TCB = dyn_cast<CallBase>(CurrentI);
            if (TCB && OMPInfo::getRTFunction(TCB->getCalledFunction()) ==
                            OMPInfo::TASK)
              break;
          }
          assert(CurrentI && "Task RT call not found");
          /// Remove the task call. We dont need it anymore, and this helps in memory analysis
          auto *NextI = CurrentI->getNextNonDebugInstruction();
          CurrentI->eraseFromParent();
          /// Split block again at the next instruction
          CurrentI = NextI;
          auto TaskDoneName =
              "task.done." + std::to_string(TaskRegion);
          BasicBlock *TaskDone =
              SplitBlock(TaskBB, CurrentI, DT, LI, nullptr);
          TaskRegion++;

          handleTaskOutlinedRegion(CB);

          /// Extract regions and create aux functions
          /// For the Done function run the analysis again
          Function *DoneFunction = createFunction(DT, TaskDone, true);
          DoneFunction->setName("task.edt.done");
          identifyEDTs(*DoneFunction);
          CallBase *DoneCB = dyn_cast<CallBase>(DoneFunction->user_back());
          BasicBlock *DoneBB = DoneCB->getParent();
          DoneBB->setName(TaskDoneName);
          NextBB = DoneBB->getNextNode();
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
    return false;
  }

  /// Attributes
  ARTSTransformer &AT;
  FunctionAnalysisManager &FAM;
  AnalysisGetter &AG;
  /// Set of Values to remove
  SmallVector<Value *, 16> ValuesToRemove;
};


void registerAAsForFunction(Attributor &A, const Function &F) {
  if (F.hasFnAttribute(Attribute::Convergent))
    A.getOrCreateAAFor<AANonConvergent>(IRPosition::function(F));

  for (auto &I : instructions(F)) {
    if (auto *LI = dyn_cast<LoadInst>(&I)) {
      bool UsedAssumedInformation = false;
      A.getAssumedSimplified(IRPosition::value(*LI), /* AA */ nullptr,
                             UsedAssumedInformation, AA::Interprocedural);
      continue;
    }
    if (auto *CI = dyn_cast<CallBase>(&I)) {
      if (CI->isIndirectCall())
        A.getOrCreateAAFor<AAIndirectCallInfo>(
            IRPosition::callsite_function(*CI));
    }
    if (auto *SI = dyn_cast<StoreInst>(&I)) {
      A.getOrCreateAAFor<AAIsDead>(IRPosition::value(*SI));
      continue;
    }
    if (auto *FI = dyn_cast<FenceInst>(&I)) {
      A.getOrCreateAAFor<AAIsDead>(IRPosition::value(*FI));
      continue;
    }
    if (auto *II = dyn_cast<IntrinsicInst>(&I)) {
      if (II->getIntrinsicID() == Intrinsic::assume) {
        A.getOrCreateAAFor<AAPotentialValues>(
            IRPosition::value(*II->getArgOperand(0)));
        continue;
      }
    }
  }
}
/// ARTS TRANSFORM

bool ARTSTransformer::run(FunctionAnalysisManager &FAM) {
  bool Changed = false;
  /// The process start in the main function, we which will converted to an EDT.
  Function *MainF = M.getFunction("main");
  if (!MainF) {
    LLVM_DEBUG(dbgs() << TAG << "Main function not found\n");
    return Changed;
  }

  
  AnalysisGetter AG(FAM);
  ARTSAnalyzer Analyzer(*this, FAM, AG);
  Analyzer.identifyEDTs(*MainF);
  Analyzer.removeValues();

  LLVM_DEBUG(dbgs() << TAG << "Module after ARTSAnalyzer:\n"
                      << M);

  /// Get the set of functions in the module
  SetVector<Function *> Functions;
  for (Function &F : M) {
    if (F.isDeclaration() && !F.hasLocalLinkage())
      continue;
    Functions.insert(&F);
  }

  /// Create attributor
  // CallGraphUpdater CGUpdater;
  // BumpPtrAllocator Allocator;
  // ARTSInformationCache InfoCache(M, AG, CGUpdater, Allocator, &Functions, *this);
  // AttributorConfig AC(CGUpdater);
  // AC.IsModulePass = true;
  // AC.DefaultInitializeLiveInternals = true;
  // AC.DeleteFns = true;
  // AC.RewriteSignatures = false;
  // AC.UseLiveness = true;
  // AC.InitializationCallback = registerAAsForFunction;
  // Attributor A(Functions, InfoCache, AC);


  // /// Run AAToARTS on the main function
  // LLVM_DEBUG(dbgs() << TAG
  //                   << "Initializing AAToARTS attributor for main function\n");
  //  /// Analyze data environment for CBs
  // for(auto *CB : CBs) {
  //   A.getOrCreateAAFor<AADataEnv>(
  //     IRPosition::callsite_function(*CB),
  //     /* QueryingAA */ nullptr,
  //     DepClassTy::NONE,
  //     /* ForceUpdate */ false,
  //     /* UpdateAfterInit */ true);
  // }
  // // A.getOrCreateAAFor<AAToARTS>(IRPosition::function(*MainF),
  // //                              /* QueryingAA */ nullptr, DepClassTy::NONE,
  // //                              /* ForceUpdate */ false,
  // //                              /* UpdateAfterInit */ true);
  // Changed |= runAttributor(A);

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

  ARTSTransformer AT(M);
  FunctionAnalysisManager &FAM =
      AM.getResult<FunctionAnalysisManagerModuleProxy>(M).getManager();
  Changed |= AT.run(FAM);
  

  /// Run ARTSTransform
  // Changed |= InfoCache.ARTSTransform.run(FAM);

  LLVM_DEBUG(dbgs() << TAG << "Module after ARTSTransformer Module Pass:\n"
                    << M);
  if (Changed)
    return PreservedAnalyses::none();
  return PreservedAnalyses::all();
}

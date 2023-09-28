
#include "llvm/Transforms/ARTS/ARTSTransform.h"
#include "llvm/Frontend/ARTS/ARTSIRBuilder.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/MemorySSA.h"
#include "llvm/Analysis/MemorySSAUpdater.h"

#include "llvm/Support/Debug.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

#include <optional>
using namespace llvm;

// DEBUG
#define DEBUG_TYPE "arts-transform"
#if !defined(NDEBUG)
static constexpr auto TAG = "[" DEBUG_TYPE "] ";
#endif

/// ---------------------------- COMMAND LINE OPTIONS ---------------------------- ///
static cl::opt<bool> EnableARTSTransformation(
    "arts-enable", cl::desc("Disable transformation to ARTS."),
    cl::Hidden, cl::init(false));

static cl::opt<bool> PrintModuleBeforeOptimizations(
    "arts-print-module-before",
    cl::desc("Print the module before the ARTSTransformer Module Pass"),
    cl::Hidden, cl::init(false));

static cl::opt<bool> PrintModuleAfterOptimizations(
    "arts-print-module-after",
    cl::desc("Print the module after the ARTSTransformer Module Pass"),
    cl::Hidden, cl::init(false));

/// ---------------------------- UTILS ---------------------------- ///


/// ---------------------------- ABSTRACT ATTRIBUTES ---------------------------- ///
/// ARTSInformationCache
/// It stores ARTS related information that the attributor can use
struct ARTSInformationCache : public InformationCache {
  ARTSInformationCache(Module &M, AnalysisGetter &AG,
                       BumpPtrAllocator &Allocator, 
                       SetVector<Function *> *Functions)
      : InformationCache(M, AG, Allocator, Functions),
        AG(AG), ARTSBuilder(M), ARTSTransform(M) {

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

  static AADataEnv &createForPosition(const IRPosition &IRP,
                                            Attributor &A);

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

struct AADataEnvBasicBlock : AADataEnv {
  AADataEnvBasicBlock(const IRPosition &IRP, Attributor &A) : AADataEnv(IRP, A) {}

  const std::string getAsStr(Attributor *A) const override {
    if (!isValidState())
      return "<invalid>";

    std::string Str("AADataEnvBasicBlock: ");
    return Str + "OK";
  }

  void initialize(Attributor &A) override {
    BasicBlock &BB = cast<BasicBlock>(getAnchorValue());
    LLVM_DEBUG(dbgs() <<"\n[AADataEnvBasicBlock] initialize: " 
                      << BB.getName() << "\n");
    /// Let's handle the rest in the updateImpl function
  }

  ChangeStatus updateImpl(Attributor &A) override {
    ChangeStatus Changed = ChangeStatus::UNCHANGED;
    LLVM_DEBUG(dbgs() << "[AADataEnvBasicBlock] updateImpl\n");
    // BasicBlock &BB = cast<BasicBlock>(getAnchorValue());
    return ChangeStatus::UNCHANGED;
  }

  ChangeStatus manifest(Attributor &A) override {
    ChangeStatus Changed = ChangeStatus::UNCHANGED;
    LLVM_DEBUG(dbgs() << "[AADataEnvBasicBlock] Manifest\n");
    return Changed;
  }
};

struct AADataEnvFunction : AADataEnv {
  AADataEnvFunction(const IRPosition &IRP, Attributor &A) : AADataEnv(IRP, A) {}

  const std::string getAsStr(Attributor *A) const override {
    if (!isValidState())
      return "<invalid>";

    std::string Str("AADataEnvFunction: ");
    return Str + "OK";
  }

  void initialize(Attributor &A) override {
    Function &F = cast<Function>(getAnchorValue());
    LLVM_DEBUG(dbgs() <<"\n[AADataEnvFunction] initialize: " 
                      << F.getName() << "\n");
    /// Let's handle the rest in the updateImpl function
  }

  bool handleEDTInitBlock(Attributor &A, EDTBlock &EB) {
    LLVM_DEBUG(dbgs() << "[AADataEnvFunction] EDT init block - FOUND\n");
    LLVM_DEBUG(dbgs() << "  - EDT callbase: " << EB.OMP.getCBName() << "\n");
    /// Check 
    auto *DEAA = A.getAAFor<AADataEnv>(
      *this, IRPosition::callsite_function(*EB.OMP.CB), DepClassTy::OPTIONAL);
    /// Check if the DEAA is valid
    if (!DEAA->getState().isValidState()) {
      LLVM_DEBUG(dbgs() <<"[AADataEnvFunction] DEAA is invalid\n");
      return false;
    }
    /// Check if the DEAA is at fixpoint
    if(!DEAA->getState().isAtFixpoint()) {
      LLVM_DEBUG(dbgs() <<"[AADataEnvFunction] DEAA is not at fixpoint\n");
      return false;
    }
    return true;
  }

  ChangeStatus updateImpl(Attributor &A) override {
    Function &F = cast<Function>(getAnchorValue());
    LLVM_DEBUG(dbgs() << "[AADataEnvFunction] updateImpl: "
                      << F.getName() << "\n");

    auto &AIC = static_cast<ARTSInformationCache &>(A.getInfoCache());

    // Callback to check a read/write instruction.
    auto CheckRWInst = [&](Instruction &I) {
      LLVM_DEBUG(dbgs() << "Checking "<< I <<"\n");
      auto *BB = I.getParent();
      auto *EB = AIC.ARTSTransform.getEDTBlock(BB);
      if(!EB) {
        LLVM_DEBUG(dbgs() << "[AADataEnvFunction] BB parent: " << BB->getName() << "\n");
        LLVM_DEBUG(dbgs() << "[AADataEnvFunction] BB is not an EDT region\n");
        indicatePessimisticFixpoint();
        return false;
      }
      /// Handle EDT init block
      if(EB->Type == EDTBlock::INIT)
        return handleEDTInitBlock(A, *EB);
      /// For now, we simply add them to the RWInsts vector
      EB->EDT->RWInsts.insert(&I);
      return true;
    };
  
    /// Call CheckRWInst for all instructions that may read or write
    /// in the function
    bool UsedAssumedInformationInCheckRWInst = false;
    if (!A.checkForAllReadWriteInstructions(
            CheckRWInst, *this, UsedAssumedInformationInCheckRWInst)) {
      LLVM_DEBUG(dbgs() << "  - checkForAllReadWriteInstructions failed\n");
      return ChangeStatus::UNCHANGED;
    }

    indicateOptimisticFixpoint();
    return ChangeStatus::CHANGED;
  }

  ChangeStatus manifest(Attributor &A) override {
    ChangeStatus Changed = ChangeStatus::UNCHANGED;
    LLVM_DEBUG(dbgs() << "[AADataEnvFunction] Manifest\n");
    return Changed;
  }
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
    LLVM_DEBUG(dbgs() << "[AADataEnvCallSite] Parallel region - FOUND\n");
    OutlinedFunction = dyn_cast<Function>(CB.getArgOperand(2));
    /// Get the number of arguments
    unsigned int NumArgs = CB.data_operands_size();
    /// Get private and shared variables
    for(unsigned int i = 3; i < NumArgs; i++) {
      Value *Arg = CB.getArgOperand(i);
      Type *ArgType = Arg->getType();
      // LLVM_DEBUG(dbgs() << "Argument: " << *Arg << "\n");
      // LLVM_DEBUG(dbgs() << "Argument type: " << *ArgType << "\n");

      /// For now assume that if it is a pointer, it is a shared variable
      if(PointerType *PT = dyn_cast<PointerType>(ArgType))
        DE.SharedVars.push_back(Arg);
      /// If not, it is a first private variable
      else
        DE.FirstprivateVars.push_back(Arg);

      /// NOTES: The outline function or the variable should have attributes that 
      /// provide more information about the lifetime of the variable. 
      /// It is also important to consider the underlying type of the variable.
      /// There may be cases, where there is a firstprivate variable that is a pointer.
      /// For this case, the pointer is private, but the data it points to is shared.
      ///
      /// Do we need to consider private variables?
    }
    /// For now, indicate optimistic fixpoint
    indicateOptimisticFixpoint();
    return ChangeStatus::CHANGED;
  }

  ChangeStatus handleTaskRegion(CallBase &CB, Attributor &A) {
    // For the shared variables we are interested, in all stores that are done to
    // the shareds field of the kmp_task_t struct. For the firstprivate variables
    // we are interested in all stores that are done to the privates field of the
    // kmp_task_t_with_privates struct.
    //
    // The AAPointerInfo attributor is used to obtain the access (read/write,
    // offsets and size) of a value. This attributor is run on the returned value
    // of the taskalloc function. The returned value is a pointer to the
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
    // obtaining stores done to offset 0 of the returned value of the taskalloc.
    // -For firstprivate variables, the access of the privates field is obtained by
    // obtaining stores done to offset 8 of the returned value of the
    // kmp_task_t_with_privates struct.
    //
    // If there is a load instruction that uses the returned value of the taskalloc
    // function, we need to run the AAPointerInfo attributor on it.
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
      LLVM_DEBUG(dbgs() <<"[AADataEnvCallSite] AAPointerInfo is invalid\n");
      return Changed;
    }
    /// Is it a fixpoint?
    if (!PI->getState().isAtFixpoint()) {
      LLVM_DEBUG(dbgs() <<"[AADataEnvCallSite] AAPointerInfo is not at fixpoint\n");
      return Changed;
    }
    LLVM_DEBUG(dbgs() << "[AADataEnvCallSite] AAPointerInfo is valid with: "
                      << PI->getOffsetBinsSize() << " bins\n");
    /// Analyze result
    /// This CB checks the access of each offset bin of PI
    const std::function<bool(const AA::RangeTy&, const SmallSet<unsigned, 4>&,
                             const AAPointerInfo *AAPI)> AccessCB = 
      [&](const AA::RangeTy &Range, const SmallSet<unsigned, 4> &AccIndex,
          const AAPointerInfo *AAPI) {
      LLVM_DEBUG(dbgs()
                  << "[" << Range.Offset << "-" << Range.Offset + Range.Size
                  << "] : " << AccIndex.size() << "\n");
      for (auto AI : AccIndex) {
        auto &Acc = AAPI->getAccess(AI);
        auto AccKind = Acc.getKind();
        auto *Inst = Acc.getLocalInst();
        LLVM_DEBUG(dbgs() << "     - " << AccKind << " - " << *Inst << "\n");

        if (Acc.isWrittenValueYetUndetermined()) {
          LLVM_DEBUG(dbgs()
                      << "       - c: Value written is not known yet \n");
          return false;
        }
        /// Store instruction. 
        if (AccKind == AAPointerInfo::AccessKind::AK_MUST_WRITE) {
          // LLVM_DEBUG(dbgs() << "       -- Store instruction\n");
          Value *value = Acc.getWrittenValue();
          if(!value)
            return false;
          /// Check if it is a shared variable
          if(Range.Offset == 0) {
            LLVM_DEBUG(dbgs() << "       - c: shared variable " << *value << "\n");
            AuxDE.SharedVars.push_back(value);
          }
          /// Check if it is a firstprivate variable
          else if((uint64_t)Range.Offset >= (uint64_t)kmp_task_t_size) {
            LLVM_DEBUG(dbgs() << "       - c: firstprivate variable " << *value << "\n");
            AuxDE.FirstprivateVars.push_back(value);
          }
          else
            LLVM_DEBUG(dbgs() << "       - c: other: " << *value << "\n");
        }
        /// Read instruction
        else if(AccKind == AAPointerInfo::AccessKind::AK_MUST_READ) {
          // LLVM_DEBUG(dbgs() << "       -- Read instruction\n");
          /// Check if it a load instruction
          if(auto *LI = dyn_cast<LoadInst>(Inst)) {
            LLVM_DEBUG(dbgs() << "       - Load instruction. Running AA on this.\n");
            auto *PIL = A.getAAFor<AAPointerInfo>(
              *this, IRPosition::value(*LI), DepClassTy::OPTIONAL);
            if (!PIL->getState().isValidState()) {
              indicatePessimisticFixpoint();
              LLVM_DEBUG(dbgs() <<"[AADataEnvCallSite] PIL is invalid\n");
              return false;
            }
            /// Is it a fixpoint?
            if (!PIL->getState().isAtFixpoint()) {
              LLVM_DEBUG(dbgs() <<"[AADataEnvCallSite] PIL is not at fixpoint\n");
              return false;
            }
            /// For all offset bins in the PI, run AccessCB
            if(!PIL->forallOffsetBins(AccessCB)) {
              LLVM_DEBUG(dbgs() << "PI forallOffsetBins failed\n");
              indicatePessimisticFixpoint();
              return false;
            }
          }
        }
      }
      return true;
    };
    
    /// For all offset bins in the PI, run AccessCB
    if(!PI->forallOffsetBins(AccessCB)) {
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

  ChangeStatus handleTaskwait() {
    return ChangeStatus::UNCHANGED;
  }

  void initialize(Attributor &A) override {
    CallBase &CB = cast<CallBase>(getAssociatedValue());
    Function *Callee = getAssociatedFunction();
    LLVM_DEBUG(dbgs() <<"[AADataEnvCallSite] initialize: " << Callee->getName() << "\n");
    switch (OMPInfo::getRTFunction(Callee)) {
      case OMPInfo::SET_NUM_THREADS: {
        auto *Arg = CB.getArgOperand(0);
        if(auto *NumThreads = dyn_cast<ConstantInt>(Arg)) {
          LLVM_DEBUG(dbgs() << "Number of threads: "
                            << NumThreads->getZExtValue() << "\n");
        }
        indicateOptimisticFixpoint();
      }
      break;
      case OMPInfo::PARALLEL:
        /// Handle it in the updateImpl function
      break;
      case OMPInfo::TASKALLOC: {
        /// The task is created by the taskalloc function, and it returns a
        /// pointer to the task (check handleTaskRegion documentation ). 
        /// We need to obtain its data environment. Since this pointer is used in
        /// functions that are not analyzable, and for purposes of this analysis we 
        /// are not interested on what happens inside those functions, we will
        /// add the nocapture attribute. A pointer is captured by the call if it
        /// makes a copy of any part of the pointer that outlives the call.
        for(auto &U : CB.uses()) {
          auto *User = U.getUser();
          // LLVM_DEBUG(dbgs() << "--- " << *U << "\n");
          // LLVM_DEBUG(dbgs() << "------ " << *User << "\n");
          // LLVM_DEBUG(dbgs() << "------ " << *(U.get()) << "\n");
          if(auto *CBI = dyn_cast<CallInst>(User)) {
            auto *Callee = CBI->getCalledFunction();
            if(OMPInfo::isTaskFunction(Callee)) {
              /// Add the attribute nocapture to the returned value
              CBI->addParamAttr(U.getOperandNo(), Attribute::NoCapture);
            }
          }
        }
        /// Lets handle the task in the updateImpl function
      } break;
      case OMPInfo::OTHER: {
        LLVM_DEBUG(dbgs() << "Other instruction - FOUND\n");
        /// Unknown caller or declarations are not analyzable, we give up.
        if (!Callee || !A.isFunctionIPOAmendable(*Callee)) {
          indicatePessimisticFixpoint();
          LLVM_DEBUG(dbgs() <<"Unknown caller or declarations are not analyzable, we give up.\n");
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
    LLVM_DEBUG(dbgs() << "[AADataEnvCallSite] updateImpl: " << Callee->getName() << "\n");

    switch (OMPInfo::getRTFunction(Callee)) {
      case OMPInfo::PARALLEL:
        Changed |= handleParallelRegion(CB, A);
      break;
      case OMPInfo::TASKALLOC:
        Changed |= handleTaskRegion(CB, A);
      break;
      default: {
        LLVM_DEBUG(dbgs() << "Other instruction - FOUND\n");
      }
      break;
    }

    if (Changed == ChangeStatus::CHANGED) {
      /// If the data environment is updated, update info in the cache
      auto &AIC = static_cast<ARTSInformationCache &>(A.getInfoCache());
      auto *EB = AIC.ARTSTransform.getEDTBlock(CB.getParent());
      auto &OMP_DE = EB->OMP.DE;
      OMP_DE.append(DE);
      // LLVM_DEBUG(dbgs() << OMP_DE << "\n");
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

AADataEnv &AADataEnv::createForPosition(const IRPosition &IRP,
                                        Attributor &A) {
  AADataEnv *AA = nullptr;
  switch (IRP.getPositionKind()) {
  case IRPosition::IRP_INVALID:
  case IRPosition::IRP_ARGUMENT:
  case IRPosition::IRP_CALL_SITE_ARGUMENT:
  case IRPosition::IRP_RETURNED:
  case IRPosition::IRP_CALL_SITE_RETURNED:
    llvm_unreachable(
        "AAToARTS can only be created for float/callsite position!");
  case IRPosition::IRP_FUNCTION:
    AA = new (A.Allocator) AADataEnvFunction(IRP, A);
    break;
  case IRPosition::IRP_CALL_SITE:
    AA = new (A.Allocator) AADataEnvCallSite(IRP, A);
    break;
  case IRPosition::IRP_FLOAT:
    AA = new (A.Allocator) AADataEnvBasicBlock(IRP, A);
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

  static AAToARTS &createForPosition(const IRPosition &IRP,
                                           Attributor &A);

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

  bool identifyEDTs(Function &F, ARTSTransformer &AT, Attributor &A) {
    bool Changed = false;
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
    ///   %1 = call i32 @__kmpc_fork_call(i32 1, i32 (i32, i8*)* @__omp_offloading_fib, i8* %0)
    ///   ...
    ///   ret i32 0
    /// It is transformed to:
    /// BB1:
    ///   %0 = alloca i32, align 4
    ///   %1 = call i32 @__kmpc_fork_call(i32 1, i32 (i32, i8*)* @__omp_offloading_fib, i8* %0)
    ///   br label %edt.done.0
    /// edt.done.0:
    ///   ...
    ///   ret i32 0
    ///
    /// For the example above, the BB that contains the call to __kmpc_fork_call will be then 
    /// replaced by the EDT initializer (GUID reserve + call to EDT allocator) and the parallel
    /// outlined function will be the EDT.
    /// We then have to analyze whether the par.done BB needs to be transformed to an EDT. This
    /// is done by analyzing the instructions of the BB to obtain its data environment. If any
    /// of the instructions in the BB uses a shared variable, the BB is transformed to an EDT.
    ///
    /// By the end of this function, we should have a map that contains the BBs that can be
    /// transformed to EDTs. The key of the map is the BB that contains the call to the function
    /// that creates the region (e.g. __kmpc_fork_call) and the value is the struct EDTInfo.
    /// Aux variables
    LoopInfo *LI = nullptr;
    DominatorTree *DT = nullptr;
    /// Region counter
    unsigned int ParallelRegion = 0;
    unsigned int TaskRegion = 0;

    LLVM_DEBUG(dbgs() << TAG << "Identifying EDT regions\n");
    EDTInfo *EI = AT.insertEDT(&F);

    if(!A.isFunctionIPOAmendable(F))
      return Changed;
    /// Get entry block
    BasicBlock *CurBB = &(F.getEntryBlock());
    AT.insertEDTBlock(EI, EDTBlock::ENTRY, CurBB);
    BasicBlock *NextBB = nullptr;
    do {
      // LLVM_DEBUG(dbgs() << TAG << "CurBB: " << *CurBB << "\n");
      NextBB = CurBB->getNextNode();
      /// Get first instruction of the function
      Instruction *CurI = &(CurBB->front());
      do {
        // LLVM_DEBUG(dbgs() << TAG << "CurI: " << *CurI << "\n");
        auto *CB = dyn_cast<CallBase>(CurI);
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
              SplitBlock(CurI->getParent(), CurI, DT, LI, nullptr,
                        "par.region." + std::to_string(ParallelRegion));
            EDTBlock *EB = AT.insertEDTBlock(EI, EDTBlock::INIT, ParallelBB, OI);
            AT.insertEDTWithDep(EI, EB);
            /// Split block at the next instruction
            CurI = CurI->getNextNonDebugInstruction();
            BasicBlock *ParallelDone = 
              SplitBlock(ParallelBB, CurI, DT, LI, nullptr,
                        "par.done." + std::to_string(ParallelRegion));
            NextBB = ParallelDone;
            ParallelRegion++;
            /// Create new EDT and add dependency from par.region 
            /// to par.done
            EI = AT.insertEDTWithDep(EI, &F);
            AT.insertEDTBlock(EI, EDTBlock::ENTRY, ParallelDone);
          }
          break;
          case OMPInfo::TASKALLOC: {
            OMPInfo OI(OMPInfo::TASK, CB);
            /// Split block at __kmpc_omp_task_alloc
            BasicBlock *TaskBB = 
              SplitBlock(CurI->getParent(), CurI, DT, LI, nullptr,
                        "task.region." + std::to_string(TaskRegion));
            EDTBlock *EB = AT.insertEDTBlock(EI, EDTBlock::INIT, TaskBB, OI);
            AT.insertEDTWithDep(EI, EB);
            /// Find the task call
            while ((CurI = CurI->getNextNonDebugInstruction())) {
              auto *TCB = dyn_cast<CallBase>(CurI);
              if(TCB && 
                OMPInfo::getRTFunction(TCB->getCalledFunction()) == OMPInfo::TASK)
                break;
            }
            assert(CurI && "Task RT call not found");
            /// Split block again at the next instruction
            CurI = CurI->getNextNonDebugInstruction();
            BasicBlock *TaskDone = 
              SplitBlock(TaskBB, CurI, DT, LI, nullptr,
                        "task.done." + std::to_string(TaskRegion));
            NextBB = TaskDone;
            TaskRegion++;
            /// Create new EDT and add dependency from par.region 
            /// to par.done
            EI = AT.insertEDTWithDep(EI, &F);
            AT.insertEDTBlock(EI, EDTBlock::ENTRY, TaskDone, OI);
          }
          break;
          case OMPInfo::TASKWAIT: {
            /// \Note A taskwait requires an event.
            /// Split block at __kmpc_omp_taskwait
            BasicBlock *TaskWaitBB = 
              SplitBlock(CurI->getParent(), CurI, DT, LI, nullptr,
                        "taskwait.region." + std::to_string(TaskRegion));
            /// Split block again at the next instruction
            CurI = CurI->getNextNonDebugInstruction();
            BasicBlock *TaskWaitDone = 
              SplitBlock(TaskWaitBB, CurI, DT, LI, nullptr,
                        "taskwait.done." + std::to_string(TaskRegion));
            NextBB = TaskWaitDone;
            /// Add the taskwait region to the map
            // AT.insertEDTBlock(TaskWaitBB, RTF, CB);
            TaskRegion++;
          }
          break;
          case OMPInfo::OTHER: {
            // if (!Callee || !A.isFunctionIPOAmendable(*Callee))
            //   continue;
            // /// Get return type of the callee
            // Type *RetTy = Callee->getReturnType();
            // /// If the return type is void, we are not interested
            // if(RetTy->isVoidTy())
            //   continue;
            /// If the return type is a pointer, it is because we probably would use 
            /// the returned value. For this case, we need to create an EDT. 
          }
          break;
          default:
            continue;
          break;
        }
      } while ((CurI = CurI->getNextNonDebugInstruction()));

    } while ((CurBB = NextBB));

    LLVM_DEBUG(dbgs() << TAG << "Identifying EDT regions done\n");
                      // << AT.EDTs << "\n");

    Changed = true;
    return Changed;
  }

  void checkValue(Value *Val, ARTSTransformer &AT, EDTInfo *EDT) {
    // Type *ValType = Val->getType();
    // LLVM_DEBUG(dbgs() << "  - Operand: " << *Val << "\n");
    // LLVM_DEBUG(dbgs() << "    - Operand type: " << *ValType << "\n");

    if(isa<Argument>(Val)) {
      // LLVM_DEBUG(dbgs() << "    - It is an Argument\n");
      return;
    }

    auto *ValInst = dyn_cast<Instruction>(Val);
    if(!ValInst) {
      /// It can be a constant value...
      // LLVM_DEBUG(dbgs() << "    - It is not an Instruction\n");
      return;
    }

    /// Check if the instruction is in the same EDT Data scope
    // LLVM_DEBUG(dbgs() << "    - It is an Instruction\n");
    auto *EB = AT.getEDTBlock(ValInst->getParent());
    EB->Analyzed = true;
    if(EB->EDT->ID == EDT->ID) {
      // LLVM_DEBUG(dbgs() << "    - It is in the same EDT\n");
      return;
    }
    else {
      // LLVM_DEBUG(dbgs() << "    - It is not in the same EDT\n");
      EDT->ExternalValues.insert(ValInst);
    }
  }

  void initialize(Attributor &A) override {
    Function *F = getAnchorScope();
    LLVM_DEBUG(dbgs() <<"\n[AAToARTSFunction] initialize: " 
                      << F->getName() << "\n" << *F << "\n");
    auto &ARTSInfoCache = static_cast<ARTSInformationCache &>(A.getInfoCache());
    auto &AT = ARTSInfoCache.ARTSTransform;
    if(!identifyEDTs(*F, AT, A)) {
      indicatePessimisticFixpoint();
      return;
    }
    /// The rest is handled in the updateImpl function
  }

  ChangeStatus updateImpl(Attributor &A) override {
    Function *F = getAnchorScope();
    LLVM_DEBUG(dbgs() << "[AAToARTSFunction] updateImpl: " << F->getName() << "\n");

    /// Get the data environment of the function. If successful, it returns
    /// the list of instructions that may read or write to memory in the function.
    auto &ARTSInfoCache = static_cast<ARTSInformationCache &>(A.getInfoCache());
    auto &AT = ARTSInfoCache.ARTSTransform;
    auto *DEAA = A.getAAFor<AADataEnv>(
      *this, IRPosition::function(*F), DepClassTy::OPTIONAL);
    if (!DEAA->getState().isValidState()) {
      LLVM_DEBUG(dbgs() <<"[AAToARTSFunction] DEAA is invalid\n");
      indicatePessimisticFixpoint();
      return;
    }
    if(!DEAA->getState().isAtFixpoint()) {
      LLVM_DEBUG(dbgs() <<"[AAToARTSFunction] DEAA is not at fixpoint\n");
      return;
    }
    LLVM_DEBUG(dbgs() <<"[AAToARTSFunction] DEAA is at fixpoint\n");

    /// Iterate through the EDT blocks to identify the list of instructions
    /// that may read or write to memory but doesn't belong to the EDT.
    for(auto *EDT : AT.EDTsFromFunction[F]) {
      // LLVM_DEBUG(dbgs() << "EDT #" << EDT->ID << "\n");
      /// Print RW instructions
      // LLVM_DEBUG(dbgs() << "RW instructions:\n");
      for(auto *I : EDT->RWInsts) {
        // LLVM_DEBUG(dbgs() << "  - " << *I << "\n");
        /// Iterate through the arguments of the callbase
        if(auto *CB = dyn_cast<CallBase>(I)) {
          for(auto &arg : CB->args())
            checkValue(arg, AT, EDT);
          continue;
        }
        /// Iterate through the operands of the instruction
        for(unsigned int i = 0; i < I->getNumOperands(); i++)
          checkValue(I->getOperand(i), AT, EDT);
        // LLVM_DEBUG(dbgs() << "\n");
      }
     
    }
    LLVM_DEBUG(dbgs() << TAG << "EDTs INFORMATION\n");
    LLVM_DEBUG(dbgs() << TAG << AT.EDTs << "\n");


    return ChangeStatus::UNCHANGED;
  }

  ChangeStatus manifest(Attributor &A) override {
    ChangeStatus Changed = ChangeStatus::UNCHANGED;
    LLVM_DEBUG(dbgs() << "[AAToARTSFunction] Manifest\n");
    return Changed;
  }
};

struct AAToARTSBasicBlock : AAToARTS {
  AAToARTSBasicBlock(const IRPosition &IRP, Attributor &A)
      : AAToARTS(IRP, A) {}

  /// See AbstractAttribute::getAsStr()
  const std::string getAsStr(Attributor *A) const override {
    if (!isValidState())
      return "<invalid>";

    std::string Str("AAToARTSBasicBlock: ");

    return Str + std::string("OK");
  }

  void handleEDTInitRegion(EDTInfo *EI, Attributor &A) {
    // auto &DE = EI->DE;
    // LLVM_DEBUG(dbgs() <<"[AAToARTSBasicBlock] Analyzing: " << DE.getCBName() << "\n");
    // /// Get data environment of the EDT callbase
    // auto *DEAA = A.getAAFor<AADataEnv>(
    //   *this, IRPosition::callsite_function(*DE.CB), DepClassTy::OPTIONAL);
    // if (!DEAA->getState().isValidState() || !DEAA->getState().isAtFixpoint()) {
    //   LLVM_DEBUG(dbgs() <<"[AAToARTSBasicBlock] DEAA is invalid or not at fixpoint\n");
    //   indicatePessimisticFixpoint();
    //   return;
    // }
    // /// Print the data environment for now
    // LLVM_DEBUG(dbgs() << DE << "\n");
  }

  void initialize(Attributor &A) override {
    // BasicBlock &BB = cast<BasicBlock>(getAnchorValue());
    // /// Get Data environment for the BB
    // LLVM_DEBUG(dbgs() <<"[AAToARTSBasicBlock] initialize: " << BB.getName() << "\n");
    // auto *DEAA = A.getAAFor<AADataEnv>(
    //   *this, IRPosition::value(BB), DepClassTy::OPTIONAL);
    // if (!DEAA->getState().isValidState() || !DEAA->getState().isAtFixpoint()) {
    //   LLVM_DEBUG(dbgs() <<"[AAToARTSBasicBlock] DEAA is invalid or not at fixpoint\n");
    //   indicatePessimisticFixpoint();
    //   return;
    // }
    // /// If it is a fixpoint, we can proceed to analyze the successors
    // if (BB.getTerminator()->getNumSuccessors() != 1) {
    //   LLVM_DEBUG(dbgs() << "[AAToARTSBasicBlock] BB has more than one successor\n");
    //   /// For now indicate pessimistic fixpoint
    //   indicatePessimisticFixpoint();
    //   return;
    // }
    // BasicBlock *SuccBB = BB.getTerminator()->getSuccessor(0);
    // /// Get SuccBB data environment
    // auto *SuccDEAA = A.getAAFor<AADataEnv>(
    //   *this, IRPosition::value(*SuccBB), DepClassTy::OPTIONAL);
    // if (!SuccDEAA->getState().isValidState() || !SuccDEAA->getState().isAtFixpoint()) {
    //   LLVM_DEBUG(dbgs() <<"[AAToARTSBasicBlock] SuccDEAA is invalid or not at fixpoint\n");
    //   indicatePessimisticFixpoint();
    //   return;
    // }
    // /// If it is a fixpoint, we can analyze both data environments
    // /// The idea here is to determine if a shared variable in the current BB 
    // /// is written or read in the successor BB. If so, we need to signal the value
    // DEAA->dumpDataEnv();
    // SuccDEAA->dumpDataEnv();

  }

  ChangeStatus updateImpl(Attributor &A) override {
    ChangeStatus Changed = ChangeStatus::UNCHANGED;
    /// Cast value to BasicBlock
    BasicBlock &BB = cast<BasicBlock>(getAnchorValue());
    LLVM_DEBUG(dbgs() << "[AAToARTSBasicBlock] updateImpl: " <<BB.getName() <<"\n");
    
    return Changed;
  }

  ChangeStatus manifest(Attributor &A) override {
    ChangeStatus Changed = ChangeStatus::UNCHANGED;
    // /// Cast value to BasicBlock
    // BasicBlock &BB = cast<BasicBlock>(getAnchorValue());
    // LLVM_DEBUG(dbgs() << "[AAToARTSBasicBlock] Manifest: " <<BB.getName() <<"\n");
    // auto &ARTSInfoCache = static_cast<ARTSInformationCache &>(A.getInfoCache());
    // auto *EI = ARTSInfoCache.ARTSTransform.getEDTBlock(BB);
    /// Get parent function
    // Function *F = BB.getParent();
    // MemorySSA &MSSA = AM.getResult<MemorySSAAnalysis>(F).getMSSA();
    // AliasAnalysis &AA = AM.getResult<AAManager>(F);

    // auto &ARTSBuilder = ARTSInfoCache.ARTSBuilder;
    // Function *newEDT = ARTSBuilder.createEDT(*EI, &BB);
    // if(!newEDT)
    //   Changed |= ChangeStatus::CHANGED;
    // /// We have to analyze the BB successors. We need to check for the following
    // /// cases:
    // /// - If the successor (par/task.done BB) reads any of the shared variables
    // ///   in the data environment, it needs to be transformed to an EDT, and the
    // ///   current BB needs to signal the value.
    // /// - If not, there is a control dependency between the current BB and the
    // ///   successor. Here, there are two cases:
    // ///   - The successor has a taskwait/taskwait_deps call. In this case,
    // ///     it would be interesting to create an ARTS API function that waits
    // ///     for a list of EDTs to finish (using the GUIDs).
    // /// 
    // /// Check if number of successors is 1. 
    // if (BB.getTerminator()->getNumSuccessors() != 1) {
    //   LLVM_DEBUG(dbgs() << "[AAToARTSBasicBlock] BB has more than one successor\n");
    //   indicatePessimisticFixpoint();
    //   return Changed;
    // }
    // BasicBlock *SuccBB = BB.getTerminator()->getSuccessor(0);
    /// Get SuccBB data environment
    // LLVM_DEBUG(dbgs() << "[AAToARTSBasicBlock] Manifest\n");
    return Changed;
  }
};

AAToARTS &AAToARTS::createForPosition(const IRPosition &IRP,
                                      Attributor &A) {
  AAToARTS *AA = nullptr;
  switch (IRP.getPositionKind()) {
  case IRPosition::IRP_INVALID:
  case IRPosition::IRP_ARGUMENT:
  case IRPosition::IRP_CALL_SITE_ARGUMENT:
  case IRPosition::IRP_RETURNED:
  case IRPosition::IRP_CALL_SITE_RETURNED:
  case IRPosition::IRP_CALL_SITE:
    llvm_unreachable(
        "AAToARTS can only be created for function/float position!");
    break;
  case IRPosition::IRP_FUNCTION:
    AA = new (A.Allocator) AAToARTSFunction(IRP, A);
    break;
  case IRPosition::IRP_FLOAT:
    AA = new (A.Allocator) AAToARTSBasicBlock(IRP, A);
    break;
  }
  return *AA;
}

const char AAToARTS::ID = 0;


/// ---------------------------- ARTS TRANSFORM ---------------------------- ///
bool ARTSTransformer::run(Attributor &A) {
  bool changed = false;
  
  /// Identify EDT regions
  // changed |= identifyEDTInitRegions(A);
  // Function *EDT = ARTSIRB.createEDT("fib");
  // LLVM_DEBUG(dbgs() << TAG << "EDT: \n" << *EDT << "\n");
  // if(!changed) {
  //   LLVM_DEBUG(dbgs() << TAG << "No EDT regions found\n");
  //   return changed;
  // }
  /// Run attributor on the EDTInitRegions
  /// The process start in the main function, we which will converted to an EDT.
  /// We start the process by iterating through the BBs of the main function. 
  /// The idea is to get the data environment of the BB. There are two cases here:
  /// - The BB is an EDTInitRegion (parallel, task...). In this case, we simply in the 
  ///   corresponding callsite.
  /// - The BB is not an EDTInitRegion. In this case, we need to run the MemorySSA
  ///   analysis on the BB to obtain the data environment (memory definition and
  ///   uses -> shared and firstprivate variables).
  /// After this, we need to analyze the BB successors. We need to check for the
  /// following cases:
  /// - If the successor (par/task.done BB) reads any of the shared variables
  ///   in the data environment.
  /// - If not, there is a control dependency between the current BB and the
  ///   successor. Here, there are two cases:
  ///   - The successor has a taskwait/taskwait_deps call. In this case,
  ///     we can crete an event

  /// Get main function
  Function *MainF = M.getFunction("main");
  if(!MainF) {
    LLVM_DEBUG(dbgs() << TAG << "Main function not found\n");
    return changed;
  }
  /// Run AAToARTS on the main function 
  LLVM_DEBUG(dbgs() << TAG << "Initializing AAToARTS attributor for main function\n");
    A.getOrCreateAAFor<AAToARTS>(
          IRPosition::function(*MainF), /* QueryingAA */ nullptr,
          DepClassTy::NONE, /* ForceUpdate */ false,
          /* UpdateAfterInit */ true);
  changed |= runAttributor(A);
  return changed;
}

bool ARTSTransformer::runAttributor(Attributor &A) {
  LLVM_DEBUG(dbgs() << TAG <<  "[Attributor] Process started\n");
  ChangeStatus Changed = A.run();
  LLVM_DEBUG(dbgs() << TAG <<  "[Attributor] Done, result: " << Changed << ".\n");

  return Changed == ChangeStatus::CHANGED;
}

/// ---------------------------- ARTS TRANSFORMATION PASS ---------------------------- ///
PreservedAnalyses ARTSTransformPass::run(Module &M, ModuleAnalysisManager &AM) {
  /// Command line options
  if (!EnableARTSTransformation)
    return PreservedAnalyses::all();
  if (PrintModuleBeforeOptimizations)
    LLVM_DEBUG(dbgs() << TAG << "Module before ARTSTransformer Module Pass:\n" << M);

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
  Attributor A(Functions, InfoCache, AC);

  /// Run ARTSTransform
  Changed |= InfoCache.ARTSTransform.run(A);

  LLVM_DEBUG(dbgs() << TAG << "Module after ARTSTransformer Module Pass:\n" << M);
  if (Changed)
    return PreservedAnalyses::none();
  return PreservedAnalyses::all();
}


#include "llvm/Transforms/ARTS/ARTSTransform.h"
#include "llvm/Frontend/ARTS/ARTSIRBuilder.h"

#include "llvm/Support/Debug.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

using namespace llvm;

// DEBUG
#define DEBUG_TYPE "arts-transform"
#if !defined(NDEBUG)
static constexpr auto TAG = "[" DEBUG_TYPE "] ";
#endif

// COMMAND LINE OPTIONS
static cl::opt<bool> DisableARTSTransformation(
    "arts-disable", cl::desc("Disable transformation to ARTS."),
    cl::Hidden, cl::init(false));

static cl::opt<bool> PrintModuleBeforeOptimizations(
    "arts-print-module-before",
    cl::desc("Print the module before the ARTSTransform Module Pass"),
    cl::Hidden, cl::init(false));

static cl::opt<bool> PrintModuleAfterOptimizations(
    "arts-print-module-after",
    cl::desc("Print the module after the ARTSTransform Module Pass"),
    cl::Hidden, cl::init(false));

// UTILS

RTFunction getRTFunction(Function *F) {
  auto calleeName = F->getName();
  if(calleeName == "__kmpc_fork_call")
    return RTFunction::PARALLEL;
  else if(calleeName == "__kmpc_omp_task_alloc")
    return RTFunction::TASKALLOC;
  else if(calleeName == "__kmpc_omp_task")
    return RTFunction::TASK;
  else if(calleeName == "__kmpc_omp_task_alloc_with_deps")
    return RTFunction::TASKDEP;
  else if(calleeName == "__kmpc_omp_taskwait")
    return RTFunction::TASKWAIT;
  else if(calleeName == "omp_set_num_threads")
    return RTFunction::SET_NUM_THREADS;
  else if(calleeName == "__kmpc_for_static_init_4")
    return RTFunction::PARALLEL_FOR;
  return RTFunction::OTHER;
}

bool isTaskFunction(Function *F) {
  auto RT = getRTFunction(F);
  if(RT == RTFunction::TASK || RT == RTFunction::TASKDEP || RT == RTFunction::TASKWAIT )
    return true;
  return false;
}

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

  void initialize(Attributor &A) override {
    Function *F = getAnchorScope();
    LLVM_DEBUG(dbgs() <<"\n[AAToARTSFunction] initialize: " 
                      << F->getName() << "\n" << *F << "\n");
  }

  ChangeStatus updateImpl(Attributor &A) override {
    Function *F = getAnchorScope();
    LLVM_DEBUG(dbgs() << "[AAToARTSFunction] updateImpl: " << F->getName() << "\n");
    return ChangeStatus::UNCHANGED;
  }

  ChangeStatus manifest(Attributor &A) override {
    ChangeStatus Changed = ChangeStatus::UNCHANGED;
    LLVM_DEBUG(dbgs() << "[AAToARTSFunction] Manifest\n");
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
  case IRPosition::IRP_FLOAT:
  case IRPosition::IRP_CALL_SITE:
    llvm_unreachable(
        "AATDGInfo can only be created for function/callsite position!");
    break;
  case IRPosition::IRP_FUNCTION:
    AA = new (A.Allocator) AAToARTSFunction(IRP, A);
    break;
  }
  return *AA;
}

const char AAToARTS::ID = 0;


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

  /// Data environment
  DataEnv DE;
  const std::string getDataEnv() const {
    std::string Str("---DATA ENVIRONMENT:\n");
    /// Iterate through the firstprivate variables
    for(auto *FPV : DE.FirstprivateVars)
      Str += std::string("Firstprivate: ") + std::string(FPV->getName()) + std::string("\n");
    /// Iterate through the private variables
    for(auto *PV : DE.PrivateVars)
      Str += std::string("Private: ") + std::string(PV->getName()) + std::string("\n");
    /// Iterate through the shared variables
    for(auto *SV : DE.SharedVars)
      Str += std::string("Shared: ") + std::string(SV->getName()) + std::string("\n");
    /// Iterate through the lastprivate variables
    for(auto *LPV : DE.LastprivateVars)
      Str += std::string("Lastprivate: ") + std::string(LPV->getName()) + std::string("\n");
    return Str;
  }
};

struct AADataEnvFunction : AADataEnv {
  AADataEnvFunction(const IRPosition &IRP, Attributor &A) : AADataEnv(IRP, A) {}

  /// See AbstractAttribute::getAsStr(Attributor *A)
  const std::string getAsStr(Attributor *A) const override {
    if (!isValidState())
      return "<invalid>";
    std::string Str("AADataEnvFunction: ");
    return Str + getDataEnv();
  }

  void initialize(Attributor &A) override {
    Function *F = getAnchorScope();
    LLVM_DEBUG(dbgs() <<"\n[AADataEnvFunction] initialize: " 
                      << F->getName() << "\n" << *F << "\n");
  }

  ChangeStatus updateImpl(Attributor &A) override {
    Function *F = getAnchorScope();
    LLVM_DEBUG(dbgs() << "[AADataEnvFunction] updateImpl: " << F->getName() << "\n");
    /// Iterate through the BB of each function
    for(auto &BB : *F) {
      /// Iterate through the instructions of each BB
      for(auto &I : BB) {
        if(auto *CB = dyn_cast<CallBase>(&I)) {
          auto *DEAA = A.getAAFor<AADataEnv>(
            *this, IRPosition::callsite_function(*CB), DepClassTy::OPTIONAL);
          if (!DEAA->getState().isValidState() || !DEAA->getState().isAtFixpoint()) {
            LLVM_DEBUG(dbgs() <<"[AADataEnvFunction] DEAA is invalid or not at fixpoint\n");
            continue;
          }
          /// if the DEAA is at fixpoint, print the data environment for now
          LLVM_DEBUG(dbgs() << DEAA->getDataEnv() << "\n");
          /// append the data environment
          // DE.append(DEAA->DE);
        }
      }
    }
    return ChangeStatus::UNCHANGED;
  }

  ChangeStatus manifest(Attributor &A) override {
    ChangeStatus Changed = ChangeStatus::UNCHANGED;
    LLVM_DEBUG(dbgs() << "[AADataEnvFunction] Manifest\n");
    return Changed;
  }
};

struct AADataEnvCallSite : AADataEnv {
  AADataEnvCallSite(const IRPosition &IRP, Attributor &A) : AADataEnv(IRP, A) {}

  /// See AbstractAttribute::getAsStr(Attributor *A)
  const std::string getAsStr(Attributor *A) const override {
    if (!isValidState())
      return "<invalid>";

    std::string Str("AADataEnvCallSite: ");
    return Str + getDataEnv();
  }

  void handleParallelRegion(CallBase &CB, Attributor &A) {
    LLVM_DEBUG(dbgs() << "[AADataEnvCallSite] Parallel region - FOUND\n");
    /// Get the number of arguments
    unsigned int NumArgs = CB.data_operands_size();
    // LLVM_DEBUG(dbgs() << "Number of arguments: " << NumArgs << "\n");
    auto OutlinedRegion = CB.getArgOperand(2);
    DE.F = dyn_cast<Function>(OutlinedRegion);

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

      /// Do we need to consider private variables?
    }
    /// Run AA on the outline function
    auto *DEF = A.getAAFor<AADataEnv>(
      *this, IRPosition::function(*DE.F), DepClassTy::NONE);
    if (!DEF->getState().isValidState()) {
      indicatePessimisticFixpoint();
      LLVM_DEBUG(dbgs() <<"[AADataEnvCallSite] DEF is invalid\n");
      return;
    }
    /// TODO: Check if it is a fixpoint, if so, append the data environment
    /// of the outline function. Otherwise, record 
  }

  ChangeStatus handleTaskRegion(CallBase &CB, Attributor &A) {
    LLVM_DEBUG(dbgs() << "[AADataEnvCallSite] Task alloc - FOUND\n");
    /* 
    For the shared variables we are interested, in all stores that are done to
    the shareds field of the kmp_task_t struct. For the firstprivate variables
    we are interested in all stores that are done to the privates field of the
    kmp_task_t_with_privates struct.

    The AAPointerInfo attributor is used to obtain the access (read/write,
    offsets and size) of a value. This attributor is run on the returned value
    of the taskalloc function. The returned value is a pointer to the
    kmp_task_t_with_privates struct.
    struct kmp_task_t_with_privates {
       kmp_task_t task_data;
       kmp_privates_t privates;
    };
    typedef struct kmp_task {
      void *shareds;
      kmp_routine_entry_t routine;
      kmp_int32 part_id;
      kmp_cmplrdata_t data1;
      kmp_cmplrdata_t data2;
    } kmp_task_t;

    - For shared variables, the access of the shareds field is obtained by
    obtaining stores done to offset 0 of the returned value of the taskalloc.
    -For firstprivate variables, the access of the privates field is obtained by
    obtaining stores done to offset 8 of the returned value of the
    kmp_task_t_with_privates struct.

    If there is a load instruction that uses the returned value of the taskalloc
    function, we need to run the AAPointerInfo attributor on it.

    The function returns ChangeStatus::CHANGED if the data environment is
    updated, ChangeStatus::UNCHANGED otherwise.
    */
    /// Aux variables
    ChangeStatus Changed = ChangeStatus::UNCHANGED;
    DataEnv AuxDE(DE.RTF, DE.F);
    /// Get context and module
    LLVMContext &Ctx = CB.getContext();
    Module *M = CB.getModule();
    /// Get the size of the kmp_task_t struct
    auto *kmp_task_t = dyn_cast<StructType>(CB.getType());
    auto kmp_task_t_size = M->getDataLayout().getTypeAllocSize(
      kmp_task_t->getTypeByName(Ctx, "struct.kmp_task_t"));
    /// Run AAPointerInfo on the returned value
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
          else if(Range.Offset >= kmp_task_t_size) {
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
      // indicatePessimisticFixpoint();
      return Changed;
    }
    /// If we reach this point, it means that the PI is valid and at fixpoint.
    indicateOptimisticFixpoint();
    /// Append the data environment
    DE.append(AuxDE);
    Changed = ChangeStatus::CHANGED;
    return Changed;
  }

  void initialize(Attributor &A) override {
    CallBase &CB = cast<CallBase>(getAssociatedValue());
    Function *Callee = getAssociatedFunction();
    LLVM_DEBUG(dbgs() <<"[AADataEnvCallSite] ----- initialize: " << Callee->getName() << "\n");
    RTF = getRTFunction(Callee);
    switch (RTF) {
      case SET_NUM_THREADS: {
        auto *Arg = CB.getArgOperand(0);
        if(auto *NumThreads = dyn_cast<ConstantInt>(Arg)) {
          LLVM_DEBUG(dbgs() << "Number of threads: " << NumThreads->getZExtValue() << "\n");
        }
        indicateOptimisticFixpoint();
      }
      break;
      case PARALLEL:
        handleParallelRegion(CB, A);
      break;
      case TASKALLOC:
        DE.RTF = RTF;
        /*
        The task is created by the taskalloc function, and it returns a
        pointer to the task (check handleTaskRegion documentation ). 
        We need to obtain its data environment. Since this pointer is used in
        functions that are not analyzable, and for purposes of this analysis we 
        are not interested on what happens inside those functions, we will
        add the nocapture attribute. A pointer is captured by the call if it
        makes a copy of any part of the pointer that outlives the call.
        */
        for(auto &U : CB.uses()) {
          auto *User = U.getUser();
          // LLVM_DEBUG(dbgs() << "--- " << *U << "\n");
          // LLVM_DEBUG(dbgs() << "------ " << *User << "\n");
          // LLVM_DEBUG(dbgs() << "------ " << *(U.get()) << "\n");
          if(auto *CBI = dyn_cast<CallInst>(User)) {
            auto *Callee = CBI->getCalledFunction();
            if(isTaskFunction(Callee)) {
              /// Add the attribute nocapture to the returned value
              CBI->addParamAttr(U.getOperandNo(), Attribute::NoCapture);
            }
          }
        }
        /// Lets handle the task in the updateImpl function
        break;
      case OTHER: {
        LLVM_DEBUG(dbgs() << "Other instruction - FOUND\n");
        /// Unknown caller or declarations are not analyzable, we give up.
        if (!Callee || !A.isFunctionIPOAmendable(*Callee)) {
          indicatePessimisticFixpoint();
          LLVM_DEBUG(dbgs() <<"Unknown caller or declarations are not analyzable, we give up.\n");
          return;
        }
        /// If the callee is known and can be used in IPO. Run AA on the callee.
        LLVM_DEBUG(dbgs() <<"Doing IPA for function.\n");
        auto *DEF = A.getAAFor<AADataEnv>(
            *this, IRPosition::function(*Callee), DepClassTy::NONE);
        // A.recordDependence(TDGInfoCS, *this, DepClassTy::REQUIRED, true);
        // /// If the TDGInfoCS fails, it means that we couldnt build a TDG for the function.
        // if (!TDGInfoCS.getState().isValidState()) {
        //   indicatePessimisticFixpoint();
        //   LLVM_DEBUG(dbgs() <<"[AADataEnvCallSite] TDGInfoCS is invalid\n");
        // }
      }
      break;
      /// Default
      default: {
        // LLVM_DEBUG(dbgs() << "Other instruction - FOUND\n");
      }
      break;
    }
  }

  ChangeStatus updateImpl(Attributor &A) override {
    ChangeStatus Changed = ChangeStatus::UNCHANGED;
    CallBase &CB = cast<CallBase>(getAssociatedValue());
    Function *Callee = getAssociatedFunction();
    LLVM_DEBUG(dbgs() << "[AADataEnvCallSite] updateImpl: " << Callee->getName() << "\n");

    switch (RTF) {
      break;
      case PARALLEL:
        // handleParallelRegion(CB);
      break;
      case TASKALLOC:
        Changed |= handleTaskRegion(CB, A);
      break;
      default: {
        // LLVM_DEBUG(dbgs() << "Other instruction - FOUND\n");
      }
      break;
    }
    return Changed;
  }

  ChangeStatus manifest(Attributor &A) override {
    ChangeStatus Changed = ChangeStatus::UNCHANGED;
    LLVM_DEBUG(dbgs() << "[AADataEnvCallSite] Manifest\n");
    return Changed;
  }

  /// Private attributes
  RTFunction RTF;
  /// Map that contains the range accesses of the task
  /// DenseMap to map RangeTy objects to Value objects
  // llvm::DenseMap<AA::RangeTy, const llvm::Value *> RangeAccesses;
  // SmallVector<RangeAccess, 4> RangeAccesses;
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
  case IRPosition::IRP_FLOAT:
    llvm_unreachable(
        "AATDGInfo can only be created for function/callsite position!");
  case IRPosition::IRP_CALL_SITE:
    AA = new (A.Allocator) AADataEnvCallSite(IRP, A);
    break;
  case IRPosition::IRP_FUNCTION:
    AA = new (A.Allocator) AADataEnvFunction(IRP, A);
    break;
  }
  return *AA;
}

const char AADataEnv::ID = 0;

/// ARTSTransform
bool ARTSTransform::run() {
  bool changed = false;
  /// Aux variables
  LoopInfo *LI = nullptr;
  DominatorTree *DT = nullptr;
  /// Region counter
  unsigned int ParallelRegion = 0;
  unsigned int TaskRegion = 0;
  /// Get the main function
  LLVM_DEBUG(dbgs() << TAG << "Analyzing main function\n");
  auto *MainFunction = M.getFunction("main");
  if(!MainFunction) {
    LLVM_DEBUG(dbgs() << TAG << "Main function not found\n");
    return changed;
  }
  /// Identify EDT regions
  LLVM_DEBUG(dbgs() << TAG << "Identifying EDT regions\n");
  /// For each function of the module
  for(auto &F : M) {
    if(!A.isFunctionIPOAmendable(F))
      continue;
    // LLVM_DEBUG(dbgs() << TAG << "Function: " << F << "\n\n");
    /// Get entry block
    BasicBlock *CurBB = &(F.getEntryBlock());
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
        RTFunction RTF = getRTFunction(Callee);
        switch (RTF) {
          case PARALLEL: {
            BasicBlock *Parallel = 
              SplitBlock(CurI->getParent(), CurI, DT, LI, nullptr,
                         "par.region." + std::to_string(ParallelRegion));
            /// Split block again at the next instruction
            CurI = CurI->getNextNonDebugInstruction();
            BasicBlock *ParallelDone = 
              SplitBlock(Parallel, CurI, DT, LI, nullptr,
                         "par.done." + std::to_string(ParallelRegion));
            NextBB = ParallelDone;
            ParallelRegion++;
          }
          break;
          case TASKALLOC: {
            /// Split block at __kmpc_omp_task_alloc
            BasicBlock *Task = 
              SplitBlock(CurI->getParent(), CurI, DT, LI, nullptr,
                         "task.region." + std::to_string(TaskRegion));
            /// Find the task call
            while ((CurI = CurI->getNextNonDebugInstruction())) {
              auto *TCB = dyn_cast<CallBase>(CurI);
              if(TCB && getRTFunction(TCB->getCalledFunction()) == RTFunction::TASK)
                break;
            }
            assert(CurI && "Task RT call not found");
            /// Split block again at the next instruction
            CurI = CurI->getNextNonDebugInstruction();
            BasicBlock *TaskDone = 
              SplitBlock(Task, CurI, DT, LI, nullptr,
                         "task.done." + std::to_string(TaskRegion));
            NextBB = TaskDone;
            TaskRegion++;
            break;
          }
          break;

          default:
            continue;
          break;
        }
      } while ((CurI = CurI->getNextNonDebugInstruction()));

    } while ((CurBB = NextBB));

    // LLVM_DEBUG(dbgs() << TAG << "Function: " << F.getName() << " DONE\n\n");
  }

  LLVM_DEBUG(dbgs() << "\n" << TAG << "Module: \n" << M << "\n");

  // /// Create and initialize ARTSIRBuilder
  // ARTSIRBuilder ARTSIRB(M);
  // ARTSIRB.initialize();
  // Function *EDT = ARTSIRB.createEDT("fib");
  // LLVM_DEBUG(dbgs() << TAG << "EDT: \n" << *EDT << "\n");
  
  changed = true;
  // /// Create AA and run it
  // A.getOrCreateAAFor<AADataEnv>(
  //         IRPosition::function(*MainFunction), /* QueryingAA */ nullptr,
  //         DepClassTy::NONE, /* ForceUpdate */ false,
  //         /* UpdateAfterInit */ false);

  // changed |= runAttributor();
  return changed;
}

bool ARTSTransform::runAttributor() {
  ChangeStatus Changed = A.run();
  LLVM_DEBUG(dbgs() << "[Attributor] Done, result: " << Changed << ".\n");

  return Changed == ChangeStatus::CHANGED;
}


// ARTS Transformation pass
PreservedAnalyses ARTSTransformPass::run(Module &M, ModuleAnalysisManager &AM) {
  /// Command line options
  if (DisableARTSTransformation)
    return PreservedAnalyses::all();
  if (PrintModuleBeforeOptimizations)
    LLVM_DEBUG(dbgs() << TAG << "Module before ARTSTransform Module Pass:\n" << M);

  /// Run the ARTSTransform Module Pass
  LLVM_DEBUG(dbgs() << TAG << "Run the ARTSTransform Module Pass\n");
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
  InformationCache InfoCache(M, AG, Allocator, nullptr);
  AttributorConfig AC(CGUpdater);
  Attributor A(Functions, InfoCache, AC);

  /// Run ARTSTransform
  ARTSTransform ARTSTransform(M, A, Functions);
  Changed |= ARTSTransform.run();

  LLVM_DEBUG(dbgs() << TAG << "Module after ARTSTransform Module Pass:\n" << M);
  if (Changed)
    return PreservedAnalyses::none();

  return PreservedAnalyses::all();
}


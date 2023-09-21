
#include "llvm/Transforms/ARTS/ARTSTransformer.h"
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

/// ---------------------------- COMMAND LINE OPTIONS ---------------------------- ///
static cl::opt<bool> DisableARTSTransformation(
    "arts-disable", cl::desc("Disable transformation to ARTS."),
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
static RTFunction getRTFunction(Function *F) {
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

static RTFunction getRTFunction(CallBase &CB) {
  auto *Callee = CB.getCalledFunction();
  if(!Callee)
    return RTFunction::OTHER;
  return getRTFunction(Callee);
}

static bool isTaskFunction(Function *F) {
  auto RT = getRTFunction(F);
  if(RT == RTFunction::TASK || RT == RTFunction::TASKDEP || RT == RTFunction::TASKWAIT )
    return true;
  return false;
}

/// ---------------------------- ABSTRACT ATTRIBUTES ---------------------------- ///
/// ARTSInformationCache
/// It stores ARTS related information that the attributor can use
struct ARTSInformationCache : public InformationCache {
  ARTSInformationCache(Module &M, AnalysisGetter &AG,
                       BumpPtrAllocator &Allocator, 
                       SetVector<Function *> *Functions)
      : InformationCache(M, AG, Allocator, Functions),
        ARTSBuilder(M), ARTSTransform(M, Functions) {

    ARTSBuilder.initialize();
  }

  /// The ARTSIRBuilder instance.
  ARTSIRBuilder ARTSBuilder;
  /// ARTSTransformer pointer
  ARTSTransformer ARTSTransform;
}

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

  const std::string getDataEnv() const {
    std::string Str("---DATA ENVIRONMENT for ");
    Str += std::string("Function: ") + std::string(DE.getCBName()) + std::string("\n");
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

  /// Data environment attribute
  DataEnv DE;
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

  ChangeStatus handleParallelRegion(CallBase &CB, Attributor &A) {
    LLVM_DEBUG(dbgs() << "[AADataEnvCallSite] Parallel region - FOUND\n");
    if(!DE.F)
      DE.F = dyn_cast<Function>(CB.getArgOperand(2));

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

      /// Do we need to consider private variables?
    }

    /// Run AA on the outline function
    // auto *DEF = A.getAAFor<AADataEnv>(
    //   *this, IRPosition::function(*DE.F), DepClassTy::NONE);
    // if (!DEF->getState().isValidState()) {
    //   indicatePessimisticFixpoint();
    //   LLVM_DEBUG(dbgs() <<"[AADataEnvCallSite] DEF is invalid\n");
    //   return;
    // }
    /// TODO: Check if it is a fixpoint, if so, append the data environment
    /// of the outline function. Otherwise, record 

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
    /// Aux variables
    if(!DE.F)
      DE.F = dyn_cast<Function>(CB.getArgOperand(5));
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
    return ChangeStatus::CHANGED;
  }

  void initialize(Attributor &A) override {
    CallBase &CB = cast<CallBase>(getAssociatedValue());
    Function *Callee = getAssociatedFunction();
    LLVM_DEBUG(dbgs() <<"[AADataEnvCallSite] ----- initialize: " << Callee->getName() << "\n");
    DE.CB = &CB;
    DE.RTF = getRTFunction(Callee);
    switch (DE.RTF) {
      case SET_NUM_THREADS: {
        auto *Arg = CB.getArgOperand(0);
        if(auto *NumThreads = dyn_cast<ConstantInt>(Arg)) {
          LLVM_DEBUG(dbgs() << "Number of threads: "
                            << NumThreads->getZExtValue() << "\n");
        }
        indicateOptimisticFixpoint();
      }
      break;
      case PARALLEL:
        handleParallelRegion(CB, A);
      break;
      case TASKALLOC: {
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
            if(isTaskFunction(Callee)) {
              /// Add the attribute nocapture to the returned value
              CBI->addParamAttr(U.getOperandNo(), Attribute::NoCapture);
            }
          }
        }
        /// Lets handle the task in the updateImpl function
      } break;
      case OTHER: {
        LLVM_DEBUG(dbgs() << "Other instruction - FOUND\n");
        /// Unknown caller or declarations are not analyzable, we give up.
        if (!Callee || !A.isFunctionIPOAmendable(*Callee)) {
          indicatePessimisticFixpoint();
          LLVM_DEBUG(dbgs() <<"Unknown caller or declarations are not analyzable, we give up.\n");
          return;
        }
        /// If the callee is known and can be used in IPO. Run AA on the callee.
        // LLVM_DEBUG(dbgs() <<"Doing IPA for function.\n");
        // auto *DEF = A.getAAFor<AADataEnv>(
        //     *this, IRPosition::function(*Callee), DepClassTy::NONE);
        // A.recordDependence(TDGInfoCS, *this, DepClassTy::REQUIRED, true);
        // /// If the TDGInfoCS fails, it means that we couldnt build a TDG for the function.
        // if (!TDGInfoCS.getState().isValidState()) {
        //   indicatePessimisticFixpoint();
        //   LLVM_DEBUG(dbgs() <<"[AADataEnvCallSite] TDGInfoCS is invalid\n");
        // }
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

    switch (RTF) {
      break;
      case PARALLEL:
        /// The parallel region should be handled in the initialize function
        // handleParallelRegion(CB);
      break;
      case TASKALLOC:
        Changed |= handleTaskRegion(CB, A);
      break;
      default: {
        LLVM_DEBUG(dbgs() << "Other instruction - FOUND\n");
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
        "AAToARTS can only be created for function/callsite position!");
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

struct AAToARTSFloating : AAToARTS {
  AAToARTSFloating(const IRPosition &IRP, Attributor &A)
      : AAToARTS(IRP, A) {}

  /// See AbstractAttribute::getAsStr()
  const std::string getAsStr() const override {
    if (!isValidState())
      return "<invalid>";

    std::string Str("AAToARTSFloating: ");

    return Str + std::string("OK");
  }

  void initialize(Attributor &A) override {
    /// Cast value to BasicBlock
    BasicBlock &BB = cast<BasicBlock>(getAnchorValue());

    auto &ARTSInfoCache = static_cast<ARTSInformationCache &>(A.getInfoCache());
    /// Get the EDT info for the BB and its data environment
    auto *EI = ARTSInfoCache.ARTSTransform.getEDTInfo(BB);
    auto &DE = EI->DE;
    if(!EI) {
      LLVM_DEBUG(dbgs() <<"[AAToARTSFloating] EI is null\n");
      indicatePessimisticFixpoint();
      return;
    }
    LLVM_DEBUG(dbgs() <<"[AAToARTSFloating] BasicBlock: " << BB << "\n");
    LLVM_DEBUG(dbgs() <<"[AAToARTSFloating] Analyzing: " << DE.getCBName() << "\n");
    /// Run DEAA on the data environment
    auto *DEAA = A.getAAFor<AADataEnv>(
      *this, IRPosition::callsite_function(*DE.CB), DepClassTy::OPTIONAL);
    if (!DEAA->getState().isValidState() || !DEAA->getState().isAtFixpoint()) {
      LLVM_DEBUG(dbgs() <<"[AAToARTSFloating] DEAA is invalid or not at fixpoint\n");
      indicatePessimisticFixpoint();
      return;
    }
    /// Convert DE Function to CallBase
    // CallBase &CB = cast<CallBase>(*DE.getCBFunction()->getEntryBlock().getFirstInsertionPt());
  }

  ChangeStatus updateImpl(Attributor &A) override {
    ChangeStatus Changed = ChangeStatus::UNCHANGED;
    LLVM_DEBUG(dbgs() << "[AAToARTSFloating] updateImpl: " << "\n");
    
    return Changed;
  }

  ChangeStatus manifest(Attributor &A) override {
    ChangeStatus Changed = ChangeStatus::UNCHANGED;
    LLVM_DEBUG(dbgs() << "[AAToARTSFloating] Manifest\n");
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
    AA = new (A.Allocator) AAToARTSFloating(IRP, A);
    break;
  }
  return *AA;
}

const char AAToARTS::ID = 0;


/// ---------------------------- ARTS TRANSFORM ---------------------------- ///
bool ARTSTransformer::run(Attributor &A) {
  bool changed = false;
  
  /// Identify EDT regions
  changed |= identifyEDTRegions();
  // Function *EDT = ARTSIRB.createEDT("fib");
  // LLVM_DEBUG(dbgs() << TAG << "EDT: \n" << *EDT << "\n");
  if(!changed) {
    LLVM_DEBUG(dbgs() << TAG << "No EDT regions found\n");
    return changed;
  }
  /// Run attributor on the EDTRegions
  for(auto &E : EDTRegions) {
    A.getOrCreateAAFor<AAToARTS>(
          IRPosition::value(E.first), /* QueryingAA */ nullptr,
          DepClassTy::NONE, /* ForceUpdate */ false,
          /* UpdateAfterInit */ false);
  }
  changed |= runAttributor(A);
  return changed;
}

bool ARTSTransformer::runAttributor(Attributor &A) {
  ChangeStatus Changed = A.run();
  LLVM_DEBUG(dbgs() << "[Attributor] Done, result: " << Changed << ".\n");

  return Changed == ChangeStatus::CHANGED;
}

bool ARTSTransformer::identifyEDTRegions() {
  bool Changed = false;
  /// This function identifies the regions that can be transformed to EDTs.
  /// If a basic block contains a call to the following functions, it
  /// most likely can be transformed to an EDT:
  /// - __kmpc_fork_call.
  /// - __kmpc_omp_task_alloc or __kmpc_omp_task_alloc_with_deps
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
  ///   br label %par.region.0
  /// par.region.0:
  ///   %1 = call i32 @__kmpc_fork_call(i32 1, i32 (i32, i8*)* @__omp_offloading_fib, i8* %0)
  ///   br label %par.done.0
  /// par.done.0:
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
            /// Add the parallel region to the map
            EDTInfo EI(RTF, CB);
            EDTRegions.insert(std::make_pair(Parallel, EI));
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
            /// Add the task region to the map
            EDTInfo EI(RTF, CB);
            EDTRegions.insert(std::make_pair(Task, EI));
            TaskRegion++;
            break;
          }
          break;
          case OTHER: {
            if (!A.isFunctionIPOAmendable(*Callee))
              continue;
            /// Get return type of the callee
            Type *RetTy = Callee->getReturnType();
            /// If the return type is void, we are not interested
            if(RetTy->isVoidTy())
              continue;
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
  }
  LLVM_DEBUG(dbgs() << TAG << "Identifying EDT regions done with " 
                    << EDTRegions.size() << " regions\n");
  for(auto &E : EDTRegions) {
    LLVM_DEBUG(dbgs() << TAG << "Region: " << *(E.first));
    // LLVM_DEBUG(dbgs() << TAG << "Region type: " << E.second.RTF << "\n");
    LLVM_DEBUG(dbgs() << TAG << "Region function: " << E.second.DE.getCBName() << "\n\n");
  }
  LLVM_DEBUG(dbgs() << "\n" << TAG << "Module: \n" << M << "\n");

  Changed = true;
  return Changed;
}

/// ---------------------------- ARTS TRANSFORMATION PASS ---------------------------- ///
PreservedAnalyses ARTSTransformPass::run(Module &M, ModuleAnalysisManager &AM) {
  /// Command line options
  if (DisableARTSTransformation)
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
  // InformationCache InfoCache(M, AG, Allocator, nullptr);
  ARTSInformationCache InfoCache(M, AG, Allocator, Functions);
  AttributorConfig AC(CGUpdater);
  Attributor A(Functions, InfoCache, AC);

  /// Run ARTSTransform
  Changed |= InfoCache.ARTSTransform.run(A);

  LLVM_DEBUG(dbgs() << TAG << "Module after ARTSTransformer Module Pass:\n" << M);
  if (Changed)
    return PreservedAnalyses::none();
  return PreservedAnalyses::all();
}

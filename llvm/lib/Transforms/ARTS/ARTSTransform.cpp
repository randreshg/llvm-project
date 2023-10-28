
#include "llvm/Transforms/ARTS/ARTSTransform.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringRef.h"
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
                     /* AllowAlloca */ false, /* AllocaBlock */ nullptr,
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
  void getDominatedBBs(BasicBlock *FromBB, DominatorTree &DT,
                       BlockSequence &DominatedBlocks) {
    Function &F = *FromBB->getParent();
    for (auto &ToBB : F) {
      if (DT.dominates(FromBB, &ToBB))
        DominatedBlocks.push_back(&ToBB);
    }
  }

  /// Remove values interface
  void removeValue(Value *V, bool RecursiveRemove = false, bool RecursiveUndef = true, 
                   Instruction *Exclude = nullptr) {
    if (isa<UndefValue>(V) || V == Exclude)
      return;
    /// Instructions
    if(auto *I = dyn_cast<Instruction>(V)) {
      /// Call instructions
      if (auto *CBI = dyn_cast<CallBase>(I)) {
        // LLVM_DEBUG(dbgs() << "   - Removing call instruction: " << *CBI << "\n"
        //                   << "     in: "<< CBI->getParent()->getName() << "\n");
        /// Iterate through the arguments and replace them with undef using int itr
        for (uint32_t ArgItr = 0; ArgItr < CBI->data_operands_size(); ArgItr++) {
          Value *Arg = CBI->getArgOperand(ArgItr);
          if (!isa<PointerType>(Arg->getType()))
            continue;

          removeValue(Arg, RecursiveRemove, RecursiveUndef, CBI);
        }
      }
      // LLVM_DEBUG(dbgs() << "   - Removing instruction: " << *I << "\n");
      replaceValueWithUndef(I, RecursiveRemove, RecursiveUndef, Exclude);
      I->eraseFromParent();
      return;
    }

    replaceValueWithUndef(V, RecursiveRemove, RecursiveUndef, Exclude);
    /// Global variables are not instructions, but we still need to remove them
    if (GlobalVariable *GV = dyn_cast<GlobalVariable>(V)) {
      // LLVM_DEBUG(dbgs() << "   - Removing global variable: " << *GV << "\n");
      GV->eraseFromParent();
      return;
    }

    /// Function
    if (Function *F = dyn_cast<Function>(V)) {
      // LLVM_DEBUG(dbgs() << "   - Removing function: " << *F << "\n");
      F->eraseFromParent();
      return;
    }
  }

  void removeValues() {
    // LLVM_DEBUG(dbgs() << "\n" << TAG << "Removing values\n");
    for (auto *V : ValuesToRemove) {
      removeValue(V, true, true);
    }
  }

  /// Function to replace uses of a Value with UndefValue. 
  /// - Instructions can be removed if requested.
  /// - The processs can also be performed in a recursive way by replacing
  ///   uses of the instructions that use the value with UndefValue.
  void replaceValueWithUndef(Value *V, bool RemoveInsts = false, bool Recursive = true,
                             Instruction *Exclude = nullptr) {
    /// If the value is undef, we dont need to do anything
    if (isa<UndefValue>(V))
      return;

    // LLVM_DEBUG(dbgs() << "  - Replacing uses of: " << *V << "\n");
    // Create a worklist to keep track of instructions to be removed
    SmallVector<Instruction *, 16> Worklist;
    // Initialize the worklist with all uses of the argument
    for (auto &Use : V->uses()) {
      if (Instruction *UserInst = dyn_cast<Instruction>(Use.getUser()))
        Worklist.push_back(UserInst);
    }
  
    // Replace uses with UndefValue and mark instructions for removal
    V->replaceAllUsesWith(UndefValue::get(V->getType()));
    while (!Worklist.empty()) {
      Instruction *Inst = Worklist.pop_back_val();
      if(Exclude || Inst == Exclude)
        continue;
      // LLVM_DEBUG(dbgs() << "   - Replacing: " << *Inst << "\n");
      // Add users of this instruction to the worklist for further processing
      if(Recursive) {
        for (auto &Use : Inst->uses()) {
          if (Instruction *UserInst = dyn_cast<Instruction>(Use.getUser()))
            Worklist.push_back(UserInst);
        }
      }
    
      // Replace uses of the argument with UndefValue
      Value *Undef = UndefValue::get(Inst->getType());
      Inst->replaceAllUsesWith(Undef);
      // Mark the instruction for removal
      if(RemoveInsts) {
        // LLVM_DEBUG(dbgs() << "    -> Instruction removed\n");
        Inst->eraseFromParent();
      }
    }
  }

  /// Given a basic block, this function creates a function that contains
  /// the instructions of the BB. The function is then inserted to the
  /// module.
  Function *
  createFunction(DominatorTree *DT, BasicBlock *FromBB, bool DTAnalysis = false,
                 std::string FunctionName = "",
                 SmallVector<Value *, 0> *ExcludeArgsFromAggregate = nullptr) {
    Function &F = *FromBB->getParent();
    AssumptionCache *AC = AG.getAnalysis<AssumptionAnalysis>(F, true);
    CodeExtractorAnalysisCache CEAC(F);
  
    /// Collect blocks
    BlockSequence Region;
    /// Get all BBs that are dominated by FromBB
    if (DTAnalysis)
      getDominatedBBs(FromBB, *DT, Region);
    else
      Region.push_back(FromBB);

    /// Extract code from the region
    CodeExtractor CE(Region, DT, /* AggregateArgs */ false, /* BFI */ nullptr,
                      /* BPI */ nullptr, AC, /* AllowVarArgs */ false,
                      /* AllowAlloca */ true, /* AllocaBlock */ nullptr);

    assert(CE.isEligible() && "Expected Region outlining to be possible!");
  
    if (ExcludeArgsFromAggregate)
      for (auto *V : *ExcludeArgsFromAggregate)
        CE.excludeArgFromAggregate(V);

    /// Generate function
    Function *OutF = CE.extractCodeRegion(CEAC);
    if(FunctionName != "")
      OutF->setName(FunctionName);
  
    // LLVM_DEBUG(dbgs() << TAG << "Function created: " << OutF->getName() << "\n");
    return OutF;
  }

  /// Analyzes the outlined region, replaces the RT call with a call to the
  /// outlined function, which is also modified to remove the arguments that
  /// are not needed.
  bool handleParallelOutlinedRegion(CallBase *RTCall) {
    /// Analyze outlined region
    const uint32_t ParallelOutlinedFunctionPos = 2;
    const uint32_t KeepArgsFrom = 2;
    const uint32_t KeepCallArgsFrom = 3;
    Function *OldFn =
        dyn_cast<Function>(RTCall->getArgOperand(ParallelOutlinedFunctionPos));
    /// Remove arguments from the outlined function
    for (uint32_t ArgItr = 0; ArgItr < KeepArgsFrom; ArgItr++) {
      Value *Arg = OldFn->args().begin() + ArgItr;
      removeValue(Arg, true, false);
    }

    /// Generate new outlined function
    SmallVector<Type *, 16> NewArgumentTypes;
    SmallVector<AttributeSet, 16> NewArgumentAttributes;

    // Collect replacement argument types and copy over existing attributes.
    for (auto ArgItr = KeepArgsFrom; ArgItr < OldFn->arg_size(); ArgItr++) {
      Value *Arg = OldFn->args().begin() + ArgItr;
      NewArgumentTypes.push_back(Arg->getType());
    }

    // Construct the new function type using the new arguments types.
    FunctionType *OldFnTy = OldFn->getFunctionType();
    Type *RetTy = OldFnTy->getReturnType();
    FunctionType *NewFnTy =
        FunctionType::get(RetTy, NewArgumentTypes, OldFnTy->isVarArg());
    LLVM_DEBUG(dbgs() << " - Function rewrite '" << OldFn->getName()
                      << "' from " << *OldFn->getFunctionType() << " to "
                      << *NewFnTy << "\n");

    // Create the new function body and insert it into the module.
    Function *NewFn = Function::Create(NewFnTy, OldFn->getLinkage(),
                                       OldFn->getAddressSpace(), "");
    OldFn->getParent()->getFunctionList().insert(OldFn->getIterator(), NewFn);
    // NewFn->takeName(OldFn);
    NewFn->setName("parallel.edt");
    OldFn->setSubprogram(nullptr);

    // Create Parallel EDT
    EdtInfo &ParallelEDT = *AT.insertEdt(EdtInfo::PARALLEL, NewFn);

    // Since we have now created the new function, splice the body of the old
    // function right into the new function, leaving the old rotting hulk of the
    // function empty.
    NewFn->splice(NewFn->begin(), OldFn);

    // Collect the new argument operands for the replacement call site.
    CallBase *OldCB = dyn_cast<CallBase>(RTCall);
    const AttributeList &OldCallAttributeList = OldCB->getAttributes();
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
    LLVMContext &Ctx = OldFn->getContext();
    CallBase *NewCB = NewCI;
    NewCB->setCallingConv(OldCB->getCallingConv());
    NewCB->takeName(OldCB);
    NewCB->setAttributes(AttributeList::get(
        Ctx, OldCallAttributeList.getFnAttrs(),
        OldCallAttributeList.getRetAttrs(), NewArgOperandAttributes));

    // Rewire the function arguments.
    Argument *OldFnArgIt = OldFn->arg_begin() + KeepArgsFrom;
    Argument *NewFnArgIt = NewFn->arg_begin();
    for (unsigned OldArgNum = KeepArgsFrom; OldArgNum < OldFn->arg_size();
         ++OldArgNum, ++OldFnArgIt) {
      NewFnArgIt->takeName(&*OldFnArgIt);
      OldFnArgIt->replaceAllUsesWith(&*NewFnArgIt);
      /// Add to Parallel EDT Data Environment
      Argument *Arg = &*OldFnArgIt;
      Type *ArgType = NewFnArgIt->getType();
      /// For now assume that if it is a pointer, it is a shared variable
      if (PointerType *PT = dyn_cast<PointerType>(ArgType))
        AT.insertArgToDE(Arg, DataEnv::SHARED, ParallelEDT);
      /// If not, it is a first private variable
      else
        AT.insertArgToDE(Arg, DataEnv::FIRSTPRIVATE, ParallelEDT);

      /// NOTES: The outline function or the variable should have attributes
      /// that provide more information about the lifetime of the variable. It
      /// is also important to consider the underlying type of the variable.
      /// There may be cases, where there is a firstprivate variable that is a
      /// pointer. For this case, the pointer is private, but the data it points
      /// to is shared.
      ++NewFnArgIt;
    }

    // Eliminate the instructions *after* we visited all of them.
    OldCB->replaceAllUsesWith(NewCB);
    OldCB->eraseFromParent();
    ValuesToRemove.push_back(OldFn);

    assert(NewFn->isDeclaration() == false && "New function is a declaration");
    LLVM_DEBUG(dbgs() << " - New CB: " << *NewCB << "\n");

    /// Identify EDT for new function
    if(!identifyEDTs(*NewFn))
      return false;
    return true;
  }

  bool handleTaskOutlinedRegion(CallBase *CB) {
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

    const uint32_t TaskOutlinedFunctionPos = 5;
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
    DataEnv TaskInfoDE;
    BasicBlock *BB = CB->getParent();
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

      /// Private variables
      if(Offset >= TaskDataSize) {
        TaskInfoDE.Privates.insert(Val);
        continue;
      }
      /// Shared variables
      TaskInfoDE.Shareds.insert(Val);
    }

    /// Analyze Outlined Function
    Function *OutlinedFn =
      dyn_cast<Function>(CB->getArgOperand(TaskOutlinedFunctionPos));
    Argument *TaskData = dyn_cast<Argument>(OutlinedFn->arg_begin() + 1);
    /// This assumes the 'desaggregation' happens in the first basic block
    BasicBlock *EntryBB = &OutlinedFn->getEntryBlock();
    BasicBlock::iterator Itr = EntryBB->begin();
    auto *TaskDataPtr = &*Itr;

    // Iterate from the second instruction onwards
    ++Itr;
    for (; Itr != BB->end(); ++Itr) {
      Instruction &I = *Itr;
      if(!isa<LoadInst>(&I))
        continue;
    
      auto *L = cast<LoadInst>(&I);
      auto *Val = L->getPointerOperand();
      int64_t Offset = -1;
      auto *BasePointer = GetPointerBaseWithConstantOffset(Val, Offset, DL);

      if(Offset == -1)
        continue;

      auto Cond = (Offset >= TaskDataSize && BasePointer == TaskData) || 
                        (Offset <  TaskDataSize && BasePointer == TaskDataPtr);
      if(!Cond) 
        continue;

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

    /// Create the new function body and insert it into the module.
    Function *NewFn = Function::Create(NewFnTy, OutlinedFn->getLinkage(),
                                       OutlinedFn->getAddressSpace());
    OutlinedFn->getParent()->getFunctionList().insert(
      OutlinedFn->getIterator(), NewFn);
    NewFn->takeName(OutlinedFn);
    NewFn->setName("task.edt");
    OutlinedFn->setSubprogram(nullptr);
    OutlinedFn->setMetadata("dbg", nullptr);

    /// Create Edt for new function
    EdtInfo &TaskEdt = *AT.insertEdt(EdtInfo::TASK, NewFn);

    /// Move BB to new function
    NewFn->splice(NewFn->begin(), OutlinedFn);
    /// Rewire the function arguments.
    Argument *NewFnArgItr = NewFn->arg_begin();
    for (auto TDItr : ValueToOffsetTD) {
      Value *V = OffsetToValueOF[TDItr.second];
      V->replaceAllUsesWith(NewFnArgItr);
      /// Add to Task data environment
      DataEnv::Type Type = TaskInfoDE.getType(TDItr.first);
      AT.insertArgToDE(&*NewFnArgItr, Type, TaskEdt);
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
    LLVM_DEBUG(dbgs() << " - New CB: " << *NewCI << "\n");
    LLVM_DEBUG(dbgs() << TaskEdt.DE << "\n");

    /// Remove Argument 0 and 1 from Original Task Outlined Function
    replaceValueWithUndef(OutlinedFn->getArg(0), true);
    replaceValueWithUndef(OutlinedFn->getArg(1), true);
    ValuesToRemove.push_back(CB);
  
    // Iterate through the basic blocks and check/replace terminators
    for (auto &NewBB : *NewFn) {
      auto *Terminator = NewBB.getTerminator();
      if (isa<ReturnInst>(Terminator)) {
        IRBuilder<> Builder(Terminator);
        Builder.CreateRetVoid();
        Terminator->eraseFromParent();
      }
    }

    if(!identifyEDTs(*NewFn))
      return false;
    return true;
  }

  /// Handles the done region and return next BB to analyze
  BasicBlock *handleDoneRegion(BasicBlock *DoneBB, DominatorTree *DT, 
                               std::string PrefixName, std::string SuffixBB) {
    LLVM_DEBUG(dbgs() << "\n" <<  TAG << "Handling done region\n");
    /// Get first instruction of BB to analyze if we need a new function
    Instruction *FirstI = &*DoneBB->begin();

    /// If the BB only has a return instruction, we can just remove it, and
    /// add the return instruction to the predecessor
    // if (isa<ReturnInst>(FirstI)) {
    //   LLVM_DEBUG(dbgs() << TAG << " - Removing return instruction\n");
    //   /// Get predecessor and add return void
    //   BasicBlock *PredBB = DoneBB->getUniquePredecessor();
    //   assert(PredBB && "Expected only one predecessor");
    //   auto *PredBBTerminator = PredBB->getTerminator();
    //   IRBuilder<> Builder(PredBBTerminator);
    //   Builder.CreateRetVoid();
    //   /// Remove BB
    //   PredBBTerminator->eraseFromParent();
    //   DoneBB->eraseFromParent();
    //   return PredBB;
    // }

    /// If it is a callbase, check if its a call to a RT function
    if(auto *CB = dyn_cast<CallBase>(FirstI)) {
      if(OMPInfo::isRTFunction(*CB))
        return DoneBB;
      /// TODO: What about other callbase?
    }

    /// Handle other instructions
    auto DoneFnName = PrefixName + "edt.done";
    Function *DoneFunction = createFunction(DT, DoneBB, true, DoneFnName);
    // LLVM_DEBUG(dbgs() << "Done Function: " << *DoneFunction << "\n");
    /// Analyze Data Environment
    BlockSequence DoneRegion;
    for (auto &BB : *DoneFunction)
      DoneRegion.push_back(&BB);
    SetVector<Value *> Inputs, Outputs, Sinks;
    getInputsOutputs(DoneRegion, DT, Inputs, Outputs, Sinks);

    /// Get caller of the Done function, and rename BB to par.done
    auto DoneBBName = PrefixName + "done." + SuffixBB;
    CallBase *DoneCB = dyn_cast<CallBase>(DoneFunction->user_back());
    auto *NewDoneBB = DoneCB->getParent();
    NewDoneBB->setName(DoneBBName);

    /// Analyze the instructions of the Done function
    identifyEDTs(*DoneFunction);
    return NewDoneBB->getNextNode();
  }

  void removeLifetimeMarkers(Function &F) {
    for (auto &BB : F) {
      auto InstIt = BB.begin();
      auto InstEnd = BB.end();

      while (InstIt != InstEnd) {
        auto NextIt = InstIt;
        ++NextIt;

        if (auto *IT = dyn_cast<IntrinsicInst>(&*InstIt)) {
          switch (IT->getIntrinsicID()) {
          case Intrinsic::lifetime_start:
          case Intrinsic::lifetime_end:
            IT->eraseFromParent();
            break;
          default:
            break;
          }
        }

        InstIt = NextIt;
      }
    }
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

    /// Remove of lifetime markers
    removeLifetimeMarkers(F);
  
    /// Aux variables
    LoopInfo *LI = nullptr;
    DominatorTree *DT = AG.getAnalysis<DominatorTreeAnalysis>(F);

    /// Region counter
    uint32_t ParallelRegion = 0;
    uint32_t TaskRegion = 0;
    LLVM_DEBUG(dbgs() << "\n");
    LLVM_DEBUG(dbgs() << TAG << "[identifyEDTs] Identifying EDT regions for: " << F.getName() << "\n");

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
        OMPInfo::RTFType RTF = OMPInfo::getRTFunction(*CB);
        switch (RTF) {
        case OMPInfo::PARALLEL: {
          LLVM_DEBUG(dbgs() << TAG << "Parallel Region Found: " << "\n  "
                            <<  *CB << "\n");

          /// Split block at __kmpc_parallel
          auto ParallelItrStr = std::to_string(ParallelRegion++);
          auto ParallelName = "par.region." + ParallelItrStr;
          BasicBlock *ParallelBB;
          if(CurrentI != &(CurrentBB->front())) {
            ParallelBB = SplitBlock(
              CurrentI->getParent(), CurrentI, DT, LI, nullptr, ParallelName);
          }
          else {
            ParallelBB = CurrentBB;
            ParallelBB->setName(ParallelName);
          }
  
          /// Split block at the next instruction
          CurrentI = CurrentI->getNextNonDebugInstruction();

          /// Analyze Done Region
          if(!isa<ReturnInst>(CurrentI)) {
            BasicBlock *ParallelDone =
              SplitBlock(ParallelBB, CurrentI, DT, LI, nullptr);
            NextBB = handleDoneRegion(ParallelDone, DT, "par.", ParallelItrStr);
          }

          /// Analyze Outlined Region
          handleParallelOutlinedRegion(CB);
        } break;
        case OMPInfo::TASKALLOC: {
          LLVM_DEBUG(dbgs() << TAG << "Task Region Found: " << "\n  "
                            <<  *CB << "\n");

          /// Split block at __kmpc_omp_task_alloc
          auto TaskItrStr = std::to_string(TaskRegion++);
          auto TaskName = "task.region." + TaskItrStr;
          BasicBlock *TaskBB;
          if(CurrentI != &(CurrentBB->front())) {
            TaskBB = SplitBlock(
              CurrentI->getParent(), CurrentI, DT, LI, nullptr, TaskName);
          }
          else {
            TaskBB = CurrentBB;
            TaskBB->setName(TaskName);
          }
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
          CurrentI = NextI;

          /// Analyze Done Region
          if(!isa<ReturnInst>(CurrentI)) {
            BasicBlock *TaskDone =
              SplitBlock(TaskBB, CurrentI, DT, LI, nullptr);
            NextBB = handleDoneRegion(TaskDone, DT, "task.", TaskItrStr);
          }

          /// Analyze Outlined Region
          handleTaskOutlinedRegion(CB);
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
  // LLVM_DEBUG(dbgs() << TAG << EDTs << "\n");
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

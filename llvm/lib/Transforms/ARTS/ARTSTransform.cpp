
#include "llvm/Transforms/ARTS/ARTSTransform.h"
#include "llvm/Frontend/ARTS/ARTSIRBuilder.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/MemoryDependenceAnalysis.h"
#include "llvm/Analysis/MemorySSA.h"
#include "llvm/Analysis/MemorySSAUpdater.h"
#include "llvm/Analysis/ValueTracking.h"

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
    return Changed;
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

  bool handleEDTInitBlock(Attributor &A, Instruction &I, EDTBlock &EB) {
    LLVM_DEBUG(dbgs() << "[AADataEnvFunction] EDT init block: " 
                      << EB.OMP.getCBName() << "\n");
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
    /// Only add the call to the RT function if init block.
    if(!EB.Analyzed) {
      EB.EDT->RWInsts.insert(&I);
      EB.Analyzed = true;
    }
    return true;
  }

  ChangeStatus updateImpl(Attributor &A) override {
    Function &F = cast<Function>(getAnchorValue());
    LLVM_DEBUG(dbgs() << "[AADataEnvFunction] updateImpl: "
                      << F.getName() << "\n");
    /// Get cache
    auto &AIC = static_cast<ARTSInformationCache &>(A.getInfoCache());
    auto &AT = AIC.ARTSTransform;
    /// Get MemorySSA
    MemorySSA &MSSA = AIC.AG.getAnalysis<MemorySSAAnalysis>(F)->getMSSA();
    MemorySSAWalker *Walker = MSSA.getWalker();
    MSSA.dump();

    auto MDA = AIC.AG.getAnalysis<MemoryDependenceAnalysis>(F);
    /// Function's EDT
    // EDTInfo *ParentEDT = AT.getEDTForFunction(&F);
  //   auto checkAndAddValue = [&](Value *Val, EDTInfo *EDT) {
  //     // Type *ValType = Val->getType();
  //     LLVM_DEBUG(dbgs() << "  - Operand: " << *Val << "\n");
  //     // LLVM_DEBUG(dbgs() << "    - Operand type: " << *ValType << "\n");
  //     if(isa<Argument>(Val)) {
  //       // LLVM_DEBUG(dbgs() << "    - It is an Argument\n");
  //       return;
  //     }
  //     auto *ValInst = dyn_cast<Instruction>(Val);
  //     if(!ValInst) {
  //       /// It can be a constant value...
  //       // LLVM_DEBUG(dbgs() << "    - It is not an Instruction\n");
  //       return false;
  //     }
  //     // /// It is also important to consider global variables
  //     // if(auto *GV = dyn_cast<GlobalVariable>(ValInst))
  //     /// Check if the instruction is in the same EDT Data scope
  //     // LLVM_DEBUG(dbgs() << "    - It is an Instruction\n");
  //     auto *EB = AT.getEDTBlock(ValInst->getParent());
  //     EB->Analyzed = true;
  //     if(EB->EDT->ID == EDT->ID) {
  //       LLVM_DEBUG(dbgs() << "    - It is in the same EDT\n");
  //       return;
  //     }
  //     LLVM_DEBUG(dbgs() << "    - It is not in the same EDT\n");
  //     EDT->ExternalValues.insert(ValInst);
  // };

    const std::function<bool(EDTBlock *CurrentEB, Instruction &I)> handleInstruction = 
      [&](EDTBlock *CurrentEB, Instruction &I) {
      LLVM_DEBUG(dbgs() << "  - Handling instruction:\n");
      // LLVM_DEBUG(dbgs() << "  - Memory access: " << *MSSA.getMemoryAccess(&I) << "\n");
      // LLVM_DEBUG(dbgs() << "  - Memory instruction: " << *Walker->getClobberingMemoryAccess(&I) << "\n");
      SmallVector<MemoryAccess *> WorkList{Walker->getClobberingMemoryAccess(&I)};
      SmallSet<MemoryAccess *, 8> Visited;
      /// Get clobbering access. If the memory instruction of the access is
      /// inside the same EDT, we ignore it
      while (!WorkList.empty()) {
        MemoryAccess *MA = WorkList.pop_back_val();
        if (!Visited.insert(MA).second)
          continue;
        /// If it is a live on entry def, we dont care about it.
        if (MSSA.isLiveOnEntryDef(MA)) {
          LLVM_DEBUG(dbgs() << "    - It is a live on entry\n");
          continue;
        }

        if (MemoryDef *Def = dyn_cast<MemoryDef>(MA)) {
          Instruction *DefInst = Def->getMemoryInst();
          LLVM_DEBUG(dbgs() << "  - Def: " << *DefInst << '\n');
          auto *ClobberEB = AT.getEDTBlock(DefInst);
          bool isInitBlock = ClobberEB->isInit();
          /// If the clobbering access is inside the same EDT, we ignore it
          if(ClobberEB->isInSameEDT(*CurrentEB)) {
            LLVM_DEBUG(dbgs() << "    - Clobbering access is in the same EDT\n");
            /// Continue looking for clobbering access
            // WorkList.push_back(
            //   Walker->getClobberingMemoryAccess(Def->getDefiningAccess(), Loc));
            continue;
          }

          LLVM_DEBUG(dbgs() << "    - Clobbering access is not in the same EDT\n");
          /// Iterate through the operands of the instruction
          for (Use &U : DefInst->operands()) {
            Value *Val = U.get();
            /// If it not an instruction, we ignore it
            auto *ValInst = dyn_cast<Instruction>(Val);
            if(!ValInst) {
              /// It can be a constant value, or an argument...
              continue;
            }
            LLVM_DEBUG(dbgs() << "    - Use: " << *ValInst << '\n');
            EDTBlock *ValEB = AT.getEDTBlock(ValInst);
            /// Ignore uses in the same block
            if(ValEB->isInSameEDT(*CurrentEB))
              continue;
            /// If ClobberEDTBlock is an init block, we need to determine
            /// if the value is a shared variable. If not, it is not a clobbering
            /// access. For now return true.
            if(isInitBlock) {
              /// Is it a shared variable?
              /// Signal value
              // return true;
              LLVM_DEBUG(dbgs() << "      - Clobbering access is an init block\n");
            }
            /// Get memory access
            auto *ValMA = MSSA.getMemoryAccess(ValInst);
            if(ValMA && isa<MemoryUse>(ValMA)) {
              /// If it is a memory use, we can ignore it. 
              /// Memory uses will be handled when we analyze the clobbering
              continue;
            }
            /// Get underlying object of the instruction
            auto *ValObj = getUnderlyingObject(Val);
            /// For now, let's print the underlying object
            LLVM_DEBUG(dbgs() << "      - Underlying object: " << *ValObj << "\n");
            // ValMA = MSSA.getMemoryAccess(ValObj);
            // if(MemoryDef *Def = dyn_cast<MemoryDef>(ValMA)) {
            //   /// If it is a memory def, we can ignore it
            //   continue;
            // }



            /// If it not an instruction, we ignore it
            // auto *ValInst = dyn_cast<Instruction>(v);
            // if(!ValInst) {
            //   /// It can be a constant value, or an argument...
            //   continue;
            // }
            // /// Check if the instruction is in the same EDT Data scope
            // auto *EB = AT.getEDTBlock(ValInst->getParent());
            // /// Is it a memory access?
            // auto *ValMA = MSSA.getMemoryAccess(ValInst);
            // if(!ValMA) {
            //   /// It is not a memory access, so we ignore it
            //   continue;
            // }
            // ...
          }
          /// If ClobberEB is an init block, we need to determine if the value
          /// is a shared variable. If it is, ClobberEB needs to signal the value
          /// to the CurrentEB. 
          

          for (auto &U : Def->uses()) {
            MemoryAccess *MA = cast<MemoryAccess>(U.getUser());
            LLVM_DEBUG(dbgs() << "  - Use: " << *MA << '\n');

            /// Get all uses of the memory access
            // for (auto &Use : MA->uses()) {

            /// If the use is in the same EDT block, we ignore it



            // if (auto *MU = cast_of_null<MemoryUse>MA) {
            //   // Process MemoryUse as needed.
            // }
            // else {
            //   // Process MemoryDef or MemoryPhi as needed.

            //   // As a user can come up twice, as an optimized access and defining
            //   // access, keep a visited list.

            //   // Check transitive uses as needed
            //   checkUses (MA); // use a worklist for an iterative algorithm
            // }
          }
          // Block numbering starts at 1.
          // unsigned long LastNumber = 0;
          // for(const MemoryAccess &MA : *(MSSA.getBlockAccesses(ClobberBB))) {
            
          // }

          return true;
        }

        const MemoryPhi *Phi = cast<MemoryPhi>(MA);
        for (const auto &Use : Phi->incoming_values())
          WorkList.push_back(cast<MemoryAccess>(&Use));
        // if(!MA) {
        //   LLVM_DEBUG(dbgs() << "  - No memory access\n");
        //   return false;
        //   // auto *ValInst = dyn_cast<Instruction>(Val);
        //   // if(!ValInst) {
        //   //   /// It can be a constant value, or an argument...
        //   //   return false;
        //   // }
        // }
        // auto *CA = Walker->getClobberingMemoryAccess(MA);
        // if(!CA) {
        //   LLVM_DEBUG(dbgs() << "  - No clobbering access\n");
        //   return false;
        // }
        // LLVM_DEBUG(dbgs() << "  - Clobbering access: " << *CA << "\n");
        // /// If it is a live on entry def, we dont care about it.
        // if (MSSA.isLiveOnEntryDef(CA)) {
        //   LLVM_DEBUG(dbgs() << "  - It is a live on entry\n");
        //   return false;
        // }
        // /// Get the EDT block of the memory instruction of the clobbering access
        // MemoryUseOrDef *CMA = dyn_cast<MemoryUseOrDef>(CA);
        // Instruction *CInst = CMA->getMemoryInst();
        // auto *ClobberEB = AT.getEDTBlock(CInst->getParent());
        // /// If the clobbering access is inside the same EDT, we ignore it
        // if(ClobberEB->isInSameEDT(*CurrentEB)) {
        //   LLVM_DEBUG(dbgs() << "  - Clobbering access is in the same EDT\n");
        //   return false;
        // }
        // else {
        //   LLVM_DEBUG(dbgs() << "  - Clobbering access is in a different EDT\n");
        /// Now we have to identify the external values. We do this by checking
        /// the operands of the instruction. If the operand is a constant, we 
        /// ignore it. If it is
        /// a global variable, we add it to the external values of the EDT.
    
        /// If ClobberEB is an init block, we need to determine if the value
        /// is a shared variable. If it is, ClobberEB needs to signal the value
        /// to the CurrentEB. 


        /// Iterate through the arguments of the CallBase
        /// the values that are used in the init block. For this, we need to
        // if(auto *CB = dyn_cast<CallBase>(I)) {
        //   for(auto &arg : CB->args()) {
        //     handleInstruction(CurrentEB, *CB);
        //     // checkAndAddValue(arg, EDT);
        //   }
        //   return true;
        // }
        // /// Iterate through the operands of the instruction
        // for(unsigned int i = 0; i < I->getNumOperands(); i++)
        //   checkAndAddValue(I->getOperand(i), EDT);
        // // LLVM_DEBUG(dbgs() << "\n");
      }
      return true;
    };

    LLVM_DEBUG(dbgs() << "[AADataEnvFunction] Checking read/write instructions\n");
    /// Callback to check a read/write instruction.
    /// It first verifies if the instruction belongs to an EDT Block, if it
    /// does it adds it to the RWInsts vector of the EDT. 
    /// Then it finds the memory access and analyzes its corresponding 
    /// clobbering access. It the clobbering access is not in the same EDT, it
    /// Analyzes the operands of the instruction to identify the external values,
    /// and fills out the DependentValues vector of the EDT.
    auto CheckRWInst = [&](Instruction &I) {
      LLVM_DEBUG(dbgs() << "Checking "<< I <<"\n");
      auto *CurrentBB = I.getParent();
      /// Ignore lifetime start/end instructions
      if(I.isLifetimeStartOrEnd())
        return true;
      /// Get the EDT block of the instruction
      EDTBlock *CurrentEB = AT.getEDTBlock(CurrentBB);
      if(!CurrentEB) {
        LLVM_DEBUG(dbgs() << "[AADataEnvFunction] BB parent: " 
                          << CurrentBB->getName() << "\n");
        LLVM_DEBUG(dbgs() << "[AADataEnvFunction] BB is not an EDT region\n");
        indicatePessimisticFixpoint();
        return false;
      }
      /// Handle EDT init block
      if(CurrentEB->Type == EDTBlock::INIT) 
        return handleEDTInitBlock(A, I, *CurrentEB);
      /// Add the instruction to the RWInsts vector
      CurrentEB->EDT->RWInsts.insert(&I);
      /// Analyze memory dependencies
      MemDepResult DepRes = MDA->getDependency(&I);
      Instruction *DepInst = DepRes.getInst();
      if(!DepInst) {
        if(DepRes.isUnknown()) {
          LLVM_DEBUG(dbgs() << "  - Dependency is unknown\n");
        }
        if(DepRes.isNonLocal()) {
          LLVM_DEBUG(dbgs() << "  - Non local dependency\n");
          /// If it is a load instruction, get operand
          if(auto *LI = dyn_cast<LoadInst>(&I)) {
            Value *UnderlyingObj = getUnderlyingObject(LI->getPointerOperand());
            LLVM_DEBUG(dbgs() << "  - Underlying object: " << *UnderlyingObj << "\n");
          }
          else {
            Value *UnderlyingObj = getUnderlyingObject(&I);
            LLVM_DEBUG(dbgs() << "  - Underlying object: " << *UnderlyingObj << "\n");
          }
        }
        

        // LLVM_DEBUG(dbgs() << "  - No memory instruction\n");
        return true;
      }
      LLVM_DEBUG(dbgs() << "  - Memory dependency: " << *DepRes.getInst() << "\n");
      if(!DepRes.isDef()) {
        LLVM_DEBUG(dbgs() << "    - Dep is not a memory definition\n");
        return true;
      }
      // if (DepRes.isUnknown() || !DepRes.getInst()) {
      //   LLVM_DEBUG(dbgs() << "  - No memory access\n");
      //   return true;
      // }
      
      // Handle instruction
      // handleInstruction(CurrentEB, I);

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
    OutlinedFunction = dyn_cast<Function>(CB.getArgOperand(2));
    /// For now, indicate optimistic fixpoint
    indicateOptimisticFixpoint();
    LLVM_DEBUG(dbgs() << "[AADataEnvCallSite] Parallel region - Data environment filled out\n");
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
      // LLVM_DEBUG(dbgs()
      //             << "[" << Range.Offset << "-" << Range.Offset + Range.Size
      //             << "] : " << AccIndex.size() << "\n");
      for (auto AI : AccIndex) {
        auto &Acc = AAPI->getAccess(AI);
        auto AccKind = Acc.getKind();
        auto *Inst = Acc.getLocalInst();
        // LLVM_DEBUG(dbgs() << "     - " << AccKind << " - " << *Inst << "\n");

        if (Acc.isWrittenValueYetUndetermined()) {
          // LLVM_DEBUG(dbgs()
          //             << "       - c: Value written is not known yet \n");
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
            // LLVM_DEBUG(dbgs() << "       - c: shared variable " << *value << "\n");
            AuxDE.SharedVars.push_back(value);
          }
          /// Check if it is a firstprivate variable
          else if((uint64_t)Range.Offset >= (uint64_t)kmp_task_t_size) {
            // LLVM_DEBUG(dbgs() << "       - c: firstprivate variable " << *value << "\n");
            AuxDE.FirstprivateVars.push_back(value);
          }
          // else
          //   LLVM_DEBUG(dbgs() << "       - c: other: " << *value << "\n");
        }
        /// Read instruction
        else if(AccKind == AAPointerInfo::AccessKind::AK_MUST_READ) {
          // LLVM_DEBUG(dbgs() << "       -- Read instruction\n");
          /// Check if it a load instruction
          if(auto *LI = dyn_cast<LoadInst>(Inst)) {
            // LLVM_DEBUG(dbgs() << "       - Load instruction. Running AA on this.\n");
            auto *PIL = A.getAAFor<AAPointerInfo>(
              *this, IRPosition::value(*LI), DepClassTy::OPTIONAL);
            if (!PIL->getState().isValidState()) {
              indicatePessimisticFixpoint();
              // LLVM_DEBUG(dbgs() <<"[AADataEnvCallSite] PIL is invalid\n");
              return false;
            }
            /// Is it a fixpoint?
            if (!PIL->getState().isAtFixpoint()) {
              // LLVM_DEBUG(dbgs() <<"[AADataEnvCallSite] PIL is not at fixpoint\n");
              return false;
            }
            /// For all offset bins in the PI, run AccessCB
            if(!PIL->forallOffsetBins(AccessCB)) {
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
      auto &OMP = EB->OMP;
      OMP.DE.append(DE);
      OMP.F = OutlinedFunction;
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
    /// IMPORTANT - TODO
    /// Keep in mind that is important to consider loops. The following pass
    /// might be relevant:
    /// - Dominator Tree
    /// - LoopInfo
    /// - Region analysis (https://youtu.be/TjpcaxlgHxk?si=WHINP-Krjk5GB_s_)
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
    /// Insert a new EDT we haven't created one yet for the function
    EDTInfo *CurrentEDT = AT.getEDTForFunction(&F);
    if(!CurrentEDT) {
      CurrentEDT = AT.insertEDT(&F);
      /// If it is the main function, insert it to the EDTforFunction map
      if(F.getName() == "main")
        AT.insertEDTForFunction(&F, CurrentEDT);
    }

    /// If function is not IPO amendable, we give up
    if(!A.isFunctionIPOAmendable(F))
      return Changed;
    
    /// Fill out set of read/write instructions
    auto CheckRWInst = [&](Instruction &I) {
      /// Ignore lifetime start/end instructions
      if(!I.isLifetimeStartOrEnd())
        AT.insertRWInst(F, &I);
      return true;
    };
    bool UsedAssumedInformationInCheckRWInst = false;
    if (!A.checkForAllReadWriteInstructions(
            CheckRWInst, *this, UsedAssumedInformationInCheckRWInst)) {
      LLVM_DEBUG(dbgs() << "  - checkForAllReadWriteInstructions failed\n");
      return Changed;
    }
    /// Get entry block
    BasicBlock *CurrentBB = &(F.getEntryBlock());
    AT.insertEDTBlock(CurrentEDT, EDTBlock::ENTRY, CurrentBB);
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
            /// Insert EDT block for the parallel init region to current EDT. Then 
            /// create empty EDT for the outlined function and set init block.
            EDTBlock *ParallelInitBlock = 
              AT.insertEDTBlock(CurrentEDT, EDTBlock::INIT, ParallelBB, OI);
            EDTInfo *ParallelEDT = AT.insertEDT(&F, ParallelInitBlock);
            // AT.insertEDTWithDep(CurrentEDT, EB);
            /// Split block at the next instruction
            CurrentI = CurrentI->getNextNonDebugInstruction();
            BasicBlock *ParallelDone = 
              SplitBlock(ParallelBB, CurrentI, DT, LI, nullptr,
                        "par.done." + std::to_string(ParallelRegion));
            NextBB = ParallelDone;
            ParallelRegion++;
            /// Create new EDT and add dependency from current EDT to the new EDT.
            /// Then, insert EDT block for the parallel done region to the new EDT.
            /// Finally add dependency from par.region to par.done
            CurrentEDT = AT.insertEDT(&F);
            // AT.insertEDTWithDep(CurrentEDT, &F);
            AT.insertEDTBlock(CurrentEDT, EDTBlock::ENTRY, ParallelDone);
            AT.insertDep(ParallelEDT, CurrentEDT);
          }
          break;
          case OMPInfo::TASKALLOC: {
            OMPInfo OI(OMPInfo::TASK, CB);
            /// Split block at __kmpc_omp_task_alloc
            BasicBlock *TaskBB = 
              SplitBlock(CurrentI->getParent(), CurrentI, DT, LI, nullptr,
                        "task.region." + std::to_string(TaskRegion));
            /// Insert EDT block for the task init region to current EDT. Then 
            /// create empty EDT for the outlined function, and add dependency from 
            /// current EDT to the new EDT.
            EDTBlock *TaskInitBlock = 
              AT.insertEDTBlock(CurrentEDT, EDTBlock::INIT, TaskBB, OI);
            EDTInfo *TaskEDT = AT.insertEDT(&F, TaskInitBlock);
            // AT.insertEDTWithDep(CurrentEDT, TaskInitBlock);
            /// Find the task call
            while ((CurrentI = CurrentI->getNextNonDebugInstruction())) {
              auto *TCB = dyn_cast<CallBase>(CurrentI);
              if(TCB && 
                OMPInfo::getRTFunction(TCB->getCalledFunction()) == OMPInfo::TASK)
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
            /// Create new EDT and add dependency from current EDT to the new EDT.
            /// Then, insert EDT block for the task done region to the new EDT.
            /// Finally add dependency from task.region to task.done
            CurrentEDT = AT.insertEDT(&F);
            // AT.insertEDTWithDep(CurrentEDT, &F);
            AT.insertEDTBlock(CurrentEDT, EDTBlock::ENTRY, TaskDone, OI);
            AT.insertDep(TaskEDT, CurrentEDT);
          }
          break;
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
      } while ((CurrentI = CurrentI->getNextNonDebugInstruction()));

    } while ((CurrentBB = NextBB));

    LLVM_DEBUG(dbgs() << TAG << "Identifying EDT regions done with: "
                      << AT.EDTsFromFunction[&F].size() << "\n");

    Changed = true;
    return Changed;
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

  // void fillExternalValues(ARTSTransformer &AT, MemorySSA &MSSA, Function *F) {
  //   /// This function fills the ExternalValues set of each EDT. It iterates
  //   /// through the EDTs and for each EDT it iterates through the RWInsts set
  //   /// to identify the values that are not in the same EDT.
  //   LLVM_DEBUG(dbgs() << TAG << "Filling ExternalValues\n");
  //   auto checkAndAddValue = [&](Value *Val, EDTInfo *EDT) {
  //     // Type *ValType = Val->getType();
  //     LLVM_DEBUG(dbgs() << "  - Operand: " << *Val << "\n");
  //     // LLVM_DEBUG(dbgs() << "    - Operand type: " << *ValType << "\n");
  //     if(isa<Argument>(Val)) {
  //       // LLVM_DEBUG(dbgs() << "    - It is an Argument\n");
  //       return;
  //     }
  //     auto *ValInst = dyn_cast<Instruction>(Val);
  //     if(!ValInst) {
  //       /// It can be a constant value...
  //       // LLVM_DEBUG(dbgs() << "    - It is not an Instruction\n");
  //       return;
  //     }
  //     // /// It is also important to consider global variables
  //     // void getGlobalsUsedByFunction(const Function &F, set<GlobalValue*> *Globals) {
  //     //   for (const BasicBlock &BB : F)
  //     //     for (const Instruction &I : BB)
  //     //       for (const Value *Op : I.operands())
  //     //         if (const GlobalValue* G = dyn_cast<GlobalValue>(*Op))
  //     //           Globals->insert(G);
  //     // }
  //     /// Check if the instruction is in the same EDT Data scope
  //     // LLVM_DEBUG(dbgs() << "    - It is an Instruction\n");
  //     auto *EB = AT.getEDTBlock(ValInst->getParent());
  //     EB->Analyzed = true;
  //     if(EB->EDT->ID == EDT->ID) {
  //       LLVM_DEBUG(dbgs() << "    - It is in the same EDT\n");
  //       return;
  //     }
  //     LLVM_DEBUG(dbgs() << "    - It is not in the same EDT\n");
  //     EDT->ExternalValues.insert(ValInst);
  //   };

  //   LLVM_DEBUG(dbgs() << "\n" << TAG << "EDTs created from this function:" << "\n");
  //   for(auto *EDT : AT.EDTsFromFunction[F]) {
  //     LLVM_DEBUG(dbgs() << "EDT #" << EDT->ID << "\n");
  //     /// If the EDT has external values, we already analyzed it.
  //     /// Skip it for now.
  //     if(EDT->ExternalValues.size() != 0)
  //       continue;
  //     /// Print RW instructions
  //     LLVM_DEBUG(dbgs() << "RW instructions: "<< EDT->RWInsts.size() << "\n");
  //     for(auto *I : EDT->RWInsts) {
  //       LLVM_DEBUG(dbgs() << "  - " << *I << "\n");
  //       MemoryAccess *MA = MSSA.getMemoryAccess(I);
  //       if(!MA) {
  //         LLVM_DEBUG(dbgs() << "    - MemoryAccess not found\n");
  //         return;
  //       }
  //       if(auto *Def = dyn_cast_or_null<MemoryDef>(MA)) {
  //         LLVM_DEBUG(dbgs() << "    MemoryDef: " << *Def << "\n");
  //         Instruction *DefInst = Def->getMemoryInst();
  //         LLVM_DEBUG(dbgs() << "        Def: " << *DefInst << '\n');
  //       }
  //       else if (auto *Use = dyn_cast_or_null<MemoryUse>(MA)) {
  //         LLVM_DEBUG(dbgs() << "    MemoryUse: " << *Use << "\n");
  //         Instruction *UseInst = Use->getMemoryInst();
  //         LLVM_DEBUG(dbgs() << "      Use: " << *UseInst << '\n');
  //       }
  //       else if (auto *Phi = dyn_cast_or_null<MemoryPhi>(MA)) {
  //       LLVM_DEBUG(dbgs() << "      MemoryPhi: " << *Phi << "\n");
  //         // Instruction *PhiInst = Phi->getMemoryInst();
  //         // LLVM_DEBUG(dbgs() << "  Phi: " << *PhiInst << '\n');
  //       }
  //       else if (auto *DefPhi = dyn_cast_or_null<MemoryDef>(MA)) {
  //         LLVM_DEBUG(dbgs() << "    MemoryDefPhi: " << *DefPhi << "\n");
  //         Instruction *DefPhiInst = DefPhi->getMemoryInst();
  //         LLVM_DEBUG(dbgs() << "      DefPhi: " << *DefPhiInst << '\n');
  //       }
  //       // else {
  //       //   LLVM_DEBUG(dbgs() << "MemoryAccess: " << *MA << "\n");
  //       //   Instruction *MAInst = MA->getMemoryInst();
  //       //   LLVM_DEBUG(dbgs() << "  MA: " << *MAInst << '\n');
  //       // }
  //       /// Iterate through the arguments of the CallBase
  //       if(auto *CB = dyn_cast<CallBase>(I)) {
  //         //     if(CB->isLifetimeStartOrEnd())
  //         //       continue;
  //         for(auto &arg : CB->args()) {
  //           checkAndAddValue(arg, EDT);
  //         }
  //         continue;
  //       }
  //       /// Iterate through the operands of the instruction
  //       for(unsigned int i = 0; i < I->getNumOperands(); i++)
  //         checkAndAddValue(I->getOperand(i), EDT);
  //       // LLVM_DEBUG(dbgs() << "\n");
  //     }
  //   }
  // }

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
      return ChangeStatus::CHANGED;
    }
    if(!DEAA->getState().isAtFixpoint()) {
      LLVM_DEBUG(dbgs() <<"[AAToARTSFunction] DEAA is not at fixpoint\n");
      return ChangeStatus::UNCHANGED;
    }
    LLVM_DEBUG(dbgs() <<"[AAToARTSFunction] DEAA is at fixpoint\n");

    
    // fillExternalValues(AT, F);
    /// Iterate through the EDTs to analyze the EDT with empty number of Blocks
    LLVM_DEBUG(dbgs() << "\n" << TAG << "EDTs with empty number of Blocks:\n");
    for(auto *EDT : AT.EDTsFromFunction[F]) {
      if(EDT->Blocks.size() != 0)
        continue;
      LLVM_DEBUG(dbgs() << "EDT #" << EDT->ID << "\n");
      auto *EDTInit = EDT->Init;
      /// The EDT must have an EDTInit and it must have an OMPInfo
      if(!EDTInit && EDTInit->HasOMP) {
        LLVM_DEBUG(dbgs() << "  - EDTInit with OMPInfo not found\n");
        indicatePessimisticFixpoint();
        return ChangeStatus::CHANGED;
      }
      /// Print where the EDTInit belongs to
      auto &OMP = EDTInit->OMP;
      LLVM_DEBUG(dbgs() << "  - EDTInit: " << EDTInit->BB->getName() << "\n");
      LLVM_DEBUG(dbgs() << "  - OMP: " << OMP.F->getName() << "\n");
      /// Add it to the EDTForFunction map
      AT.insertEDTForFunction(OMP.F, EDT);
      /// Run AAToARTS on the EDTInit function
      auto *OMPFAA = A.getAAFor<AAToARTS>(
        *this, IRPosition::function(*OMP.F), DepClassTy::OPTIONAL);
      if (!OMPFAA->getState().isValidState()) {
        LLVM_DEBUG(dbgs() <<"[AAToARTSFunction] OMPFAA is invalid\n");
        indicatePessimisticFixpoint();
        return ChangeStatus::CHANGED;
      }
      if(!OMPFAA->getState().isAtFixpoint()) {
        LLVM_DEBUG(dbgs() <<"[AAToARTSFunction] OMPFAA is not at fixpoint\n");
        return ChangeStatus::UNCHANGED;
      }
    }

    
    
    
    /// If this is the main function
    if(F->getName() == "main") {
      LLVM_DEBUG(dbgs() << TAG << "----------- PROCESS HAS FINISHED -----------\n");
      EDTInfo *MainEDT = AT.getEDTForFunction(F);
      LLVM_DEBUG(dbgs() << TAG << "----------- EDTs INFORMATION -----------\n");
      LLVM_DEBUG(dbgs() << *MainEDT << "\n");

    /// If it is not an EDT region, get data environment based on the 
    /// MemorySSA analysis. We will do this process until 
    /// the function is done or we reach another EDT region.

    // DominatorTree *DT = AG.getAnalysis<DominatorTreeAnalysis>(*F);
    // ScalarEvolution *SE =
    //   A.getInfoCache().getAnalysisResultForFunction<ScalarEvolutionAnalysis>(F);
    // LoopInfo *LI = AG.getAnalysis<LoopAnalysis>(*F);

    // LV = &getAnalysis<LiveVariables>();
    // LoopInfo &LI = AG.getAnalysis<LoopInfoWrapperPass>(*F)->getLoopInfo();

    }
    indicateOptimisticFixpoint();
    return ChangeStatus::CHANGED;
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


/// ---------------------------- ARTS TRANSFORM ---------------------------- ///
bool ARTSTransformer::run(Attributor &A) {
  bool changed = false;
  /// The process start in the main function, we which will converted to an EDT.
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

  LLVM_DEBUG(dbgs() << TAG << "\nEDTs INFORMATION\n");
  LLVM_DEBUG(dbgs() << TAG << EDTs << "\n");
  return changed;
}

bool ARTSTransformer::runAttributor(Attributor &A) {
  LLVM_DEBUG(dbgs() << TAG <<  "[Attributor] Process started\n");
  ChangeStatus Changed = A.run();
  LLVM_DEBUG(dbgs() << TAG <<  "[Attributor] Done, result: " << Changed << ".\n");

  return Changed == ChangeStatus::CHANGED;
}

/// ------------------------ ARTS TRANSFORMATION PASS ------------------------ ///
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
  AC.IsModulePass = true;
  AC.DeleteFns = false;
  AC.RewriteSignatures = false;
  Attributor A(Functions, InfoCache, AC);

  /// Run ARTSTransform
  Changed |= InfoCache.ARTSTransform.run(A);

  LLVM_DEBUG(dbgs() << TAG << "Module after ARTSTransformer Module Pass:\n" << M);
  if (Changed)
    return PreservedAnalyses::none();
  return PreservedAnalyses::all();
}

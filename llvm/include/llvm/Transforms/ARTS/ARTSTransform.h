#ifndef LLVM_TRANSFORMS_ARTS_H
#define LLVM_TRANSFORMS_ARTS_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Transforms/IPO/Attributor.h"

namespace llvm {

/// DATA ENVIRONMENT
/// Struct to store information about the data environment of the OpenMP
/// regions
struct DataEnv {
  /// Interface 
  DataEnv(){};
  DataEnv(DataEnv &DE) { append(DE); };
  DataEnv &operator=(const DataEnv &DE) {
    PrivateVars = DE.PrivateVars;
    SharedVars = DE.SharedVars;
    FirstprivateVars = DE.FirstprivateVars;
    LastprivateVars = DE.LastprivateVars;
    return *this;
  }

  void append(const DataEnv &DE) {
    PrivateVars.append(DE.PrivateVars.begin(), DE.PrivateVars.end());
    SharedVars.append(DE.SharedVars.begin(), DE.SharedVars.end());
    FirstprivateVars.append(DE.FirstprivateVars.begin(),
                            DE.FirstprivateVars.end());
    LastprivateVars.append(DE.LastprivateVars.begin(),
                           DE.LastprivateVars.end());
  };

  /// Attributes 
  SmallVector<Value *, 4> PrivateVars;
  SmallVector<Value *, 4> SharedVars;
  SmallVector<Value *, 4> FirstprivateVars;
  SmallVector<Value *, 4> LastprivateVars;
};

inline raw_ostream &operator<<(raw_ostream &OS, DataEnv &DE) {
  OS << "Data environment: \n";
  OS << "Firstprivate: " << DE.FirstprivateVars.size() << "\n";
  for (auto *V : DE.FirstprivateVars)
    OS << "  - " << *V << "\n";
  OS << "Private: " << DE.PrivateVars.size() << "\n";
  for (auto *V : DE.PrivateVars)
    OS << "  - " << *V << "\n";
  OS << "Shared: " << DE.SharedVars.size() << "\n";
  for (auto *V : DE.SharedVars)
    OS << "  - " << *V << "\n";
  OS << "Lastprivate: " << DE.LastprivateVars.size() << "\n";
  for (auto *V : DE.LastprivateVars)
    OS << "  - " << *V << "\n";
  OS << "\n";
  return OS;
}

/// OMP INFO 
/// Struct to store information about OpenMP regions
struct OMPInfo {
  enum RTFType {
    OTHER = 0,
    PARALLEL,
    PARALLEL_FOR,
    TASKALLOC,
    TASK,
    TASKWAIT,
    TASKDEP,
    SET_NUM_THREADS
  };

  /// Interface 
  OMPInfo() : RTF(OTHER), CB(nullptr), F(nullptr){};
  OMPInfo(OMPInfo &OI) : RTF(OI.RTF), DE(OI.DE), CB(OI.CB), F(OI.F){};
  OMPInfo(RTFType RTF, CallBase *CB) : RTF(RTF), CB(CB), F(nullptr){};
  OMPInfo &operator=(const OMPInfo &OI) {
    RTF = OI.RTF;
    DE = OI.DE;
    CB = OI.CB;
    F = OI.F;
    return *this;
  }
  /// Attributes 
  RTFType RTF;
  DataEnv DE;
  CallBase *CB; // CallBase of the OpenMP region
  Function *F;  // Pointer to the outlined function

  /// Helper Functions
  static RTFType getRTFunction(Function *F) {
    if (!F)
      return RTFType::OTHER;
    auto CalleeName = F->getName();
    if (CalleeName == "__kmpc_fork_call")
      return RTFType::PARALLEL;
    if (CalleeName == "__kmpc_omp_task_alloc")
      return RTFType::TASKALLOC;
    if (CalleeName == "__kmpc_omp_task")
      return RTFType::TASK;
    if (CalleeName == "__kmpc_omp_task_alloc_with_deps")
      return RTFType::TASKDEP;
    if (CalleeName == "__kmpc_omp_taskwait")
      return RTFType::TASKWAIT;
    if (CalleeName == "omp_set_num_threads")
      return RTFType::SET_NUM_THREADS;
    if (CalleeName == "__kmpc_for_static_init_4")
      return RTFType::PARALLEL_FOR;
    return RTFType::OTHER;
  }

  static RTFType getRTFunction(CallBase &CB) {
    auto *Callee = CB.getCalledFunction();
    return getRTFunction(Callee);
  }

  static RTFType getRTFunction(Instruction *I) {
    auto *CB = dyn_cast<CallBase>(I);
    if (!CB)
      return RTFType::OTHER;
    return getRTFunction(*CB);
  }

  static bool isTaskFunction(Function *F) {
    auto RT = getRTFunction(F);
    if (RT == RTFType::TASK || RT == RTFType::TASKDEP ||
        RT == RTFType::TASKWAIT)
      return true;
    return false;
  }

  const StringRef getCBName() const {
    if (CB)
      return CB->getCalledFunction()->getName();
    return "";
  }

  const StringRef getFName() const {
    if (F)
      return F->getName();
    return "";
  }

  const Function *getCBFunction() const {
    if (CB)
      return CB->getCalledFunction();
    return nullptr;
  }
};

/// EDT DEP 
struct EDTInfo;
struct EDTDep {
  /// Interface 
  EDTDep() : EDT(nullptr){};
  EDTDep(EDTInfo *EDT) : EDT(EDT){};
  // EDTDep(EDTInfo *EDT, DataEnv *DE) : EDT(EDT), DE(DE) {};

  /// Attributes 
  // EDT where the value will be signaled to
  EDTInfo *EDT;
  /// Values to be signaled
  SmallVector<Value *, 4> Values;
};

/// EDT INFO 
struct EDTBlock {
  enum BlockType { ENTRY = 0, INIT_ALLOCA, INIT, EVENT, OTHER };
  /// Interface 
  EDTBlock(EDTInfo *EDT) : EDT(EDT), Type(OTHER), BB(nullptr){};
  EDTBlock(EDTInfo *EDT, BlockType Type, BasicBlock *BB)
      : EDT(EDT), Type(Type), BB(BB){};
  EDTBlock(EDTBlock &B) : EDT(B.EDT), Type(B.Type), BB(B.BB){};

  void setOMPInfo(OMPInfo &OMPI) {
    HasOMP = true;
    OMP = OMPI;
  }

  bool isInSameEDT(EDTBlock &B) { return (EDT == B.EDT); }

  bool isInit() { return (Type == INIT); }
  bool isEntry() { return (Type == ENTRY); }
  bool isOther() { return (Type == OTHER); }

  /// Attributes 
  /// Pointer to EDT where the EDTBlock belongs
  EDTInfo *EDT;
  /// Type of the EDTBlock
  BlockType Type;
  /// BasicBlock of the EDTBlock
  BasicBlock *BB;
  /// Flag to indicate if the EDT has OpenMP RT calls
  bool HasOMP = false;
  /// OpenMP information of the EDT
  OMPInfo OMP;
  /// Flag to indicate if the EDTBlock was Transformed
  bool Transformed = false;
  /// Flag to indicate if the EDTBlock was Analyzed
  bool Analyzed = false;
};

inline raw_ostream &operator<<(raw_ostream &OS, EDTBlock &EB) {
  StringRef Transformed = EB.Transformed ? "True: " : "False";
  OS << "-- EDTBlock --\n";
  OS << "  - BB Name: " << EB.BB->getName() << "\n";
  OS << "  - Block Type: ";
  if (EB.Type == EDTBlock::INIT_ALLOCA)
    OS << "Init Alloca";
  else if (EB.Type == EDTBlock::INIT)
    OS << "Init";
  else if (EB.Type == EDTBlock::EVENT)
    OS << "Event";
  else if (EB.Type == EDTBlock::ENTRY)
    OS << "Entry";
  else
    OS << "Other";
  OS << "\n";
  const char *HasOMP = EB.HasOMP ? "True" : "False";
  OS << "  - OMP: " << HasOMP << "\n";
  OS << "  - Transformed: " << Transformed;
  OS << *EB.BB;
  return OS;
}

/// EDT INFO 
/// Struct to store information about EDTs
struct EDTInfo {
  /// Interface 
  EDTInfo(uint64_t ID) : ID(ID){};
  EDTInfo(uint64_t ID, Function *F, DataEnv DE) : ID(ID), F(F), DE(DE){};

  EDTBlock *insertEDTBlock(EDTBlock Block) {
    EDTBlock *EDTB = new EDTBlock(Block);
    Blocks.insert(EDTB);
    return EDTB;
  }

  EDTBlock *insertEDTBlock(EDTBlock::BlockType Ty, BasicBlock *BB) {
    EDTBlock *EDTB = new EDTBlock(this, Ty, BB);
    Blocks.insert(EDTB);
    return EDTB;
  }

  void addDep(EDTDep *SI) { Deps.insert(SI); }

  void setF(Function *F) { this->F = F; }

  /// Attributes 
  /// GUID of the EDT
  uint64_t ID;
  /// Pointer to the EDT function
  Function *F = nullptr;
  /// Data environment of the EDT
  DataEnv DE;
  /// Init EDTBlock
  EDTBlock *Init = nullptr;
  /// Indicates if the EDT was analyzed or not
  bool Analyzed = false;
  /// List of EDT Dependencies - successors
  SmallPtrSet<EDTDep *, 4> Deps;
  /// List of BasicBlocks that are part of the EDT
  SmallPtrSet<EDTBlock *, 4> Blocks;
  /// Set of instructions that may read or write to memory
  SmallPtrSet<Instruction *, 4> RWInsts;
  /// Set of used instructions with values not declared in the EDT
  SmallPtrSet<Instruction *, 4> ExtInsts;
};

inline raw_ostream &operator<<(raw_ostream &OS, EDTInfo &EI) {
  OS << "----- EDT -----\n";
  OS << "ID: " << EI.ID << "\n";
  OS << "Number of Deps: " << EI.Deps.size() << "\n";
  for (auto *Dep : EI.Deps) {
    OS << "  - " << Dep->EDT->ID << "\n";
  }
  OS << "Number of RWInsts: " << EI.RWInsts.size() << "\n";
  for (auto *I : EI.RWInsts) {
    OS << "  - " << *I << "\n";
  }
  OS << "Number of ExtInsts: " << EI.ExtInsts.size() << "\n";
  for (auto *I : EI.ExtInsts) {
    OS << "  - " << *I << "\n";
  }
  OS << "Number of Blocks: " << EI.Blocks.size() << "\n";
  for (auto *B : EI.Blocks) {
    OS << *B << "\n";
  }
  return OS;
}

/// ARTS TRANSFORM 
struct ARTSTransformer {
  /// Interface 
  ARTSTransformer(Module &M) : M(M) {}
  ~ARTSTransformer() {
    /// Delete EDTs
    // for (const auto &It : EDTBlocks)
    //   delete (It.second);
    // EDTBlocks.clear();
  }
  bool run(Attributor &A);
  bool runAttributor(Attributor &A);

  /// Helper Functions 
  /// /// Insert EDT
  EDTInfo *insertEDT(Function *F, EDTBlock *Init = nullptr) {
    EDTInfo *EDTI = new EDTInfo(EDTID++);
    if (Init) {
      assert(Init->Type == EDTBlock::BlockType::INIT &&
             "Init block must be of type INIT");
      EDTI->Init = Init;
    }
    EDTs.insert(EDTI);
    EDTsFromFunction[F].insert(EDTI);
    return EDTI;
  }

  void insertDep(EDTInfo *From, EDTInfo *To) {
    /// Create dependency
    EDTDep *SI = new EDTDep(To);
    From->addDep(SI);
  }

  /// EDTBlock Map 
  /// Return the EDTBlock for a given BB or `nullptr` if there are
  /// none.
  EDTBlock *getEDTBlock(BasicBlock *BB) {
    auto It = EDTBlocks.find(BB);
    if (It != EDTBlocks.end())
      return (It->second);
    return nullptr;
  }

  EDTBlock *getEDTBlock(Instruction *I) { return getEDTBlock(I->getParent()); }

  /// Create EDTBlock and insert it into the EDT and map
  EDTBlock *insertEDTBlock(EDTInfo *EI, EDTBlock::BlockType Ty,
                           BasicBlock *BB) {
    /// Insert EDT block
    EDTBlock *EDTB = EI->insertEDTBlock(Ty, BB);
    /// Add it to the map
    auto It = EDTBlocks.find(BB);
    if (It != EDTBlocks.end())
      delete (It->second);
    EDTBlocks.insert(std::make_pair(BB, EDTB));
    /// Return EDT block
    return EDTB;
  }

  /// Create EDTBlock and insert it into the EDT and map
  EDTBlock *insertEDTBlock(EDTInfo *EI, EDTBlock::BlockType Ty, BasicBlock *BB,
                           OMPInfo &OMP) {
    EDTBlock *EDTB = insertEDTBlock(EI, Ty, BB);
    EDTB->setOMPInfo(OMP);
    return EDTB;
  }

  /// Get EDT for function
  EDTInfo *getEDTForFunction(Function *F) {
    auto It = EDTForFunction.find(F);
    if (It != EDTForFunction.end())
      return It->second;
    return nullptr;
  }

  /// Insert EDT for function
  void insertEDTForFunction(Function *F, EDTInfo *EDTI) {
    EDTForFunction.insert(std::make_pair(F, EDTI));
  }

  /// RWInsts Map 
  void insertRWInst(Function *F, Instruction *I) {
    RWInsts[F].insert(I);
    EDTBlock *EDTB = getEDTBlock(I);
    EDTB->EDT->RWInsts.insert(I);
  }

  /// Attributes 
  /// The underlying module.
  Module &M;
  /// Set of valid functions in the module.
  // SetVector<Function *> &Functions;
  /// List of EDTs
  uint64_t EDTID = 0;
  SmallPtrSet<EDTInfo *, 4> EDTs;
  /// Maps the basic block to the EDT Block
  DenseMap<BasicBlock *, EDTBlock *> EDTBlocks;
  /// Maps a function to the set of EDTs that were created from it.
  /// For example, if inside the main function there is a parallel region
  /// 2 EDTs will be created from it: one for the parallel region and one for
  /// the outlined function.
  DenseMap<Function *, SmallPtrSet<EDTInfo *, 4>> EDTsFromFunction;
  /// Maps a function to the EDT that represents it.
  /// There are EDTs that represent the same function. This map is used to
  /// identify them.
  DenseMap<Function *, EDTInfo *> EDTForFunction;
  /// Maps a function to set the instructions that may read or write to memory
  DenseMap<Function *, SmallPtrSet<Instruction *, 4>> RWInsts;
};

inline raw_ostream &operator<<(raw_ostream &OS,
                               SmallPtrSet<EDTInfo *, 4> &EDTs) {
  OS << "DUMPING EDTs\n";
  OS << "Number of EDTs: " << EDTs.size() << "\n";
  for (auto *E : EDTs) {
    OS << "\n" << *E;
  }
  return OS;
}

/// ARTS TRANSFORM PASS
///  From OpenMP to ARTS transformation pass.
class ARTSTransformPass : public PassInfoMixin<ARTSTransformPass> {
public:
  ARTSTransformPass() = default;
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
};
} // namespace llvm

#endif // LLVM_TRANSFORMS_ARTS_H

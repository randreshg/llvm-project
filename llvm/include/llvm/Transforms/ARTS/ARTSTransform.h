#ifndef LLVM_TRANSFORMS_ARTS_H
#define LLVM_TRANSFORMS_ARTS_H

#include "llvm/Transforms/IPO/Attributor.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/IR/PassManager.h"

namespace llvm {


/// ---------------------------- DATA ENVIRONMENT ---------------------------- ///
/// Struct to store information about the data environment of the OpenMP regions
struct DataEnv {
  /// ---------------------------- Interface ---------------------------- ///
  DataEnv() {};
  DataEnv(DataEnv &DE) { append(DE);};
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
    FirstprivateVars.append(DE.FirstprivateVars.begin(), DE.FirstprivateVars.end());
    LastprivateVars.append(DE.LastprivateVars.begin(), DE.LastprivateVars.end());
  };

  /// ---------------------------- Attributes ---------------------------- ///
  SmallVector<Value *, 4> PrivateVars;
  SmallVector<Value *, 4> SharedVars;
  SmallVector<Value *, 4> FirstprivateVars;
  SmallVector<Value *, 4> LastprivateVars;
};

inline raw_ostream &operator<<(raw_ostream &OS, DataEnv &DE) {
  OS << "Data environment: \n";
  OS << "Firstprivate: " << DE.FirstprivateVars.size() << "\n";
  for(auto *V : DE.FirstprivateVars)
    OS << "  - " << *V << "\n";
  OS << "Private: " << DE.PrivateVars.size() << "\n";
  for(auto *V : DE.PrivateVars)
    OS << "  - " << *V << "\n";
  OS << "Shared: " << DE.SharedVars.size() << "\n";
  for(auto *V : DE.SharedVars)
    OS << "  - " << *V << "\n";
  OS << "Lastprivate: " << DE.LastprivateVars.size() << "\n";
  for(auto *V : DE.LastprivateVars)
    OS << "  - " << *V << "\n";
  OS << "\n";
  return OS;
}

/// ---------------------------- OMP INFO ---------------------------- ///
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

  /// ---------------------------- Interface ---------------------------- ///
  OMPInfo() : RTF(OTHER), CB(nullptr), F(nullptr) {};
  OMPInfo(OMPInfo &OI) 
    : RTF(OI.RTF), DE(OI.DE), CB(OI.CB), F(OI.F) {};
  OMPInfo(RTFType RTF, CallBase *CB) 
    : RTF(RTF), CB(CB), F(nullptr) {};
  OMPInfo &operator=(const OMPInfo &OI) {
    RTF = OI.RTF;
    DE = OI.DE;
    CB = OI.CB;
    F = OI.F;
    return *this;
  }
  /// ---------------------------- Attributes ---------------------------- ///
  RTFType RTF;
  DataEnv DE;
  CallBase *CB;         // CallBase of the OpenMP region
  Function *F;          // Pointer to the outlined function

  /// ---------------------------- Helper Functions ---------------------------- ///
  static RTFType getRTFunction(Function *F) {
    if(!F)
      return RTFType::OTHER;
    auto calleeName = F->getName();
    if(calleeName == "__kmpc_fork_call")
      return RTFType::PARALLEL;
    else if(calleeName == "__kmpc_omp_task_alloc")
      return RTFType::TASKALLOC;
    else if(calleeName == "__kmpc_omp_task")
      return RTFType::TASK;
    else if(calleeName == "__kmpc_omp_task_alloc_with_deps")
      return RTFType::TASKDEP;
    else if(calleeName == "__kmpc_omp_taskwait")
      return RTFType::TASKWAIT;
    else if(calleeName == "omp_set_num_threads")
      return RTFType::SET_NUM_THREADS;
    else if(calleeName == "__kmpc_for_static_init_4")
      return RTFType::PARALLEL_FOR;
    return RTFType::OTHER;
  }

  static RTFType getRTFunction(CallBase &CB) {
    auto *Callee = CB.getCalledFunction();
    return getRTFunction(Callee);
  }

  static bool isTaskFunction(Function *F) {
    auto RT = getRTFunction(F);
    if(RT == RTFType::TASK || RT == RTFType::TASKDEP || RT == RTFType::TASKWAIT )
      return true;
    return false;
  }

  const StringRef getCBName() const {
    if(CB)
      return CB->getCalledFunction()->getName();
    return "";
  }

  const StringRef getFName() const {
    if(F)
      return F->getName();
    return "";
  }

  const Function *getCBFunction() const {
    if(CB)
      return CB->getCalledFunction();
    return nullptr;
  }
};

/// ---------------------------- EDT DEP ---------------------------- ///
struct EDTInfo;

struct EDTDep {
  /// ---------------------------- Interface ---------------------------- ///
  EDTDep() : EDT(nullptr) {};
  EDTDep(EDTInfo *EDT) : EDT(EDT) {};
  // EDTDep(EDTInfo *EDT, DataEnv *DE) : EDT(EDT), DE(DE) {};

  /// ---------------------------- Attributes ---------------------------- ///
  // EDT where the value will be signaled to
  EDTInfo *EDT;
  /// Values to be signaled
  SmallVector<Value*, 4> values;
};

/// ---------------------------- EDT INFO ---------------------------- ///
struct EDTBlock {
  enum BlockType {
    ENTRY = 0,
    INIT_ALLOCA,
    INIT,
    EVENT,
    OTHER
  };
  /// ---------------------------- Interface ---------------------------- ///
  EDTBlock(EDTInfo *EDT) : EDT(EDT), Type(OTHER), BB(nullptr) {};
  EDTBlock(EDTInfo *EDT, BlockType Type, BasicBlock *BB) 
          : EDT(EDT), Type(Type), BB(BB) {};
  EDTBlock(EDTBlock &B) 
          : EDT(B.EDT), Type(B.Type), BB(B.BB) {};
  /// Overwrite = operator
  // EDTBlock &operator=(const EDTBlock &B) {
  //   EDT = B.EDT;
  //   Type = B.Type;
  //   BB = B.BB;
  //   return *this;
  // }

  void setOMPInfo(OMPInfo &OMPI) {
    HasOMP = true;
    OMP = OMPI;
  }

  /// ---------------------------- Attributes ---------------------------- ///
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
  if(EB.Type == EDTBlock::INIT_ALLOCA)
    OS << "Init Alloca";
  else if(EB.Type == EDTBlock::INIT)
    OS << "Init";
  else if(EB.Type == EDTBlock::EVENT)
    OS << "Event";
  else if(EB.Type == EDTBlock::ENTRY)
    OS << "Entry";
  else
    OS << "Other";
  OS << "\n";
  auto HasOMP = EB.HasOMP ? "True" : "False";
  OS << "  - OMP: " << HasOMP << "\n";
  OS << "  - Transformed: " << Transformed << "\n";
  return OS;
}

/// ---------------------------- EDT INFO ---------------------------- ///
/// Struct to store information about EDTs
struct EDTInfo {
  /// ---------------------------- Interface ---------------------------- ///
  EDTInfo(uint64_t ID) : ID(ID), F(nullptr) {};
  EDTInfo(uint64_t ID, Function *F, DataEnv DE) 
        : ID(ID), F(F), DE(DE) {};

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

  void addDep(EDTDep *SI) {
    Deps.insert(SI);
  }

  /// ---------------------------- Attributes ---------------------------- ///
  /// GUID of the EDT
  uint64_t ID;
  /// Pointer to the EDT function
  Function *F;
  /// Data environment of the EDT
  DataEnv DE;
  /// Init EDTBlock
  EDTBlock *Init = nullptr;
  /// List of EDT Dependencies - successors
  SmallPtrSet<EDTDep *, 4> Deps;
  /// List of BasicBlocks that are part of the EDT
  SmallPtrSet<EDTBlock *, 4> Blocks;
  /// Set of instructions that may read or write to memory
  SmallPtrSet<Instruction *, 4> RWInsts;
  /// Set of values not declared in the EDT
  SmallPtrSet<Instruction *, 4> ExternalValues;
};

inline raw_ostream &operator<<(raw_ostream &OS, EDTInfo &EI) {
  OS << "----- EDT -----\n";
  OS << "ID: " << EI.ID << "\n";
  OS << "Number of Deps: " << EI.Deps.size() << "\n";
  for(auto *Dep : EI.Deps){
    OS << "  - " << Dep->EDT->ID << "\n";
  }
  OS << "Number of RWInsts: " << EI.RWInsts.size() << "\n";
  for(auto *I : EI.RWInsts) {
    OS << "  - " << *I << "\n";
  }
  OS << "Number of ExternalValues: " << EI.ExternalValues.size() << "\n";
  for(auto *I : EI.ExternalValues) {
    OS << "  - " << *I << "\n";
  }
  OS << "Number of Blocks: " << EI.Blocks.size() << "\n";
  for(auto *B : EI.Blocks) {
    OS << *B << "\n";
  }
  return OS;
}

/// ---------------------------- ARTS TRANSFORM ---------------------------- ///
struct ARTSTransformer {
  /// ---------------------------- Interface ---------------------------- ///
  ARTSTransformer(Module &M) : M(M) {}
  ~ARTSTransformer() {
    /// Delete EDTs
    // for (const auto &It : EDTBlocks)
    //   delete (It.second);
    // EDTBlocks.clear();
  }
  bool run(Attributor &A);
  bool runAttributor(Attributor &A);

  /// ---------------------------- Helper Functions ---------------------------- ///
  /// Insert EDT
  EDTInfo *insertEDT(Function *F) {
    EDTInfo *EDTI = new EDTInfo(EDTID++);
    EDTs.insert(EDTI);
    EDTsFromFunction[F].insert(EDTI);
    return EDTI;
  }

  /// Insert EDT and add dependency. 
  /// Creates a new EDT and adds a dependency from the given EDT to the new one.
  EDTInfo *insertEDTWithDep(EDTInfo *From, Function *F) {
    EDTInfo *EDTI = insertEDT(F);
    /// Create dependency
    EDTDep *SI = new EDTDep(EDTI);
    From->addDep(SI);
    /// Return new EDT
    return EDTI;
  }

  /// Insert EDT, add dependency and init EDTBlock.
  EDTInfo *insertEDTWithDep(EDTInfo *From, EDTBlock *Init) {
    EDTInfo *EDTI = insertEDTWithDep(From, Init->BB->getParent());
    EDTI->Init = Init;
    return EDTI;
  }

  /// ---------------------------- EDTBlock Map ---------------------------- ///
  /// Return the EDTBlock for a given BB or `nullptr` if there are
  /// none.
  EDTBlock *getEDTBlock(BasicBlock *BB) {
    auto It = EDTBlocks.find(BB);
    if (It != EDTBlocks.end())
      return (It->second);
    return nullptr;
  }

  /// Create EDTBlock and insert it into the EDT and map
  EDTBlock *insertEDTBlock(
      EDTInfo *EI, EDTBlock::BlockType Ty, BasicBlock *BB) {
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
  EDTBlock *insertEDTBlock(EDTInfo *EI, EDTBlock::BlockType Ty,
                           BasicBlock *BB, OMPInfo &OMP) {
    EDTBlock *EDTB = insertEDTBlock(EI, Ty, BB);
    EDTB->setOMPInfo(OMP);
    return EDTB;
  }

  /// ---------------------------- Attributes ---------------------------- ///
  /// The underlying module.
  Module &M;
  /// Set of valid functions in the module.
  // SetVector<Function *> &Functions;
  /// List of EDTs
  uint64_t EDTID = 0;
  SmallPtrSet<EDTInfo *, 4> EDTs;
  /// Maps a function to the set of EDTs that were created from it
  DenseMap<Function *, SmallPtrSet<EDTInfo *, 4>> EDTsFromFunction;
  /// Maps the basic block to the EDT Block
  DenseMap<BasicBlock *, EDTBlock *> EDTBlocks;
  /// 
};

inline raw_ostream &operator<<
    (raw_ostream &OS, SmallPtrSet<EDTInfo*, 4> &EDTs) {
  OS << "DUMPING EDTs\n";
  OS << "Number of EDTs: " << EDTs.size() << "\n";
  for(auto *E : EDTs) {
    OS << "\n" << *E;
  }
  return OS;
}

/// ---------------------------- ARTS TRANSFORM PASS ---------------------------- ///
/// From OpenMP to ARTS transformation pass.
class ARTSTransformPass : public PassInfoMixin<ARTSTransformPass> {
public:
  ARTSTransformPass() = default;
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
};
} // namespace llvm

#endif // LLVM_TRANSFORMS_ARTS_H

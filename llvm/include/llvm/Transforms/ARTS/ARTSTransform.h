#ifndef LLVM_TRANSFORMS_ARTS_H
#define LLVM_TRANSFORMS_ARTS_H

#include "llvm/Transforms/IPO/Attributor.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/IR/PassManager.h"

namespace llvm {

enum RTFunction {
  OTHER = 0,
  PARALLEL,
  PARALLEL_FOR,
  TASKALLOC,
  TASK,
  TASKWAIT,
  TASKDEP,
  SET_NUM_THREADS
};


/// ---------------------------- DATA ENVIRONMENT ---------------------------- ///
/// Struct to store information about the data environment of the OpenMP regions

struct DataEnv {
  /// ---------------------------- Interface ---------------------------- ///
  DataEnv() : RTF(OTHER), CB(nullptr), F(nullptr) {};
  DataEnv(DataEnv &DE) : RTF(DE.RTF), CB(DE.CB), F(DE.F) {};
  DataEnv(RTFunction RTF, CallBase *CB) 
    : RTF(RTF), CB(CB), F(nullptr) {};
  DataEnv(RTFunction RTF, CallBase *CB, Function *F) 
    : RTF(RTF), CB(CB), F(F) {};

  void append(const DataEnv &DE) {
    if(DE.RTF != RTF || DE.CB != CB)
      return;
    if(DE.F && !F)
      F = DE.F;
    PrivateVars.append(DE.PrivateVars.begin(), DE.PrivateVars.end());
    SharedVars.append(DE.SharedVars.begin(), DE.SharedVars.end());
    FirstprivateVars.append(DE.FirstprivateVars.begin(), DE.FirstprivateVars.end());
    LastprivateVars.append(DE.LastprivateVars.begin(), DE.LastprivateVars.end());
  };

  void clamp(const DataEnv &DE) {
    RTF = DE.RTF;
    CB = DE.CB;
    F = DE.F;
    append(DE);
  };

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

  /// ---------------------------- Attributes ---------------------------- ///
  RTFunction RTF;
  CallBase *CB;         // CallBase of the OpenMP region
  Function *F;          // Pointer to the outlined function
  SmallVector<Value *, 4> PrivateVars;
  SmallVector<Value *, 4> SharedVars;
  SmallVector<Value *, 4> FirstprivateVars;
  SmallVector<Value *, 4> LastprivateVars;
};

inline raw_ostream &operator<<(raw_ostream &OS, DataEnv &DE) {
    OS << "Data environment for " << DE.getCBName() << "\n";
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

/// ---------------------------- SIGNAL INFO ---------------------------- ///
struct EDTInfo;

struct EDTDep {
  /// ---------------------------- Interface ---------------------------- ///
  EDTDep() : EDT(nullptr) {};
  EDTDep(EDTInfo *EDT) : EDT(EDT) {};
  // EDTDep(EDTInfo *EDT, DataEnv *DE) : EDT(EDT), DE(DE) {};

  /// ---------------------------- Attributes ---------------------------- ///
  // EDT where the value will be signaled to
  EDTInfo *EDTI;
  /// Values to be signaled
  SmallVector<Value*, 4> values;
}

/// ---------------------------- EDT INFO ---------------------------- ///
/// Struct to store information about the EDT initializer
struct EDTInitInfo {
  /// ---------------------------- Interface ---------------------------- ///
  EDTInitInfo() : F(nullptr), CB(nullptr) {};
  EDTInitInfo(Function *F, CallBase *CB) : F(F), CB(CB) {};
  EDTInitInfo(RTFunction RTF, CallBase *CB) 
    : F(nullptr), CB(CB) {};
  /// ---------------------------- Attributes ---------------------------- ///
  /// List of basic blocks that are part of the EDT initializer
  SmallVector<BasicBlock*, 4> EDTInit;
  /// 
  Function *F;          // Pointer to the outlined function
  CallBase *CB;         // CallBase of the OpenMP region
};

struct EDTBody {
  /// ---------------------------- Interface ---------------------------- ///
  EDTBody() : Ty(OTHER), BB(nullptr) {};
  EDTBody(EDTBodyType Ty, BasicBlock *BB) : Ty(Ty), BB(BB) {};
  EDTBody(EDTBody EDTB) : Ty(EDTB.Ty), BB(EDTB.BB) {};

  /// ---------------------------- Attributes ---------------------------- ///
  enum Type {
    OTHER = 0,
    INIT_ALLOCA,
    INIT,
    EVENT
  };
  EDTBodyType Ty;
  BasicBlock *BB;
  bool transformed = false;
}

inline raw_ostream &operator<<(raw_ostream &OS, EDTBody &EB) {
  StringRef transformed = EB.transformed ? "True: " : "False";
  if(EB.type == EDTBody::INIT_ALLOCA)
    OS << "Init Alloca"
  else if(EB.type == EDTBody::INIT)
    OS << "Init";
  else if(EB.type == EDTBody::EVENT)
    OS << "Event";
  else
    OS << "Other";
  OS << " - Transformed: "<< transformed << *(EB.BB) << "\n";
  return OS;
}

/// Struct to store information about EDTs
struct EDTInfo {
  /// ---------------------------- Interface ---------------------------- ///
  EDTInfo(uint64_t ID) : ID(ID), F(nullptr) {};
  EDTInfo(Function *F, DataEnv DE) : F(F), DE(DE) {};
  EDTInfo(RTFunction DE_RTF, CallBase *DE_CB) 
    : F(nullptr), DE(DE_RTF, DE_CB) {};

  void insertEDTBody(EDTBody Body) {
    EDTBody *EDTB = new EDTBody(Body);
    body.push_back(EDTB);
  }

  void insertEDTBody(EDTBodyType Ty, BasicBlock *BB) {
    EDTBody *EDTB = new EDTBody(Ty, BB);
    body.push_back(EDTB);
  }

  void addDep(EDTDep *SI) {
    deps.push_back(SI);
  }

  /// ---------------------------- Attributes ---------------------------- ///
  /// GUID of the EDT
  uint64_t ID;
  // Pointer to the EDT function
  Function *F;
  /// Data environment of the EDT
  DataEnv DE;
  /// List of EDT Dependencies - successors
  SmallVector<EDTDep*, 4> deps;
  /// List of BasicBlocks that are part of the EDT
  SmallVector<EDTBody*, 4> body;
};

inline raw_ostream &operator<<(raw_ostream &OS, EDTInfo &EI) {
  OS << "EDT body: " << body.size() << "\n";
  for(auto *EDTB : body)
    OS << "  - " << *EDTB << "\n";
  OS << "EDT deps: " << deps.size() << "\n";

  OS << "\n";
  return OS;
}

/// ---------------------------- ARTS TRANSFORM ---------------------------- ///
struct ARTSTransformer {
  /// ---------------------------- Interface ---------------------------- ///
  ARTSTransformer(Module &M) : M(M) {}
  ~ARTSTransformer() {
    /// Delete EDTs
    for (const auto &It : EDTInitRegions)
      delete (It.second);
    EDTInitRegions.clear();
  }
  bool run(Attributor &A);
  bool runAttributor(Attributor &A);


  /// Return the EDT info for a given BB or `nullptr` if there are
  /// none.
  EDTInfo *getEDTInfo(BasicBlock &BB) {
    auto It = EDTInitRegions.find(&BB);
    if (It != EDTInitRegions.end())
      return (It->second);
    return nullptr;
  }

  /// Insert a new EDT info for a given BB and remove the old one if
  /// there was one.
  void insertEDTInitRegion(BasicBlock *BB, EDTInfo *EDTI) {
    auto It = EDTInitRegions.find(BB);
    if (It != EDTInitRegions.end())
      delete (It->second);
    EDTInitRegions.insert(std::make_pair(BB, EDTI));
  }

  void insertEDTInitRegion(BasicBlock *BB, EDTInfo EDTI) {
    insertEDTInitRegion(BB, new EDTInfo(EDTI));
  }

  void insertEDTInitRegion(BasicBlock *BB, RTFunction DE_RTF, CallBase *DE_CB) {
    insertEDTInitRegion(BB, new EDTInfo(DE_RTF, DE_CB));
  }

  /// ---------------------------- Helper Functions ---------------------------- ///
  /// Insert EDT
  EDTInfo *insertEDT() {
    EDTInfo *EDTI = new EDTInfo(EDTID++);
    EDTs.push_back(EDTI);
    return EDTI;
  }

  /// Insert EDT and add dependency. 
  /// Creates a new EDT and adds a dependency from the 
  /// given EDT to the new one.
  EDTInfo *insertEDTWithDep(EDTInfo *from) {
    EDTInfo *EDTI = insertEDT(EDTID++);
    /// Create dependency
    EDTDep *SI = new EDTDep(EDTI);
    from->addDep(SI);
    /// Return new EDT
    return EDTI;
  }

  /// ---------------------------- Attributes ---------------------------- ///
  /// The underlying module.
  Module &M;
  /// Set of valid functions in the module.
  // SetVector<Function *> &Functions;
  /// List of EDTs
  uint64_t EDTID = 0;
  SmallVector<EDTInfo*, 4> EDTs;
  /// EDT Init Regions
  DenseMap<BasicBlock *, EDTInfo *> EDTInitRegions;
  /// 
};

inline raw_ostream &operator<<
    (raw_ostream &OS, SmallVector<EDTInfo*, 4> &EDTs) {
  OS << "EDTs: " << EDTs.size() << "\n";
  for(auto *EDTI : EDTs)
    OS << "  - " << *EDTI << "\n";
  OS << "\n";
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

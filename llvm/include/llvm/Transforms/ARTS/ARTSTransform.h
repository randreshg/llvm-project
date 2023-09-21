#ifndef LLVM_TRANSFORMS_ARTS_H
#define LLVM_TRANSFORMS_ARTS_H

#include "llvm/IR/PassManager.h"
#include "llvm/Transforms/IPO/Attributor.h"

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
  DataEnv(DataEnv &DE) : RTF(DE.RTF), CB(DE.CB), F(F) {};
  DataEnv(RTFunction RTF, CallBase *CB) 
    : RTF(RTF), CB(CB), F(nullptr) {};
  DataEnv(RTFunction RTF, CallBase *CB, Function *F) 
    : RTF(RTF), CB(CB), F(F) {};

  void append(DataEnv &DE) {
    if(DE.RTF != RTF || DE.CB != CB)
      return;
    if(DE.F && !F)
      F = DE.F;
    PrivateVars.append(DE.PrivateVars.begin(), DE.PrivateVars.end());
    SharedVars.append(DE.SharedVars.begin(), DE.SharedVars.end());
    FirstprivateVars.append(DE.FirstprivateVars.begin(), DE.FirstprivateVars.end());
    LastprivateVars.append(DE.LastprivateVars.begin(), DE.LastprivateVars.end());
  };

  StringRef getCBName() {
    if(CB)
      return CB->getCalledFunction()->getName();
    return "";
  }

  StringRef getFName() {
    if(F)
      return F->getName();
    return "";
  }

  Function *getCBFunction() {
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

/// ---------------------------- EDT INFO ---------------------------- ///
/// Struct to store information about EDTs
struct EDTInfo {
  /// ---------------------------- Interface ---------------------------- ///
  EDTInfo() : F(nullptr) {};
  EDTInfo(Function *F, DataEnv DE) : F(F), DE(DE) {};
  EDTInfo(RTFunction RTF, Function *F) : F(nullptr), DE(RTF, F) {};
  /// ---------------------------- Attributes ---------------------------- ///
  Function *F;
  DataEnv DE;
};

/// ---------------------------- ARTS TRANSFORM ---------------------------- ///
struct ARTSTransformer {
  /// ---------------------------- Interface ---------------------------- ///
  ARTSTransformer(Module &M, SetVector<Function *> &Functions) 
                : M(M), Functions(Functions) {}
  bool run(Attributor &A);
  bool runAttributor(Attributor &A);

  /// Return the EDT info for a given BB or `nullptr` if there are
  /// none.
  const EDTInfo *getEDTInfo(BasicBlock &BB) const {
    auto I = EDTRegions.find(&BB);
    if (I != EDTRegions.end())
      return &(I->second.get());
    return nullptr;
  }

private:
  bool identifyEDTRegions();

  /// ---------------------------- Attributes ---------------------------- ///
  /// The underlying module.
  Module &M;
  /// Set of valid functions in the module.
  SetVector<Function *> &Functions;
  /// EDT Regions
  DenseMap<BasicBlock *, EDTInfo > EDTRegions;
};

/// ---------------------------- ARTS TRANSFORM PASS ---------------------------- ///
/// From OpenMP to ARTS transformation pass.
class ARTSTransformPass : public PassInfoMixin<ARTSTransformPass> {
public:
  ARTSTransformPass() = default;
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
};
} // namespace llvm

#endif // LLVM_TRANSFORMS_ARTS_H

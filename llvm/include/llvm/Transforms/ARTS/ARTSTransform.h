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

struct DataEnv {
  RTFunction RTF;
  Function *F;
  SmallVector<Value *, 4> PrivateVars;
  SmallVector<Value *, 4> SharedVars;
  SmallVector<Value *, 4> FirstprivateVars;
  SmallVector<Value *, 4> LastprivateVars;
};

/// ARTS transform 
struct ARTSTransform {
  ARTSTransform(Module &M,  Attributor &A, SetVector<Function *> &Functions) 
                : M(M), A(A), Functions(Functions) {}
  bool run();
  bool runAttributor();

  /// Attributes
  /// The underlying module.
  Module &M;
  /// Attributor instance.
  Attributor &A;
  /// Set of valid functions in the module.
  SetVector<Function *> &Functions;
};

/// From OpenMP to ARTS transformation pass.
class ARTSTransformPass : public PassInfoMixin<ARTSTransformPass> {
public:
  ARTSTransformPass() = default;
  // ARTSTransformPass(ThinOrFullLTOPhase LTOPhase) : LTOPhase(LTOPhase) {}

  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);

private:
  // const ThinOrFullLTOPhase LTOPhase = ThinOrFullLTOPhase::None;
};
} // namespace llvm

#endif // LLVM_TRANSFORMS_ARTS_H

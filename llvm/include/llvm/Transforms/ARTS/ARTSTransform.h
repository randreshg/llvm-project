#ifndef LLVM_TRANSFORMS_ARTS_H
#define LLVM_TRANSFORMS_ARTS_H

#include "llvm/IR/PassManager.h"

namespace llvm {

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

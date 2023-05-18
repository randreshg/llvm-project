//===- IPO/OpenMPOpt.h - Collection of OpenMP optimizations -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_IPO_OPENMPOPT_H
#define LLVM_TRANSFORMS_IPO_OPENMPOPT_H

#include "llvm/Analysis/CGSCCPassManager.h"
#include "llvm/Analysis/LazyCallGraph.h"
#include "llvm/IR/PassManager.h"

namespace llvm {

namespace omp {

/// Summary of a kernel (=entry point for target offloading).
using Kernel = Function *;

/// Set of kernels in the module
using KernelSet = SetVector<Kernel>;

/// Helper to determine if \p M contains OpenMP.
bool containsOpenMP(Module &M);

/// Helper to determine if \p M is a OpenMP target offloading device module.
bool isOpenMPDevice(Module &M);

/// Get OpenMP device kernels in \p M.
KernelSet getDeviceKernels(Module &M);

/// Tasks structures and typedefs extracted from kmp.h
typedef int kmp_int32;
typedef int64_t kmp_int64;
typedef intptr_t kmp_intptr_t;
typedef unsigned char uint8;

typedef struct kmp_depend_info {
  Value *base_addr;
  size_t len;
  union {
    uint8 flag; // flag as an unsigned char
    struct { // flag as a set of 8 bits
      unsigned in : 1;
      unsigned out : 1;
      unsigned mtx : 1;
      unsigned set : 1;
      unsigned unused : 3;
      unsigned all : 1;
    } flags;
  };
} TaskDependInfo;

struct TaskInfo {
  int id;                                     // Task id
  SmallVector<uint64_t, 2> successors;        // Ids of successors
  SmallVector<uint64_t, 2> predecessors;      // Ids of predecessors
  SmallVector<TaskDependInfo, 2> TaskDepInfo; // Task dependency information
  // SmallVector<int64_t> FirstPrivateData;
};

} // namespace omp

/// OpenMP optimizations pass.
class OpenMPOptPass : public PassInfoMixin<OpenMPOptPass> {
public:
  OpenMPOptPass() = default;
  OpenMPOptPass(ThinOrFullLTOPhase LTOPhase) : LTOPhase(LTOPhase) {}

  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);

private:
  const ThinOrFullLTOPhase LTOPhase = ThinOrFullLTOPhase::None;
};

class OpenMPOptCGSCCPass : public PassInfoMixin<OpenMPOptCGSCCPass> {
public:
  OpenMPOptCGSCCPass() = default;
  OpenMPOptCGSCCPass(ThinOrFullLTOPhase LTOPhase) : LTOPhase(LTOPhase) {}

  PreservedAnalyses run(LazyCallGraph::SCC &C, CGSCCAnalysisManager &AM,
                        LazyCallGraph &CG, CGSCCUpdateResult &UR);

private:
  const ThinOrFullLTOPhase LTOPhase = ThinOrFullLTOPhase::None;
};

} // end namespace llvm

#endif // LLVM_TRANSFORMS_IPO_OPENMPOPT_H

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
typedef unsigned char uint8;

/// Task dependency information
typedef struct kmp_depend_info {
  Value *BasePtr;               // Alloca instruction of the base pointer
  size_t BaseLen;               // Size of the base pointer
  // Offset for array section
  // Size -> infinite...
  union {
    uint8 Flag;                 // Flag as an unsigned char
    struct {                    // Flag as a set of 8 bits
      unsigned in : 1;
      unsigned out : 1;
      unsigned mtx : 1;
      unsigned set : 1;
      unsigned unused : 3;
      unsigned all : 1;
    } Flags;
  };
} TaskDependInfo;

/// Task information
struct TaskInfo {
  TaskInfo(CallBase *TaskCB, bool HasDep) : TaskCB(TaskCB), HasDep(HasDep) {}

  CallBase *TaskCB = nullptr;
  bool HasDep = false;
  // int id;                                     // Task id
  SmallVector<uint64_t, 2> successors;        // Ids of successors
  SmallVector<uint64_t, 2> predecessors;      // Ids of predecessors
  SmallVector<TaskDependInfo, 2> TaskDepInfo; // Task dependency information
  // SmallVector<int64_t> FirstPrivateData;
};

/// Task dependency graph
// class TaskDependencyGraph {
// SmallVector<TaskInfo *> Tasks;

// public:
//   bool addTask(TaskInfo &TaskFound) {
//     Tasks.push_back(&TaskFound);
//     return true;
//   }
//   bool checkDependency(TaskDependInfo &Source, TaskDependInfo &Dest);
//   void addTaskDependInfo(TaskInfo &TaskFound, CallInst &TaskCall);
// };

/// Set of kernels in the module
using TaskSet = SmallVector<TaskInfo>;
///  Map from callbase to its position in the taskset
using TaskMap = DenseMap<CallBase *, int>;

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

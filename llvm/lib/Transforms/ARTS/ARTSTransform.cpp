//===- TaskDependencyGraph.cpp - Generation of a static OpenMP task dependency
// graph
//--------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Generates an static OpenMP task dependency graph.
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/ARTS/ARTSTransform.h"

#include "llvm/Support/Debug.h"
#include "llvm/Transforms/IPO/Attributor.h"
#include "llvm/IR/BasicBlock.h"

using namespace llvm;

// DEBUG
#define DEBUG_TYPE "arts-transform"
#if !defined(NDEBUG)
static constexpr auto TAG = "[" DEBUG_TYPE "] ";
#endif

// COMMAND LINE OPTIONS
static cl::opt<bool> DisableARTSTransformation(
    "arts-disable", cl::desc("Disable transformation to ARTS."),
    cl::Hidden, cl::init(false));

static cl::opt<bool> PrintModuleBeforeOptimizations(
    "arts-print-module-before",
    cl::desc("Print the module before the ARTSTransform Module Pass"),
    cl::Hidden, cl::init(false));

static cl::opt<bool> PrintModuleAfterOptimizations(
    "arts-print-module-after",
    cl::desc("Print the module after the ARTSTransform Module Pass"),
    cl::Hidden, cl::init(false));

// ARTS

// ARTS TRANSFORMATION PASS
PreservedAnalyses ARTSTransformPass::run(Module &M, ModuleAnalysisManager &AM) {
  /// Command line options
  if (DisableARTSTransformation)
    return PreservedAnalyses::all();
  
  if (PrintModuleBeforeOptimizations)
    LLVM_DEBUG(dbgs() << TAG << "Module before ARTSTransform Module Pass:\n" << M);

  /// Run the ARTSTransform Module Pass
  LLVM_DEBUG(dbgs() << TAG << "Run the ARTSTransform Module Pass\n");
  FunctionAnalysisManager &FAM =
      AM.getResult<FunctionAnalysisManagerModuleProxy>(M).getManager();

  bool Changed = false;

  /// Get the set of functions in the module
  SetVector<Function *> Functions;

  LLVM_DEBUG(dbgs() << TAG << "Set of functions in the module:\n");
  for (Function &F : M) {
    if (F.isDeclaration() && !F.hasLocalLinkage())
      continue;
    Functions.insert(&F);
    /// Print the function name
    LLVM_DEBUG(dbgs() << "Function: " << F.getName() << "\n");
  }

  if (Functions.empty())
    return PreservedAnalyses::none();


  LLVM_DEBUG(dbgs() << "\n" << TAG << "Module information:\n");
  /// Iterate through the BB of each function
  for(auto *F : Functions) {
    LLVM_DEBUG(dbgs() << "-----\nFunction: " << F->getName());
    for(auto &BB : *F) {
      /// Print function name and BB information
      LLVM_DEBUG(dbgs() << BB << "\n");
      /// Iterate through the instructions of each BB
      // for(auto &I : BB) {
      //   /// Check if the instruction is a call to omp_set_num_threads
      //   if(auto *CI = dyn_cast<CallInst>(&I)) {
      //     if(auto *Callee = CI->getCalledFunction()) {
      //       if(Callee->getName() == "omp_set_num_threads") {
      //         /// Get the number of threads
      //         auto *Arg = CI->getArgOperand(0);
      //         if(auto *NumThreads = dyn_cast<ConstantInt>(Arg)) {
      //           /// Print the number of threads
      //           LLVM_DEBUG(dbgs() << "Number of threads: " << NumThreads->getZExtValue() << "\n");
      //         }
      //       }
      //     }
      //   }
    }
  }

  /// The analysis start in the main function
  Function *Main = M.getFunction("main");

  if (Changed)
    return PreservedAnalyses::none();

  return PreservedAnalyses::all();
}


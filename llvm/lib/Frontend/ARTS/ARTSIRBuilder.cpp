//===- ARTSIRBuilder.cpp - Builder for LLVM-IR for ARTS directives ----===//
//===----------------------------------------------------------------------===//
/// \file
///
/// This file implements the ARTSIRBuilder class, which is used as a
/// convenient way to create LLVM instructions for ARTS directives.
///
//===----------------------------------------------------------------------===//

#include "llvm/Frontend/ARTS/ARTSIRBuilder.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/CodeMetrics.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/OptimizationRemarkEmitter.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/Frontend/ARTS/ARTSConstants.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/Debug.h"
#include "llvm/Transforms/ARTS/ARTSTransform.h"
// #include "llvm/MC/TargetRegistry.h"
// #include "llvm/Support/CommandLine.h"
// #include "llvm/Support/ErrorHandling.h"
// #include "llvm/Support/FileSystem.h"
// #include "llvm/Target/TargetMachine.h"
// #include "llvm/Target/TargetOptions.h"
// #include "llvm/Transforms/Utils/BasicBlockUtils.h"
// #include "llvm/Transforms/Utils/Cloning.h"
// #include "llvm/Transforms/Utils/CodeExtractor.h"
// #include "llvm/Transforms/Utils/LoopPeel.h"
// #include "llvm/Transforms/Utils/UnrollLoop.h"

#include <cstdint>
#include <optional>

// DEBUG
#define DEBUG_TYPE "arts-ir-builder"
#if !defined(NDEBUG)
static constexpr auto TAG = "[" DEBUG_TYPE "] ";
#endif

using namespace llvm;
using namespace arts;

FunctionCallee ARTSIRBuilder::getOrCreateRuntimeFunction(Module &M,
                                                         RuntimeFunction FnID) {
  FunctionType *FnTy = nullptr;
  Function *Fn = nullptr;

  // Try to find the declaration in the module first.
  switch (FnID) {
#define ARTS_RTL(Enum, Str, IsVarArg, ReturnType, ...)                         \
  case Enum:                                                                   \
    FnTy = FunctionType::get(ReturnType, ArrayRef<Type *>{__VA_ARGS__},        \
                             IsVarArg);                                        \
    Fn = M.getFunction(Str);                                                   \
    break;
#include "llvm/Frontend/ARTS/ARTSKinds.def"
  }

  if (!Fn) {
    // Create a new declaration if we need one.
    switch (FnID) {
#define ARTS_RTL(Enum, Str, ...)                                               \
  case Enum:                                                                   \
    Fn = Function::Create(FnTy, GlobalValue::ExternalLinkage, Str, M);         \
    break;
#include "llvm/Frontend/ARTS/ARTSKinds.def"
    }

    LLVM_DEBUG(dbgs() << "Created ARTS runtime function " << Fn->getName()
                      << " with type " << *Fn->getFunctionType() << "\n");
    // addAttributes(FnID, *Fn);

  } else {
    LLVM_DEBUG(dbgs() << "Found ARTS runtime function " << Fn->getName()
                      << " with type " << *Fn->getFunctionType() << "\n");
  }

  assert(Fn && "Failed to create ARTS runtime function");

  return {FnTy, Fn};
}

Function *ARTSIRBuilder::getOrCreateRuntimeFunctionPtr(RuntimeFunction FnID) {
  FunctionCallee RTLFn = getOrCreateRuntimeFunction(M, FnID);
  auto *Fn = dyn_cast<llvm::Function>(RTLFn.getCallee());
  assert(Fn && "Failed to create ARTS runtime function pointer");
  return Fn;
}

void ARTSIRBuilder::initialize() {
  LLVM_DEBUG(dbgs() << TAG << "Initializing ARTSIRBuilder\n");
  // Initialize the module with the runtime functions.
  initializeTypes();
  LLVM_DEBUG(dbgs() << TAG << "ARTSIRBuilder initialized\n");
}

void ARTSIRBuilder::finalize() {
  LLVM_DEBUG(dbgs() << TAG << "Finalizing ARTSIRBuilder\n");
  // Finalize the module with the runtime functions.
  // finalizeModule(M);
  LLVM_DEBUG(dbgs() << TAG << "ARTSIRBuilder finalized\n");
}

AllocaInst *ARTSIRBuilder::reserveEDTGuid(BasicBlock *EntryBB, uint32_t Node) {
  auto OldInsertPoint = Builder.saveIP();
  Builder.SetInsertPoint(EntryBB);
  /// Create the call to reserve the GUID
  ConstantInt *ARTS_EDT_Enum =
      ConstantInt::get(Builder.getContext(), APInt(32, 1));
  Value *Args[] = {ARTS_EDT_Enum,
                   Builder.CreateIntCast(ConstantInt::get(Int32, Node), Int32,
                                         /*isSigned*/ false)};
  CallInst *ReserveGuidCall = Builder.CreateCall(
      getOrCreateRuntimeFunctionPtr(ARTSRTL_artsReserveGuidRoute), Args);
  /// Create allocation of the GUID
  AllocaInst *GuidAddr = Builder.CreateAlloca(Int32Ptr, nullptr, "guid.addr");
  /// Store the GUID
  Builder.CreateStore(ReserveGuidCall, GuidAddr);
  Builder.restoreIP(OldInsertPoint);
  return GuidAddr;
}

// void ARTSIRBuilder::insertEDTBlock(EDTBlock *EB, Function *EDTFunc) {
//   LLVM_DEBUG(dbgs() << TAG << "Inserting EDT Block\n");
//   auto *BB = EB->BB;
//   /// Detach BB from its parent
//   BB->removeFromParent();
//   /// Attach BB to the EDT Func
//   if(EB->isEntry()) {
//     /// Remove entry block
//     BasicBlock *EntryBB = &EDTFunc->getEntryBlock();
//     EntryBB->removeFromParent();
//   }
//   BB->insertInto(EDTFunc);
//   /// Redirect last BB to BB
//   BasicBlock *LastBB = &EDTFunc->back();
//   if(LastBB != BB)
//     redirectTo(LastBB, BB);
// }

Function *ARTSIRBuilder::createEdt(StringRef Name) {
  const std::string FuncName = (Name + ".edt").str();
  LLVM_DEBUG(dbgs() << TAG << "Creating EDT: " << FuncName << "\n");
  Function *Func =
      Function::Create(EdtFunction, GlobalValue::InternalLinkage, FuncName, M);
  /// Add entry BB that returns void
  BasicBlock *EntryBB = BasicBlock::Create(Builder.getContext(), "entry", Func);
  Builder.SetInsertPoint(EntryBB);
  Builder.CreateRetVoid();
  return Func;
}

Function *ARTSIRBuilder::initializeEDT(EdtInfo &EI, Function *EDTFunc,
                                       BasicBlock *CurBB) {
  auto &DE = EI.DE;
  /// Get CurBB parent
  Function *Func = CurBB->getParent();
  LLVM_DEBUG(dbgs() << TAG << "CurBB parent: " << Func->getName() << "\n");
  /// Generate entry region.
  /// This region will have the declarations of the GUIDs.
  BasicBlock *EntryBB =
      BasicBlock::Create(Builder.getContext(), "edt.entry", Func);
  StringRef BBNameAppend = "";
  if (CurBB) {
    /// Create a copy of CurBB terminator
    Instruction *Term = CurBB->getTerminator();
    if (!Term) {
      LLVM_DEBUG(dbgs() << TAG << "CurBB has no terminator\n");
      // return nullptr;
    } else
      LLVM_DEBUG(dbgs() << TAG << "CurBB has terminator: " << *Term << "\n");
    // redirectTo(CurBB, EntryBB);
    BBNameAppend = CurBB->getName();
    EntryBB->setName("edt.entry." + BBNameAppend);
  }
  /// Reserve the GUIDs for the EDTs
  AllocaInst *GuidAddr = reserveEDTGuid(EntryBB, 0);
  /// Create EDT body
  BasicBlock *BodyBB = BasicBlock::Create(Builder.getContext(),
                                          "edt.body." + BBNameAppend, Func);
  Builder.SetInsertPoint(BodyBB);
  /// Branch to the body
  redirectTo(EntryBB, BodyBB);

  /// Paramc are the number of static parameters.
  /// It corresponds to the number of first private variables.
  int32_t NumParamC = DE.FirstPrivates.size();
  AllocaInst *ParamC = Builder.CreateAlloca(Int32, nullptr, "paramc");
  Builder.CreateStore(ConstantInt::get(Int32, NumParamC), ParamC);

  /// Paramv are the static parameters that are copied into the EDT closure.
  /// It corresponds to the private variables.
  AllocaInst *ParamVArray =
      Builder.CreateAlloca(Int64Ptr, nullptr, "paramv.array");
  for (auto En : enumerate(DE.FirstPrivates)) {
    unsigned Index = En.index();
    Value *Val = En.value();
    /// Create the GEP to store the value in the ParamV array
    Value *ParamVArrayElemPtr = Builder.CreateConstInBoundsGEP2_64(
        Int64Ptr, ParamVArray, 0, Index, "paramv.array.elem." + Twine(Index));
    /// If the value is a constant int, we need to store it in a variable
    if (ConstantInt *CI = dyn_cast<ConstantInt>(Val)) {
      /// Create the variable
      Val = Builder.CreateAlloca(Int32, nullptr, "paramv.val." + Twine(Index));
      /// Store the value in the variable
      Builder.CreateStore(CI, Val);
    }
    /// Cast the value to int64
    Value *Casted = Builder.CreateBitCast(
        Val, Int64Ptr, "paramv.val." + Twine(Index) + ".casted");
    Builder.CreateStore(Casted, ParamVArrayElemPtr);
  }

  /// Depc is the number of dependencies required for the EDT to run.
  /// It corresponds to the number of shared variables.
  int32_t NumDepC = DE.Shareds.size();
  AllocaInst *DepC = Builder.CreateAlloca(Int32, nullptr, "depc");
  Builder.CreateStore(ConstantInt::get(Int32, NumDepC), DepC);

  /// Insert call to artsEdtCreateWithGuid
  // Function *EDTFunc = Function::Create(EdtFunction,
  //                                      GlobalValue::ExternalLinkage,
  //                                      FuncName, M);
  Value *Args[] = {Builder.CreateBitCast(EDTFunc, EdtFunctionPtr),
                   Builder.CreateBitCast(GuidAddr, Int32Ptr),
                   Builder.CreateLoad(Int32, ParamC),
                   Builder.CreateBitCast(ParamVArray, Int64Ptr),
                   Builder.CreateLoad(Int32, DepC)};

  Function *F = getOrCreateRuntimeFunctionPtr(ARTSRTL_artsEdtCreateWithGuid);
  LLVM_DEBUG(dbgs() << TAG << "Creating call to artsEdtCreateWithGuid: \n"
                    << *F << "\n");
  Builder.CreateCall(F, Args);
  return nullptr;
}

/// ---------------------------- Private ---------------------------- ///
void ARTSIRBuilder::initializeTypes() {
  LLVMContext &Ctx = M.getContext();
  StructType *T;
#define ARTS_TYPE(VarName, InitValue) VarName = InitValue;
#define ARTS_ARRAY_TYPE(VarName, ElemTy, ArraySize)                            \
  VarName##Ty = ArrayType::get(ElemTy, ArraySize);                             \
  VarName##PtrTy = PointerType::getUnqual(VarName##Ty);
#define ARTS_FUNCTION_TYPE(VarName, IsVarArg, ReturnType, ...)                 \
  VarName = FunctionType::get(ReturnType, {__VA_ARGS__}, IsVarArg);            \
  VarName##Ptr = PointerType::getUnqual(VarName);
#define ARTS_STRUCT_TYPE(VarName, StructName, Packed, ...)                     \
  T = StructType::getTypeByName(Ctx, StructName);                              \
  if (!T)                                                                      \
    T = StructType::create(Ctx, {__VA_ARGS__}, StructName, Packed);            \
  VarName = T;                                                                 \
  VarName##Ptr = PointerType::getUnqual(T);
#include "llvm/Frontend/ARTS/ARTSKinds.def"
}

/// ---------------------------- Utils ---------------------------- ///
void ARTSIRBuilder::redirectTo(BasicBlock *Source, BasicBlock *Target) {
  if (Instruction *Term = Source->getTerminator()) {
    auto *Br = cast<BranchInst>(Term);
    assert(!Br->isConditional() &&
           "BB's terminator must be an unconditional branch (or degenerate)");
    BasicBlock *Succ = Br->getSuccessor(0);
    Succ->removePredecessor(Source, /*KeepOneInputPHIs=*/true);
    Br->setSuccessor(0, Target);
    return;
  }
  /// Create unconditional branch
  BranchInst::Create(Target, Source);
}

//===----------------------------------------------------------------------===//
//
// This file defines the ARTSIRBuilder class and helpers used as a convenient
// way to create LLVM instructions for ARTS directives.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_API_ARTS_ARTSIRBUILDER_H
#define LLVM_API_ARTS_ARTSIRBUILDER_H

// #include "llvm/Analysis/MemorySSAUpdater.h"
// #include "llvm/IR/DebugLoc.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Frontend/ARTS/ARTSConstants.h"

#include "llvm/Transforms/ARTS/ARTSTransform.h"

namespace llvm {
// class ARTSIRBuilder;

/// An interface to create LLVM-IR for ARTS directives.
///
/// Each ARTS directive has a corresponding public generator method.
class ARTSIRBuilder {
public:
  /// Create a new ARTSIRBuilder operating on the given module \p M.
  ARTSIRBuilder(Module &M) : M(M), Builder(M.getContext()) {}
  ~ARTSIRBuilder() {}

  /// ---------------------------- Interface ---------------------------- ///
  /// Initialize the internal state, this will put structures types and
  /// potentially other helpers into the underlying module. Must be called
  /// before any other method and only once! This internal state includes
  /// Types used in the ARTSIRBuilder generated from ARTSKinds.def
  void initialize();

  /// Finalize the underlying module, e.g., by outlining regions.
  /// \param Fn                    The function to be finalized. If not used,
  ///                              all functions are finalized.
  void finalize();

  /// Type used throughout for insertion points.
  using InsertPointTy = IRBuilder<>::InsertPoint;

  /// Return the function declaration for the runtime function with \p FnID.
  FunctionCallee getOrCreateRuntimeFunction(Module &M,
                                            arts::RuntimeFunction FnID);

  Function *getOrCreateRuntimeFunctionPtr(arts::RuntimeFunction FnID);

  /// Interface to add ARTS methods

  /// Creates EDT Function. It creates an empty function with the
  /// correct signature and returns it.
  Function *createEdt(StringRef Name);

  /// Given a EDT Block and a EDT Function, it inserts the Basic Block
  /// into the Function.
  // void insertEDTBlock(EDTBlock *EB, Function *EDTFunc);



  /// Initializes EDT Function. It inserts the call to the
  /// runtime function to reserve the GUID for the EDT, then it calls 
  /// artsEdtCreateWithGuid to create the EDT. 
  Function *initializeEDT(EdtInfo &EDTI, Function *EDTFunc,
                          BasicBlock *CurBB = nullptr);
  AllocaInst *reserveEDTGuid(BasicBlock *EntryBB, uint32_t Node);

  /// Declarations for LLVM-IR types (simple, array, function and structure) are
  /// generated below. Their names are defined and used in ARTSKinds.def. Here
  /// we provide the declarations, the initializeTypes function will provide the
  /// values.
  ///
  ///{
  #define ARTS_TYPE(VarName, InitValue) Type *VarName = nullptr;
  #define ARTS_ARRAY_TYPE(VarName, ElemTy, ArraySize)                             \
    ArrayType *VarName##Ty = nullptr;                                            \
    PointerType *VarName##PtrTy = nullptr;
  #define ARTS_FUNCTION_TYPE(VarName, IsVarArg, ReturnType, ...)                  \
    FunctionType *VarName = nullptr;                                             \
    PointerType *VarName##Ptr = nullptr;
  #define ARTS_STRUCT_TYPE(VarName, StrName, ...)                                 \
    StructType *VarName = nullptr;                                               \
    PointerType *VarName##Ptr = nullptr;
  #include "llvm/Frontend/ARTS/ARTSKinds.def"
  ///}
  /// ---------------------------- Utils ---------------------------- ///
  /// Make \p Source branch to \p Target.
  ///
  /// Handles two situations:
  /// * \p Source already has an unconditional branch.
  /// * \p Source is a degenerate block (no terminator because the BB is
  ///             the current head of the IR construction).
  void redirectTo(BasicBlock *Source, BasicBlock *Target);

private:
  /// ---------------------------- Private ---------------------------- ///
  /// Create all simple and struct types exposed by the runtime and remember
  /// the llvm::PointerTypes of them for easy access later.
  void initializeTypes();

/// ---------------------------- Attributes ---------------------------- ///
  /// The underlying LLVM-IR module
  Module &M;

  /// The LLVM-IR Builder used to create IR.
  IRBuilder<> Builder;

};

} // end namespace llvm

#endif // LLVM_API_ARTS_ARTSIRBUILDER_H
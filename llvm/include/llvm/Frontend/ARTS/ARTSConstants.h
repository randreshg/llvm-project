//===- ARTSConstants.h - OpenMP related constants and helpers ------ C++ -*-===//
//===----------------------------------------------------------------------===//
/// \file
///
/// This file defines constants and helpers used when dealing with OpenMP.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_API_ARTS_ARTSCONSTANTS_H
#define LLVM_API_ARTS_ARTSCONSTANTS_H

#include "llvm/ADT/BitmaskEnum.h"

// #include "llvm/ADT/StringRef.h"

namespace llvm {
namespace arts {
/// IDs for all arts runtime library (RTL) functions.
enum class RuntimeFunction {
  #define ARTS_RTL(Enum, ...) Enum,
  #include "llvm/Frontend/ARTS/ARTSKinds.def"
};

#define ARTS_RTL(Enum, ...) constexpr auto Enum = arts::RuntimeFunction::Enum;
#include "llvm/Frontend/ARTS/ARTSKinds.def"

} // end namespace arts
} // end namespace llvm

#endif // LLVM_API_ARTS_ARTSCONSTANTS_H

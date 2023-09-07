add_llvm_component_library(LLVMarts
  ARTSTransform.cpp

  ADDITIONAL_HEADER_DIRS
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/Transforms
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/Transforms/ARTS

  DEPENDS
  intrinsics_gen
  omp_gen

  COMPONENT_NAME
  ARTS

  LINK_COMPONENTS
  AggressiveInstCombine
  Analysis
  BitReader
  BitWriter
  Core
  FrontendOpenMP
  InstCombine
  IRReader
  Linker
  Object
  ProfileData
  Scalar
  Support
  TargetParser
  TransformUtils
  Vectorize
  Instrumentation
  )

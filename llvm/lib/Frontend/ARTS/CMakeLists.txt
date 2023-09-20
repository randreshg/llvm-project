add_llvm_component_library(LLVMFrontendARTS
  ARTSIRBuilder.cpp

  ADDITIONAL_HEADER_DIRS
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/Frontend
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/Frontend/ARTS

  COMPONENT_NAME
  FrontendARTS

  DEPENDS
  intrinsics_gen
  omp_gen

  LINK_COMPONENTS
  Core
  Support
  TargetParser
  TransformUtils
  Analysis
  MC
  Scalar
  BitReader
  )

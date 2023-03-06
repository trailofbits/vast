#
# VAST Frontend Praser
#
link_directories(${LLVM_LIBRARY_DIR})

set( LLVM_LINK_COMPONENTS
  ${LLVM_TARGETS_TO_BUILD}
  Option
  Support
  TargetParser
)

get_property(DIALECT_LIBS GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(CONVERSION_LIBS GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

add_library(vast_frontend_tool
    compiler_invocation.cpp
)

target_link_libraries(vast_frontend_tool
    PRIVATE
      clangBasic
      clangDriver
      clangFrontend

      LLVMSupport

      vast::codegen
      vast::frontend
      vast::settings
)

add_library(vast::frontend_tool ALIAS vast_frontend_tool)

add_executable(vast-front
  driver.cpp
  cc1.cpp
)

target_link_libraries(vast-front
    PRIVATE
        ${DIALECT_LIBS}
        ${CONVERSION_LIBS}

        MLIRHighLevel
        MLIRHighLevelTransforms

        MLIRIR
        MLIRParser
        MLIRPass
        MLIRSupport

        LLVM
        LLVMSupport

        FromSourceParser

        clangAST
        clangBasic
        clangCodeGen
        clangDriver
        clangFrontend
        clangSerialization
        clangTooling

        vast::frontend
        vast::frontend_tool
        vast::settings
)

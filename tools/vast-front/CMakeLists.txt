#
# VAST Frontend Praser
#

get_property(DIALECT_LIBS GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(CONVERSION_LIBS GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

add_library(vast_frontend_tool
  compiler_invocation.cpp
)

target_link_libraries(vast_frontend_tool
  PUBLIC
    clangBasic
    clangDriver
    clangFrontend

    ${LLVM_LIBS}

    vast::codegen
    vast::frontend
    vast::target_llvmir
  PRIVATE
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

    FromSourceParser

    clangAST
    clangCodeGen
    clangFrontend
    clangSerialization
    clangTooling

    vast::frontend_tool
    vast::settings
)

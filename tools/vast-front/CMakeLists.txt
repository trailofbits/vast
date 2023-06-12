#
# VAST Frontend Praser
#

get_property(DIALECT_LIBS GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(CONVERSION_LIBS GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

add_library(vast_frontend_tool
  compiler_invocation.cpp
)


mk_clang_libs(frontend_tool_clang_libs clangBasic clangDriver clangFrontend)
target_link_libraries(vast_frontend_tool
  PUBLIC
    ${frontend_tool_clang_libs}

    ${LLVM_LIBS}

    vast::codegen
    vast::frontend
  PRIVATE
    vast::settings
)

add_library(vast::frontend_tool ALIAS vast_frontend_tool)

add_executable(vast-front
  driver.cpp
  cc1.cpp
)

mk_clang_libs(front_clang_libs clangAST clangFrontend clangSerialization clangTooling)
target_link_libraries(vast-front
  PRIVATE
    ${DIALECT_LIBS}

    vast::conversion::all

    MLIRHighLevel
    MLIRHighLevelTransforms

    MLIRIR
    MLIRParser
    MLIRPass
    MLIRSupport

    FromSourceParser

    ${front_clang_libs}

    vast::frontend_tool
    vast::settings
)

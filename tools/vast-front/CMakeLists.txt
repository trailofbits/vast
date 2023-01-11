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
        clangBasic
        clangFrontend
        clangSerialization
        clangTooling
)

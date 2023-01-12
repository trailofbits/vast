#
# VAST Frontend Praser
#
set(LLVM_LINK_COMPONENTS Core Support AsmParser)

get_property(DIALECT_LIBS GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(CONVERSION_LIBS GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

add_executable(vast-cc vast-cc.cpp)
llvm_update_compile_flags(vast-cc)

target_link_libraries(vast-cc
    PRIVATE
        ${DIALECT_LIBS}
        ${CONVERSION_LIBS}

        MLIROptLib
        MLIRHighLevel
        MLIRHighLevelTransforms

        MLIRIR
        MLIRParser
        MLIRPass
        MLIRSupport

        FromSourceParser

        vast::frontend
)

mlir_check_link_libraries(vast-cc)

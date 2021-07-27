#
# VAST MLIR Optimizer
#
get_property(DIALECT_LIBS GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(CONVERSION_LIBS GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

add_llvm_executable(vast-opt vast-opt.cpp)
llvm_update_compile_flags(vast-opt)

target_link_libraries(vast-opt
    PRIVATE
        ${DIALECT_LIBS}
        ${CONVERSION_LIBS}
        MLIROptLib
        MLIRVast
)

mlir_check_all_link_libraries(vast-opt)
#
# VAST MUTATE Tool
#
set(LLVM_LINK_COMPONENTS Core Support AsmParser)

get_property(DIALECT_LIBS GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(CONVERSION_LIBS GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

add_llvm_executable(vast-mutate
  vast-mutate.cpp
  mutator.cpp
  mutatorUtil.cpp
)
llvm_update_compile_flags(vast-mutate)

#target_compile_options(vast-mutate PRIVATE -stdlib=libc++)

target_link_libraries(vast-mutate
    PRIVATE
        ${DIALECT_LIBS}
        ${CONVERSION_LIBS}

        MLIROptLib
        MLIRHighLevel
        MLIRHighLevelTransforms

        MLIRIR
        MLIRParser
        MLIRPass
        MLIRSPIRV
        MLIRTranslation
        MLIRSupport

        FromSourceParser
)

mlir_check_all_link_libraries(vast-mutate)


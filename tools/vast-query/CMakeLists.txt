#
# VAST MLIR Query
#
get_property(DIALECT_LIBS GLOBAL PROPERTY MLIR_DIALECT_LIBS)

add_llvm_executable(vast-query vast-query.cpp)
llvm_update_compile_flags(vast-query)

target_link_libraries(vast-query
    PRIVATE
        ${DIALECT_LIBS}
        MLIRHighLevel
)

mlir_check_all_link_libraries(vast-query)

if (NOT LLVM_ENABLE_RTTI)
  set_target_properties(vast-query PROPERTIES COMPILE_FLAGS "-fno-rtti")
endif()

#
# VAST REPL Tool
#
get_property(DIALECT_LIBS GLOBAL PROPERTY MLIR_DIALECT_LIBS)

add_llvm_executable(vast-repl
  vast-repl.cpp
  codegen.cpp
)
llvm_update_compile_flags(vast-repl)

target_link_libraries(vast-repl
    PRIVATE
      vast_translation_api

      ${DIALECT_LIBS}
      MLIRHighLevel

      clangAST
      clangFrontend
      clangTooling
)

mlir_check_all_link_libraries(vast-repl)

if (NOT LLVM_ENABLE_RTTI)
  set_target_properties(vast-repl PROPERTIES COMPILE_FLAGS "-fno-rtti")
endif()

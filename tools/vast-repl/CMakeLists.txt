#
# VAST REPL Tool
#
get_property(DIALECT_LIBS GLOBAL PROPERTY MLIR_DIALECT_LIBS)

add_executable(vast-repl
  vast-repl.cpp
  codegen.cpp
  command.cpp
)

target_link_libraries(vast-repl
    PRIVATE
      vast::translation_api
      vast::settings

      ${DIALECT_LIBS}
      MLIRHighLevel

      clangAST
      clangFrontend
      clangSerialization
      clangTooling
)

mlir_check_all_link_libraries(vast-repl)

if (NOT LLVM_ENABLE_RTTI)
  set_target_properties(vast-repl PROPERTIES COMPILE_FLAGS "-fno-rtti")
endif()

add_library( vast_translation_api STATIC
  CodeGenStmtVisitor.cpp
  CodeGenTypeVisitor.cpp
  DataLayout.cpp
  CodeGen.cpp
)

target_link_libraries( vast_translation_api
    PRIVATE
        clangAST
        clangASTMatchers
        clangBasic

        MLIRHighLevel

        vast_settings
)

if (NOT LLVM_ENABLE_RTTI)
  set_target_properties(vast_translation_api PROPERTIES COMPILE_FLAGS "-fno-rtti")
endif()

add_library( FromSourceParser
  FromSource.cpp
)

target_link_libraries( FromSourceParser
    PRIVATE
        clangAST
        clangASTMatchers
        clangBasic
        clangFrontend
        clangSerialization
        clangTooling

        vast_settings
        vast_translation_api
)

if (NOT LLVM_ENABLE_RTTI)
  set_target_properties(FromSourceParser PROPERTIES COMPILE_FLAGS "-fno-rtti")
endif()


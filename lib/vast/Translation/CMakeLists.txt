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

        MLIRMeta
        MLIRHighLevel
        MLIRSupport

        vast::settings
)

if (NOT LLVM_ENABLE_RTTI)
  set_target_properties(vast_translation_api PROPERTIES COMPILE_FLAGS "-fno-rtti")
endif()

add_library(vast::translation_api ALIAS vast_translation_api)


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

        MLIRSupport

        vast::settings
        vast::translation_api
)

if (NOT LLVM_ENABLE_RTTI)
  set_target_properties(FromSourceParser PROPERTIES COMPILE_FLAGS "-fno-rtti")
endif()


add_library( vast_translation_api STATIC
  HighLevelTypeConverter.cpp
  HighLevelVisitor.cpp
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


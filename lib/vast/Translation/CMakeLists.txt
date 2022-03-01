add_library( vast_translation_api STATIC
  HighLevelTypeConverter.cpp
  HighLevelVisitor.cpp
)

target_link_libraries( vast_translation_api
    PRIVATE
        clangAST
        clangASTMatchers
        clangBasic

        MLIRHighLevel

        vast_settings
)

set_target_properties(vast_translation_api
  PROPERTIES PUBLIC_HEADER "${VAST_TRANSLATION_HEADERS}"
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

option(VAST_TRANSLATION_INSTALL "Generate the install target." ${GAP_MASTER_PROJECT})

install(TARGETS vast_translation_api
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    PUBLIC_HEADER DESTINATION include
)

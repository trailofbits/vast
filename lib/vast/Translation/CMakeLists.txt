add_library( vast_translation_api STATIC
  CodeGenScope.cpp
  CodeGenStmtVisitor.cpp
  CodeGenTypeVisitor.cpp
  DataLayout.cpp
  CodeGen.cpp
  CodeGenDriver.cpp
  CodeGenTypeDriver.cpp
  CodeGenFunction.cpp
  Mangler.cpp
)

target_link_libraries( vast_translation_api
  PRIVATE
    clangAST
    clangASTMatchers
    clangBasic

    VASTMeta
    VASTHighLevel
    MLIRSupport

    vast::settings
    vast::codegen
)

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

# TODO: Move to vast_add_library
set_property(GLOBAL APPEND PROPERTY VAST_INSTALL_TARGETS
    vast_translation_api
)



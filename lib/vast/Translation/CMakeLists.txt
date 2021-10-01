add_library( FromSourceParser
    Types.cpp
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

        MLIRHighLevel

        vast_settings
)

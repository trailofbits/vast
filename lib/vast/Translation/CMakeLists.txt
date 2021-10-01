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


if (NOT LLVM_ENABLE_RTTI)
  set_target_properties(FromSourceParser PROPERTIES COMPILE_FLAGS "-fno-rtti")
endif()

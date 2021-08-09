add_library( FromSourceParser
    Types.cpp
    FromSource.cpp
)

target_link_libraries(FromSourceParser PRIVATE
    pasta_compiler
)
add_library( FromSourceParser
    FromSource.cpp
)

target_link_libraries(FromSourceParser PRIVATE cppast)
add_subdirectory(PDLL)
add_subdirectory(Common)
add_subdirectory(FromHL)

add_library( VASTConversions INTERFACE )

add_dependencies( VASTConversions
    VASTConversionPassIncGen
)

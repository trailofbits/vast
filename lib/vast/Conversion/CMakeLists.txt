add_subdirectory(ABI)
add_subdirectory(PDLL)
add_subdirectory(Common)
add_subdirectory(FromHL)
add_subdirectory(Core)

add_library(vast_conversion_all INTERFACE)
target_link_libraries(vast_conversion_all
  INTERFACE
    vast::conversion::abi
    vast::conversion::common
    vast::conversion::core
    vast::conversion::from_hl
)
add_dependencies(vast_conversion_all VASTConversionPassIncGen)
add_library(vast::conversion::all ALIAS vast_conversion_all)

set_property(GLOBAL APPEND PROPERTY VAST_INSTALL_TARGETS
    vast_conversion_all
    vast_conversion_abi
    vast_conversion_common
    vast_conversion_core
    vast_conversion_from_hl
)

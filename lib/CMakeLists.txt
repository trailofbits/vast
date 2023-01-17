add_subdirectory(vast)

add_library( vast_api INTERFACE )

target_link_libraries( vast_api
    INTERFACE
        vast::translation_api
        vast::settings
)

add_library(vast::api ALIAS vast_api)



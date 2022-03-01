add_subdirectory(vast)

add_library( vast_api INTERFACE )

target_link_libraries( vast_api
    INTERFACE
        vast_translation_api
        vast_settings
)

add_library(vast::api ALIAS vast_api)



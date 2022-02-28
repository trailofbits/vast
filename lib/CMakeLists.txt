add_subdirectory(vast)

add_library( vast_api INTERFACE )

target_link_libraries( vast_api
    INTERFACE
        vast_translation_api
)

if (NOT LLVM_ENABLE_RTTI)
  set_target_properties(vast_api PROPERTIES COMPILE_FLAGS "-fno-rtti")
endif()


add_library(vast::api ALIAS vast_api)

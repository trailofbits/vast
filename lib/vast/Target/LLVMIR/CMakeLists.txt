add_library( vast_target_llvmir
    Convert.cpp
)

target_link_libraries( vast_target_llvmir
    PRIVATE
        MLIRSupport

        vast_settings
)

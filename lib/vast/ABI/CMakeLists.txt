add_library( vast_abi STATIC
  ABI.cpp
)

target_link_libraries( vast_abi
    PRIVATE
        MLIRMeta
        MLIRHighLevel
        MLIRSupport

        vast_settings
)

if (NOT LLVM_ENABLE_RTTI)
  set_target_properties(vast_abi PROPERTIES COMPILE_FLAGS "-fno-rtti")
endif()

# TODO: Move to vast_add_library
set_property(TARGET vast APPEND PROPERTY VAST_INSTALL_TARGETS vast_abi)

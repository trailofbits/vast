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

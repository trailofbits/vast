add_library( vast_target_llvmir
    Convert.cpp
)

target_link_libraries( vast_target_llvmir
    PRIVATE
        MLIRSupport

        vast_settings
)

if (NOT LLVM_ENABLE_RTTI)
  set_target_properties(vast_target_llvmir PROPERTIES COMPILE_FLAGS "-fno-rtti")
endif()

if (VAST_BUILD_DIALECTS)
    add_subdirectory(Dialect)
    add_subdirectory(Interfaces)
endif()

if (VAST_BUILD_CONVERSIONS)
    add_subdirectory(ABI)
    add_subdirectory(Conversion)
    add_subdirectory(Target)
endif()

if (VAST_BUILD_FRONTEND)
    add_subdirectory(CodeGen)
    add_subdirectory(Frontend)
endif()

add_subdirectory(Tower)
add_subdirectory(Util)
add_subdirectory(Analyses)

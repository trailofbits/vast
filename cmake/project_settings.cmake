# This is only executed once; use a macro (and not a function) so that
# everything defined here does not end up in a separate namespace
macro(settings_main)
  # default build type
  if(WIN32)
    set(CMAKE_BUILD_TYPE Release)
  else()
    # Set a default build type if none was specified
    if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
      message( STATUS "Setting build type to 'RelWithDebInfo' as none was specified." )
      set(CMAKE_BUILD_TYPE
          RelWithDebInfo
          CACHE STRING "Choose the type of build." FORCE)
      # Set the possible values of build type for cmake-gui, ccmake
      set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
                  "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
    endif()
  endif()

  message(STATUS "Install prefix: ${CMAKE_INSTALL_PREFIX}")

  # generate a compile commands JSON file.
  set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

  #
  # compiler and linker flags
  #
  option(ENABLE_IPO
    "Enable Interprocedural Optimization, aka Link Time Optimization (LTO)" OFF
  )

  if(ENABLE_IPO)
    include(CheckIPOSupported)
    check_ipo_supported(RESULT result OUTPUT output)
    if(result)
      set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
    else()
      message(WARNING "IPO is not supported: ${output}")
    endif()
  endif()

  # Globally set the required C++ standard
  set(CMAKE_CXX_STANDARD 20)
  set(CMAKE_CXX_EXTENSIONS OFF)

  if(UNIX)
    if(APPLE)
      set(PLATFORM_NAME "macos")
    else()
      set(PLATFORM_NAME "linux")
    endif()

  elseif(WIN32)
    set(PLATFORM_NAME "windows")

  else()
    message("This platform is not officially supported")
  endif()

  set(SETTINGS_CMAKE_ true)
endmacro()

if(NOT DEFINED SETTINGS_CMAKE_)
  settings_main()
endif()

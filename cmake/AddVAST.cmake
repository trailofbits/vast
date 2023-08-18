include_guard()

include(VASTProcessSources)

# Clear out any pre-existing compile_commands file before processing. This
# allows for generating a clean compile_commands on each configure.
file(REMOVE ${CMAKE_BINARY_DIR}/tablegen_compile_commands.yml)

# VAST_TARGET_DEFINITIONS must contain the name of the .td file to process,
# while VAST_TARGET_DEPENDS may contain additional file dependencies.
# Extra parameters for `tablegen' may come after `ofn' parameter.
# Adds the name of the generated file to VAST_TABLEGEN_OUTPUT.

# Reimplements LLVM's tablegen function from `TableGen.cmake`
function(vast_tablegen ofn)
    set(VAST_TABLEGEN_OUTPUT
        ${VAST_TABLEGEN_OUTPUT} ${CMAKE_CURRENT_BINARY_DIR}/${ofn}
        PARENT_SCOPE
    )

    cmake_parse_arguments(ARG "" "" "DEPENDS;EXTRA_INCLUDES" ${ARGN})

    if (NOT "${VAST_TABLEGEN_PROJECT}" STREQUAL "")
        set(project ${VAST_TABLEGEN_PROJECT})
    endif()

    # Validate calling context.
    if (NOT VAST_TABLEGEN_EXE)
        message(FATAL_ERROR "VAST_TABLEGEN_EXE not set")
    endif()

    # Use depfile instead of globbing arbitrary *.td(s) for Ninja.
    if (CMAKE_GENERATOR MATCHES "Ninja")
        # Make output path relative to build.ninja, assuming located on
        # ${CMAKE_BINARY_DIR}.
        # CMake emits build targets as relative paths but Ninja doesn't identify
        # absolute path (in *.d) as relative path (in build.ninja)
        # Note that tblgen is executed on ${CMAKE_BINARY_DIR} as working directory.
        file(RELATIVE_PATH ofn_rel ${CMAKE_BINARY_DIR} ${CMAKE_CURRENT_BINARY_DIR}/${ofn})
        set(additional_cmdline
            -o ${ofn_rel}
            -d ${ofn_rel}.d
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
            DEPFILE ${CMAKE_CURRENT_BINARY_DIR}/${ofn}.d
        )
        set(local_tds)
        set(global_tds)
    else()
        file(GLOB local_tds "*.td")
        file(GLOB_RECURSE global_tds "${VAST_MAIN_INCLUDE_DIR}/vast/*.td")
        set(additional_cmdline -o ${CMAKE_CURRENT_BINARY_DIR}/${ofn})
    endif()

    if (IS_ABSOLUTE ${VAST_TARGET_DEFINITIONS})
        set(VAST_TARGET_DEFINITIONS_ABSOLUTE ${VAST_TARGET_DEFINITIONS})
    else()
        set(VAST_TARGET_DEFINITIONS_ABSOLUTE
            ${CMAKE_CURRENT_SOURCE_DIR}/${VAST_TARGET_DEFINITIONS}
        )
    endif()

    # We need both _TABLEGEN_TARGET and _TABLEGEN_EXE in the  DEPENDS list
    # (both the target and the file) to have .inc files rebuilt on
    # a tablegen change, as cmake does not propagate file-level dependencies
    # of custom targets. See the following ticket for more information:
    # https://cmake.org/Bug/view.php?id=15858
    # The dependency on both, the target and the file, produces the same
    # dependency twice in the result file when
    # ("${VAST_TABLEGEN_TARGET}" STREQUAL "${VAST_TABLEGEN_EXE}")
    # but lets us having smaller and cleaner code here.
    get_directory_property(vast_tablegen_includes INCLUDE_DIRECTORIES)
    list(APPEND vast_tablegen_includes ${ARG_EXTRA_INCLUDES})
    list(APPEND vast_tablegen_includes ${VAST_MAIN_INCLUDE_DIR})
    # Filter out empty items before prepending each entry with -I
    list(REMOVE_ITEM vast_tablegen_includes "")
    list(TRANSFORM vast_tablegen_includes PREPEND -I)

    set(tablegen_exe ${VAST_TABLEGEN_EXE})
    set(tablegen_depends ${VAST_TABLEGEN_TARGET} ${tablegen_exe})

    add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${ofn}
        COMMAND ${tablegen_exe} ${ARG_UNPARSED_ARGUMENTS} -I ${CMAKE_CURRENT_SOURCE_DIR}
        ${vast_tablegen_includes}
        ${VAST_TABLEGEN_FLAGS}
        ${VAST_TARGET_DEFINITIONS_ABSOLUTE}
        ${additional_cmdline}
        # The file in VAST_TARGET_DEFINITIONS may be not in the current
        # directory and local_tds may not contain it, so we must
        # explicitly list it here:
        DEPENDS ${ARG_DEPENDS} ${tablegen_depends}
        ${local_tds} ${global_tds}
        ${VAST_TARGET_DEFINITIONS_ABSOLUTE}
        ${VAST_TARGET_DEPENDS}
        COMMENT "Building ${ofn}..."
    )

    # `make clean' must remove all those generated files:
    set_property(DIRECTORY APPEND PROPERTY ADDITIONAL_MAKE_CLEAN_FILES ${ofn})

    set(VAST_TABLEGEN_OUTPUT ${VAST_TABLEGEN_OUTPUT} ${CMAKE_CURRENT_BINARY_DIR}/${ofn} PARENT_SCOPE)
    set_source_files_properties(${CMAKE_CURRENT_BINARY_DIR}/${ofn} PROPERTIES GENERATED 1)
endfunction(vast_tablegen)

function(vast_tablegen_compile_command)
    cmake_parse_arguments(ARG "" "" "DEPENDS;EXTRA_INCLUDES" ${ARGN})
    get_directory_property(vast_tablegen_includes INCLUDE_DIRECTORIES)
    list(APPEND vast_tablegen_includes ${ARG_EXTRA_INCLUDES})

    # Filter out any empty include items.
    list(REMOVE_ITEM vast_tablegen_includes "")

    # Build the absolute path for the current input file.
    if (IS_ABSOLUTE ${VAST_TARGET_DEFINITIONS})
        set(VAST_TARGET_DEFINITIONS_ABSOLUTE ${VAST_TARGET_DEFINITIONS})
    else()
        set(VAST_TARGET_DEFINITIONS_ABSOLUTE ${CMAKE_CURRENT_SOURCE_DIR}/${VAST_TARGET_DEFINITIONS})
    endif()

    # Append the includes used for this file to the tablegen_compile_commands file.
    file(APPEND ${CMAKE_BINARY_DIR}/tablegen_compile_commands.yml
        "--- !FileInfo:\n"
        "  filepath: \"${VAST_TARGET_DEFINITIONS_ABSOLUTE}\"\n"
        "  includes: \"${CMAKE_CURRENT_SOURCE_DIR};${vast_tablegen_includes}\"\n"
    )
endfunction(vast_tablegen_compile_command)

function(add_public_vast_tablegen_target target)
    if(NOT VAST_TABLEGEN_OUTPUT)
        message(FATAL_ERROR "Requires tablegen() definitions as VAST_TABLEGEN_OUTPUT.")
    endif()
    add_custom_target(${target} DEPENDS ${VAST_TABLEGEN_OUTPUT})
    if(VAST_COMMON_DEPENDS)
        add_dependencies(${target} ${VAST_COMMON_DEPENDS})
    endif()
    set_target_properties(${target} PROPERTIES FOLDER "Tablegenning")
    set(VAST_COMMON_DEPENDS ${VAST_COMMON_DEPENDS} ${target} PARENT_SCOPE)
endfunction(add_public_vast_tablegen_target)

function(add_vast_dialect dialect dialect_namespace)
    set(VAST_TARGET_DEFINITIONS ${dialect}.td)
    vast_tablegen_compile_command()
    vast_tablegen(${dialect}.h.inc -gen-op-decls)
    vast_tablegen(${dialect}.cpp.inc -gen-op-defs)
    vast_tablegen(${dialect}Types.h.inc -gen-typedef-decls -typedefs-dialect=${dialect_namespace})
    vast_tablegen(${dialect}Types.cpp.inc -gen-typedef-defs -typedefs-dialect=${dialect_namespace})
    vast_tablegen(${dialect}Dialect.h.inc -gen-dialect-decls -dialect=${dialect_namespace})
    vast_tablegen(${dialect}Dialect.cpp.inc -gen-dialect-defs -dialect=${dialect_namespace})
    add_public_vast_tablegen_target(VAST${dialect}IncGen)
    add_dependencies(vast-headers VAST${dialect}IncGen)
    vast_tablegen(${dialect}Attributes.h.inc -gen-attrdef-decls)
    vast_tablegen(${dialect}Attributes.cpp.inc -gen-attrdef-defs)
    add_public_vast_tablegen_target(VAST${dialect}AttributesIncGen)
    add_dependencies(vast-headers VAST${dialect}AttributesIncGen)
    vast_tablegen(${dialect}Enums.h.inc -gen-enum-decls -dialect=${dialect_namespace})
    vast_tablegen(${dialect}Enums.cpp.inc -gen-enum-defs -dialect=${dialect_namespace})
    add_public_vast_tablegen_target(VAST${dialect}EnumsIncGen)
    add_dependencies(vast-headers VAST${dialect}EnumsIncGen)
endfunction()

function(add_vast_doc doc_filename output_file output_directory command)
  set(VAST_TARGET_DEFINITIONS ${doc_filename}.td)
  vast_tablegen(${output_file}.md ${command} ${ARGN})
  set(GEN_DOC_FILE ${VAST_BINARY_DIR}/docs/${output_directory}${output_file}.md)
  add_custom_command(
          OUTPUT ${GEN_DOC_FILE}
          COMMAND ${CMAKE_COMMAND} -E copy
                  ${CMAKE_CURRENT_BINARY_DIR}/${output_file}.md
                  ${GEN_DOC_FILE}
          DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${output_file}.md)
  add_custom_target(${output_file}DocGen DEPENDS ${GEN_DOC_FILE})
  add_dependencies(vast-doc ${output_file}DocGen)
endfunction(add_vast_doc)

function(add_vast_dialect_with_doc dialect dialect_namespace)
    add_vast_dialect(${dialect} ${dialect_namespace})
    add_vast_doc(${dialect} ${dialect} ${dialect}/ -gen-dialect-doc -dialect=${dialect_namespace})
endfunction(add_vast_dialect_with_doc)

function(add_vast_dialect_conversion_passes dialect)
    set(VAST_TARGET_DEFINITIONS Passes.td)
    vast_tablegen(Passes.h.inc -gen-pass-decls -name ${dialect})
    vast_tablegen(Passes.capi.h.inc -gen-pass-capi-header --prefix ${dialect})
    vast_tablegen(Passes.capi.cpp.inc -gen-pass-capi-impl --prefix ${dialect})
    vast_tablegen_compile_command()
    add_public_vast_tablegen_target(VAST${dialect}TransformsIncGen)
    add_dependencies(vast-headers VAST${dialect}TransformsIncGen)
    add_vast_doc(Passes ${dialect}Passes ./ -gen-pass-doc)
endfunction()

function(add_vast_dialect_with_doc_and_passes dialect dialect_namespace)
    add_vast_dialect_with_doc(${dialect} ${dialect_namespace})
    add_vast_dialect_conversion_passes(${dialect})
endfunction()

function(add_vast_op_interface interface)
  set(VAST_TARGET_DEFINITIONS ${interface}.td)
  vast_tablegen(${interface}.h.inc -gen-op-interface-decls)
  vast_tablegen(${interface}.cpp.inc -gen-op-interface-defs)
  vast_tablegen_compile_command(${interface})
  add_public_vast_tablegen_target(VAST${interface}IncGen)
  add_dependencies(vast-generic-headers VAST${interface}IncGen)
endfunction()

function(add_vast_attr_interface interface)
  set(VAST_TARGET_DEFINITIONS ${interface}.td)
  vast_tablegen(${interface}.h.inc -gen-attr-interface-decls)
  vast_tablegen(${interface}.cpp.inc -gen-attr-interface-defs)
  vast_tablegen_compile_command(${interface})
  add_public_vast_tablegen_target(VAST${interface}IncGen)
  add_dependencies(vast-generic-headers VAST${interface}IncGen)
endfunction()

# Sets ${srcs} to contain the list of additional headers for the target. Extra
# arguments are included into the list of additional headers.
function(set_vast_additional_headers_and_sources)
  set(srcs)
  if(MSVC_IDE OR XCODE)
    # Add public headers
    file(RELATIVE_PATH lib_path
      ${VAST_SOURCE_DIR}/lib/
      ${CMAKE_CURRENT_SOURCE_DIR}
    )
    if(NOT lib_path MATCHES "^[.][.]")
      file( GLOB_RECURSE headers
        ${VAST_SOURCE_DIR}/include/vast/${lib_path}/*.h
        ${VAST_SOURCE_DIR}/include/vast/${lib_path}/*.hpp
        ${VAST_SOURCE_DIR}/include/vast/${lib_path}/*.def
      )
      set_source_files_properties(${headers} PROPERTIES HEADER_FILE_ONLY ON)

      file( GLOB_RECURSE tds
        ${VAST_SOURCE_DIR}/include/vast/${lib_path}/*.td
      )
      source_group("TableGen descriptions" FILES ${tds})
      set_source_files_properties(${tds}} PROPERTIES HEADER_FILE_ONLY ON)

      if(headers OR tds)
        set(srcs ${headers} ${tds})
      endif()
    endif()
  endif(MSVC_IDE OR XCODE)
  if(srcs OR ARGN)
    set(srcs
      ADDITIONAL_HEADERS
      ${srcs}
      ${ARGN} # It may contain unparsed unknown args.
      PARENT_SCOPE
    )
  endif()
endfunction()

# Checks that the LLVM components are not listed in the extra arguments,
# assumed to be coming from the LINK_LIBS variable.
function(check_llvm_components_usage name)
  # LINK_COMPONENTS is necessary to allow libLLVM.so to be properly
  # substituted for individual library dependencies if LLVM_LINK_LLVM_DYLIB
  # Perhaps this should be in llvm_add_library instead?  However, it fails
  # on libclang-cpp.so
  get_property(llvm_component_libs GLOBAL PROPERTY LLVM_COMPONENT_LIBS)
  foreach(lib ${ARGN})
    if(${lib} IN_LIST llvm_component_libs)
      message(SEND_ERROR "${name} specifies LINK_LIBS ${lib}, but LINK_LIBS cannot be used for LLVM libraries.  Please use LINK_COMPONENTS instead.")
    endif()
  endforeach()
endfunction()

# `vast_add_library_impl` reimplements `llvm_add_library`
#
# vast_add_library_impl(name sources...
#   SHARED;STATIC
#     STATIC by default w/o BUILD_SHARED_LIBS.
#     SHARED by default w/  BUILD_SHARED_LIBS.
#   OBJECT
#     Also create an OBJECT library target. Default if STATIC && SHARED.
#   MODULE
#     Target ${name} might not be created on unsupported platforms.
#     Check with "if(TARGET ${name})".
#   DISABLE_VAST_LINK_VAST_DYLIB
#     Do not link this library to libVAST, even if
#     VAST_LINK_VAST_DYLIB is enabled.
#   OUTPUT_NAME name
#     Corresponds to OUTPUT_NAME in target properties.
#   DEPENDS targets...
#     Same semantics as add_dependencies().
#   LINK_COMPONENTS components...
#     Same as the variable LLVM_LINK_COMPONENTS.
#   LINK_LIBS lib_targets...
#     Same semantics as target_link_libraries().
#   ADDITIONAL_HEADERS
#     May specify header files for IDE generators.
#   SONAME
#     Should set SONAME link flags and create symlinks
#   NO_INSTALL_RPATH
#     Suppress default RPATH settings in shared libraries.
#   )
function(vast_add_library_impl name)
  cmake_parse_arguments(ARG
    "MODULE;SHARED;STATIC;OBJECT;DISABLE_VAST_LINK_VAST_DYLIB;SONAME;NO_INSTALL_RPATH"
    "OUTPUT_NAME;ENTITLEMENTS;BUNDLE_PATH"
    "ADDITIONAL_HEADERS;DEPENDS;LINK_COMPONENTS;LINK_LIBS;OBJLIBS"
    ${ARGN}
  )

  list(APPEND VAST_COMMON_DEPENDS ${ARG_DEPENDS})

  # link gap by default
  list(APPEND ARG_LINK_LIBS PRIVATE vast::settings)

  if(ARG_ADDITIONAL_HEADERS)
    # Pass through ADDITIONAL_HEADERS.
    set(ARG_ADDITIONAL_HEADERS ADDITIONAL_HEADERS ${ARG_ADDITIONAL_HEADERS})
  endif()

  if(ARG_OBJLIBS)
    set(ALL_FILES ${ARG_OBJLIBS})
  else()
    vast_process_sources(ALL_FILES ${ARG_UNPARSED_ARGUMENTS} ${ARG_ADDITIONAL_HEADERS})
  endif()

  if(ARG_MODULE)
    if(ARG_SHARED OR ARG_STATIC)
      message(WARNING "MODULE with SHARED|STATIC doesn't make sense.")
    endif()
  else()
    if(BUILD_SHARED_LIBS AND NOT ARG_STATIC)
      set(ARG_SHARED TRUE)
    endif()
    if(NOT ARG_SHARED)
      set(ARG_STATIC TRUE)
    endif()
  endif()

  # Generate objlib
  if((ARG_SHARED AND ARG_STATIC) OR ARG_OBJECT)
    # Generate an obj library for both targets.
    set(obj_name "obj.${name}")
    add_library(${obj_name} OBJECT EXCLUDE_FROM_ALL
      ${ALL_FILES}
    )

    # Do add_dependencies(obj) later due to CMake issue 14747.
    list(APPEND objlibs ${obj_name})

    # Bring in the target include directories from our original target.
    target_include_directories(${obj_name}
      PRIVATE
        $<TARGET_PROPERTY:${name},INCLUDE_DIRECTORIES>
    )

    set_target_properties(${obj_name} PROPERTIES FOLDER "Object Libraries")
    if(ARG_DEPENDS)
      add_dependencies(${obj_name} ${ARG_DEPENDS})
    endif()
    # Treat link libraries like PUBLIC dependencies.  LINK_LIBS might
    # result in generating header files.  Add a dependendency so that
    # the generated header is created before this object library.
    if(ARG_LINK_LIBS)
      cmake_parse_arguments(LINK_LIBS_ARG
        ""
        ""
        "PUBLIC;PRIVATE"
        ${ARG_LINK_LIBS}
      )
      foreach(link_lib ${LINK_LIBS_ARG_PUBLIC})
        if (LLVM_PTHREAD_LIB)
          # Can't specify a dependence on -lpthread
          if(NOT ${link_lib} STREQUAL ${LLVM_PTHREAD_LIB})
            add_dependencies(${obj_name} ${link_lib})
          endif()
        else()
          add_dependencies(${obj_name} ${link_lib})
        endif()
      endforeach()
    endif()
  endif()

  if(ARG_SHARED AND ARG_STATIC)
    # static
    set(name_static "${name}_static")
    if(ARG_OUTPUT_NAME)
      set(output_name OUTPUT_NAME "${ARG_OUTPUT_NAME}")
    endif()
    # DEPENDS has been appended to VAST_COMMON_LIBS.
    vast_add_library(${name_static} STATIC
      ${output_name}
      OBJLIBS ${ALL_FILES} # objlib
      LINK_LIBS ${ARG_LINK_LIBS}
      LINK_COMPONENTS ${ARG_LINK_COMPONENTS}
    )

    # Bring in the target link info from our original target.
    target_link_directories(${name_static} PRIVATE $<TARGET_PROPERTY:${name},LINK_DIRECTORIES>)
    target_link_libraries(${name_static} PRIVATE $<TARGET_PROPERTY:${name},LINK_LIBRARIES>)

    # FIXME: Add name_static to anywhere in TARGET ${name}'s PROPERTY.
    set(ARG_STATIC)
  endif()

  if(ARG_MODULE)
    add_library(${name} MODULE ${ALL_FILES})
  elseif(ARG_SHARED)
    # FIXME: add_windows_version_resource_file(ALL_FILES ${ALL_FILES})
    add_library(${name} SHARED ${ALL_FILES})
  else()
    add_library(${name} STATIC ${ALL_FILES})
  endif()

  target_include_directories(${name}
    PRIVATE
      $<BUILD_INTERFACE:${VAST_SOURCE_DIR}/include>
      $<BUILD_INTERFACE:${VAST_BINARY_DIR}/include>
      $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  )

  if(NOT ARG_NO_INSTALL_RPATH)
    if(ARG_MODULE OR ARG_SHARED)
      message(FATAL_ERROR "Unsupported NO_INSTALL_RPATH argument.")
    endif()
  endif()

  if(DEFINED windows_resource_file)
    message(FATAL_ERROR "Unsupported windows_resource_file.")
  endif()

  set_output_directory(${name} BINARY_DIR ${VAST_RUNTIME_OUTPUT_INTDIR} LIBRARY_DIR ${VAST_LIBRARY_OUTPUT_INTDIR})

  if(ARG_OUTPUT_NAME)
    set_target_properties(${name}
      PROPERTIES
      OUTPUT_NAME ${ARG_OUTPUT_NAME}
    )
  endif()

  if(ARG_MODULE)
    set_target_properties(${name} PROPERTIES
      PREFIX ""
      SUFFIX ${CMAKE_SHARED_MODULE_SUFFIX}
    )
  endif()

  if(ARG_SHARED)
    if(MSVC)
      set_target_properties(${name} PROPERTIES
        PREFIX ""
      )
    endif()

    # Set SOVERSION on shared libraries that lack explicit SONAME
    # specifier, on *nix systems that are not Darwin.
    if(UNIX AND NOT APPLE AND NOT ARG_SONAME)
      set_target_properties(${name}
        PROPERTIES
        # Since 4.0.0, the ABI version is indicated by the major version
        SOVERSION ${VAST_VERSION_MAJOR}${VAST_VERSION_SUFFIX}
        VERSION ${VAST_VERSION_MAJOR}${VAST_VERSION_SUFFIX}
      )
    endif()
  endif()

  if(ARG_MODULE OR ARG_SHARED)
    # Do not add -Dname_EXPORTS to the command-line when building files in this
    # target. Doing so is actively harmful for the modules build because it
    # creates extra module variants, and not useful because we don't use these
    # macros.
    set_target_properties( ${name} PROPERTIES DEFINE_SYMBOL "" )

    if (VAST_EXPORTED_SYMBOL_FILE)
      message(FATAL_ERROR "Unsupported VAST_EXPORTED_SYMBOL_FILE.")
    endif()
  endif()

  if(ARG_SHARED)
    if(NOT APPLE AND ARG_SONAME)
      get_target_property(output_name ${name} OUTPUT_NAME)
      if(${output_name} STREQUAL "output_name-NOTFOUND")
        set(output_name ${name})
      endif()
      set(library_name ${output_name}-${VAST_VERSION_MAJOR}${VAST_VERSION_SUFFIX})
      set(api_name ${output_name}-${VAST_VERSION_MAJOR}.${VAST_VERSION_MINOR}.${VAST_VERSION_PATCH}${VAST_VERSION_SUFFIX})
      set_target_properties(${name} PROPERTIES OUTPUT_NAME ${library_name})
      if (UNIX)
        message(FATAL_ERROR "Unsupported symlink installation.")
      endif()
    endif()
  endif()

  if(ARG_STATIC)
    set(libtype PUBLIC)
  else()
    # We can use PRIVATE since SO knows its dependent libs.
    set(libtype PRIVATE)
  endif()

  target_link_libraries(${name} ${libtype}
      ${ARG_LINK_LIBS}
      ${lib_deps}
      ${llvm_libs}
  )

  if(VAST_COMMON_DEPENDS)
    add_dependencies(${name} ${VAST_COMMON_DEPENDS})
    # Add dependencies also to objlibs.
    # CMake issue 14747 --  add_dependencies() might be ignored to objlib's user.
    foreach(objlib ${objlibs})
      add_dependencies(${objlib} ${VAST_COMMON_DEPENDS})
    endforeach()
  endif()

  if(ARG_SHARED OR ARG_MODULE)
    message(FATAL_ERROR "Unsupported llvm_externalize_debuginfo.")
  endif()
  # clang and newer versions of ninja use high-resolutions timestamps,
  # but older versions of libtool on Darwin don't, so the archive will
  # often get an older timestamp than the last object that was added
  # or updated.  To fix this, we add a custom command to touch archive
  # after it's been built so that ninja won't rebuild it unnecessarily
  # the next time it's run.
  if(ARG_STATIC AND VAST_TOUCH_STATIC_LIBRARIES)
    add_custom_command(TARGET ${name}
      POST_BUILD
      COMMAND touch ${VAST_LIBRARY_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}${name}${CMAKE_STATIC_LIBRARY_SUFFIX}
    )
  endif()
endfunction(vast_add_library_impl)

# Adapts `add_mlir_library`.
#
# Declare a vast library which can be compiled in libVAST.so
# In addition to everything that llvm_add_library accepts, this
# also has the following option:
# EXCLUDE_FROM_LIBVAST
#   Don't include this library in libVAST.so.  This option should be used
#   for test libraries, executable-specific libraries, or rarely used libraries
#   with large dependencies.
# ENABLE_AGGREGATION
#   Forces generation of an OBJECT library, exports additional metadata,
#   and installs additional object files needed to include this as part of an
#   aggregate shared library.
#   TODO: Make this the default for all VAST libraries once all libraries
#   are compatible with building an object library.
function(add_vast_library name)
  cmake_parse_arguments(ARG
  "SHARED;INSTALL_WITH_TOOLCHAIN;EXCLUDE_FROM_LIBVAST;DISABLE_INSTALL;ENABLE_AGGREGATION"
  ""
  "ADDITIONAL_HEADERS;DEPENDS;LINK_COMPONENTS;LINK_LIBS"
  ${ARGN}
  )
  set_vast_additional_headers_and_sources(${ARG_ADDITIONAL_HEADERS})

  set(name VAST${name})

  # Is an object library needed.
  set(NEEDS_OBJECT_LIB OFF)
  if(ARG_ENABLE_AGGREGATION)
    set(NEEDS_OBJECT_LIB ON)
  endif()

  # Determine type of library.
  if(ARG_SHARED)
    set(LIBTYPE SHARED)
  else()
    # llvm_add_library ignores BUILD_SHARED_LIBS if STATIC is explicitly set,
    # so we need to handle it here.
    if(BUILD_SHARED_LIBS)
      set(LIBTYPE SHARED)
    else()
      set(LIBTYPE STATIC)
    endif()
    # Test libraries and such shouldn't be include in libVAST.so
    if(NOT ARG_EXCLUDE_FROM_LIBVAST)
      set(NEEDS_OBJECT_LIB ON)
      set_property(GLOBAL APPEND PROPERTY VAST_STATIC_LIBS ${name})
      set_property(GLOBAL APPEND PROPERTY VAST_LLVM_LINK_COMPONENTS ${ARG_LINK_COMPONENTS})
      set_property(GLOBAL APPEND PROPERTY VAST_LLVM_LINK_COMPONENTS ${LLVM_LINK_COMPONENTS})
    endif()
  endif()

  if(NEEDS_OBJECT_LIB AND NOT XCODE)
    # The Xcode generator doesn't handle object libraries correctly.
    # We special case xcode when building aggregates.
    list(APPEND LIBTYPE OBJECT)
  endif()

  check_llvm_components_usage(${name} ${ARG_LINK_LIBS})

  list(APPEND ARG_DEPENDS vast-generic-headers)
  vast_add_library_impl(${name} ${LIBTYPE} ${ARG_UNPARSED_ARGUMENTS} ${srcs} DEPENDS ${ARG_DEPENDS} LINK_COMPONENTS ${ARG_LINK_COMPONENTS} LINK_LIBS ${ARG_LINK_LIBS})

  if(TARGET ${name})
    target_link_libraries(${name} INTERFACE ${VAST_COMMON_LIBS})
    if(NOT ARG_DISABLE_INSTALL)
      add_vast_library_install(${name})
    endif()
  else()
    # Add empty "phony" target
    add_custom_target(${name})
  endif()
  set_target_properties(${name} PROPERTIES FOLDER "VAST libraries")

  # Setup aggregate.
  if(ARG_ENABLE_AGGREGATION)
    # Compute and store the properties needed to build aggregates.
    set(AGGREGATE_OBJECTS)
    set(AGGREGATE_OBJECT_LIB)
    set(AGGREGATE_DEPS)
    if(XCODE)
      # XCode has limited support for object libraries. Instead, add dep flags
      # that force the entire library to be embedded.
      list(APPEND AGGREGATE_DEPS "-force_load" "${name}")
    else()
      list(APPEND AGGREGATE_OBJECTS "$<TARGET_OBJECTS:obj.${name}>")
      list(APPEND AGGREGATE_OBJECT_LIB "obj.${name}")
    endif()

    # For each declared dependency, transform it into a generator expression
    # which excludes it if the ultimate link target is excluding the library.
    set(NEW_LINK_LIBRARIES)
    get_target_property(CURRENT_LINK_LIBRARIES  ${name} LINK_LIBRARIES)
    get_vast_filtered_link_libraries(NEW_LINK_LIBRARIES ${CURRENT_LINK_LIBRARIES})
    set_target_properties(${name} PROPERTIES LINK_LIBRARIES "${NEW_LINK_LIBRARIES}")
    list(APPEND AGGREGATE_DEPS ${NEW_LINK_LIBRARIES})
    set_target_properties(${name} PROPERTIES
      EXPORT_PROPERTIES "VAST_AGGREGATE_OBJECT_LIB_IMPORTED;VAST_AGGREGATE_DEP_LIBS_IMPORTED"
      VAST_AGGREGATE_OBJECTS "${AGGREGATE_OBJECTS}"
      VAST_AGGREGATE_DEPS "${AGGREGATE_DEPS}"
      VAST_AGGREGATE_OBJECT_LIB_IMPORTED "${AGGREGATE_OBJECT_LIB}"
      VAST_AGGREGATE_DEP_LIBS_IMPORTED "${CURRENT_LINK_LIBRARIES}"
    )

    # In order for out-of-tree projects to build aggregates of this library,
    # we need to install the OBJECT library.
    if(VAST_INSTALL_AGGREGATE_OBJECTS AND NOT ARG_DISABLE_INSTALL)
      add_vast_library_install(obj.${name})
    endif()
  endif()
endfunction(add_vast_library)

# Declare the library associated with a dialect.
function(add_vast_dialect_library name)
    set_property(GLOBAL APPEND PROPERTY VAST_DIALECT_LIBS VAST${name})
    add_vast_library(${ARGV}
      DEPENDS vast-headers
      LINK_LIBS PRIVATE VASTUtil
    )
endfunction(add_vast_dialect_library)

# Declare the library associated with a conversion.
function(add_vast_conversion_library name)
    set_property(GLOBAL APPEND PROPERTY VAST_CONVERSION_LIBS VAST${name})
    add_vast_library(${ARGV}
      DEPENDS vast-headers
      LINK_LIBS PRIVATE VASTUtil
    )
endfunction(add_vast_conversion_library)

# Declare the library associated with an extension.
function(add_vast_extension_library name)
    set_property(GLOBAL APPEND PROPERTY VAST_EXTENSION_LIBS VAST${name})
    add_vast_library(${ARGV}
      DEPENDS vast-headers
      LINK_LIBS PRIVATE VASTUtil
    )
endfunction(add_vast_extension_library)

# Declare the library associated with a translation.
function(add_vast_translation_library name)
    set_property(GLOBAL APPEND PROPERTY VAST_TRANSLATION_LIBS VAST${name})
    add_vast_library(${ARGV}
      DEPENDS vast-headers
      LINK_LIBS PRIVATE VASTUtil
    )
endfunction(add_vast_translation_library)

function(add_vast_interface_library name)
  set_property(GLOBAL APPEND PROPERTY VAST_INTERFACE_LIBS VAST${name})
  add_vast_library(${ARGV}
    DEPENDS vast-headers
    LINK_LIBS PRIVATE VASTUtil
  )
endfunction(add_vast_interface_library)

# Adds an VAST library target for installation.
# This is usually done as part of add_vast_library but is broken out for cases
# where non-standard library builds can be installed.
function(add_vast_library_install name)
  get_vast_target_export_arg(${name} VAST export_to_vast_targets UMBRELLA vast-libraries)
  install(TARGETS ${name}
    COMPONENT ${name}
    ${export_to_vast_targets}
    LIBRARY DESTINATION lib${VAST_LIBDIR_SUFFIX}
    ARCHIVE DESTINATION lib${VAST_LIBDIR_SUFFIX}
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
    # Note that CMake will create a directory like:
    #   objects-${CMAKE_BUILD_TYPE}/obj.LibName
    # and put object files there.
    OBJECTS DESTINATION lib${VAST_LIBDIR_SUFFIX}
  )

  add_vast_install_targets(install-${name} DEPENDS ${name} COMPONENT ${name})

  set_property(GLOBAL APPEND PROPERTY VAST_ALL_LIBS ${name})
  set_property(GLOBAL APPEND PROPERTY VAST_EXPORTS ${name})
endfunction()

function(add_vast_install_targets target)
  cmake_parse_arguments(ARG "" "COMPONENT;PREFIX;SYMLINK" "DEPENDS" ${ARGN})
  if(ARG_COMPONENT)
    set(component_option -DCMAKE_INSTALL_COMPONENT="${ARG_COMPONENT}")
  endif()
  if(ARG_PREFIX)
    set(prefix_option -DCMAKE_INSTALL_PREFIX="${ARG_PREFIX}")
  endif()

  set(file_dependencies)
  set(target_dependencies)
  foreach(dependency ${ARG_DEPENDS})
    if(TARGET ${dependency})
      list(APPEND target_dependencies ${dependency})
    else()
      list(APPEND file_dependencies ${dependency})
    endif()
  endforeach()

  add_custom_target(${target}
    DEPENDS ${file_dependencies}
    COMMAND "${CMAKE_COMMAND}"
            ${component_option}
            ${prefix_option}
            -P "${CMAKE_BINARY_DIR}/cmake_install.cmake"
    USES_TERMINAL
  )

  set_target_properties(${target} PROPERTIES FOLDER "Component Install Targets")
  add_custom_target(${target}-stripped
    DEPENDS ${file_dependencies}
    COMMAND "${CMAKE_COMMAND}"
            ${component_option}
            ${prefix_option}
            -DCMAKE_INSTALL_DO_STRIP=1
            -P "${CMAKE_BINARY_DIR}/cmake_install.cmake"
    USES_TERMINAL
  )

  set_target_properties(${target}-stripped PROPERTIES FOLDER "Component Install Targets (Stripped)")
  if(target_dependencies)
    add_dependencies(${target} ${target_dependencies})
    add_dependencies(${target}-stripped ${target_dependencies})
  endif()

  if(ARG_SYMLINK)
    add_dependencies(${target} install-${ARG_SYMLINK})
    add_dependencies(${target}-stripped install-${ARG_SYMLINK}-stripped)
  endif()
endfunction()

# Get the EXPORT argument to use for an install command for a target in a
# project. The project export set for is named ${project}Targets. Also set the
# ${PROJECT}_HAS_EXPORTS global property to mark the project as
# having exports.
# - target: The target to get the EXPORT argument for.
# - project: The project to produce the argument for. IMPORTANT: The casing of
#   this argument should match the casing used by the project's Config.cmake
# - export_arg_var The variable with this name is set in the caller's scope to
#   the EXPORT argument for the target for the project.
# - UMBRELLA: The (optional) umbrella target that the target is a part of. For
#   example, all VAST libraries have the umbrella target vast-libraries.
function(get_vast_target_export_arg target project export_arg_var)
  string(TOUPPER "${project}" project_upper)
  set(suffix "Targets")
  set(${export_arg_var} EXPORT ${project}${suffix} PARENT_SCOPE)
  set_property(GLOBAL PROPERTY ${project_upper}_HAS_EXPORTS True)
endfunction()

function(add_vast_executable target)
  cmake_parse_arguments(ARG "" "" "LINK_LIBS" ${ARGN})

  add_executable(${target} ${ARG_UNPARSED_ARGUMENTS})

  get_property(VAST_LIBS GLOBAL PROPERTY VAST_ALL_LIBS)

  target_link_libraries(${target}
    PRIVATE
      ${VAST_LIBS}
      ${MLIR_LIBS}
      ${ARG_LINK_LIBS}
      vast::settings
  )

  install(TARGETS ${target}
    COMPONENT ${target}
    LIBRARY DESTINATION lib${VAST_LIBDIR_SUFFIX}
    ARCHIVE DESTINATION lib${VAST_LIBDIR_SUFFIX}
    RUNTIME DESTINATION ${VAST_TOOLS_INSTALL_DIR}
    OBJECTS DESTINATION lib${VAST_LIBDIR_SUFFIX}
  )

  add_vast_install_targets(install-${target} DEPENDS ${target} COMPONENT ${target})

  vast_check_link_libraries(${target})
endfunction(add_vast_executable)

# Verification tools to aid debugging.
function(vast_check_link_libraries name)
  if(TARGET ${name})
    get_target_property(type ${name} TYPE)
    if (${type} STREQUAL "INTERFACE_LIBRARY")
      get_target_property(libs ${name} INTERFACE_LINK_LIBRARIES)
    else()
      get_target_property(libs ${name} LINK_LIBRARIES)
    endif()
    # message("${name} libs are: ${libs}")
    set(linking_llvm FALSE)
    foreach(lib ${libs})
      if(lib)
        if(${lib} MATCHES "^LLVM$")
          set(linking_llvm TRUE)
        endif()
        if((${lib} MATCHES "^LLVM.+") AND ${linking_llvm})
          # This will almost always cause execution problems, since the
          # same symbol might be loaded from 2 separate libraries.  This
          # often comes from referring to an LLVM library target
          # explicitly in target_link_libraries()
          message(WARNING "WARNING: ${name} links LLVM and ${lib}!")
        endif()
      endif()
    endforeach()
  endif()
endfunction(vast_check_link_libraries)

function(vast_check_all_link_libraries name)
  vast_check_link_libraries(${name})
  if(TARGET ${name})
    get_target_property(libs ${name} LINK_LIBRARIES)
    # message("${name} libs are: ${libs}")
    foreach(lib ${libs})
      vast_check_link_libraries(${lib})
    endforeach()
  endif()
endfunction(vast_check_all_link_libraries)

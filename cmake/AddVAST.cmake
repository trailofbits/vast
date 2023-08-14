include_guard()

# Clear out any pre-existing compile_commands file before processing. This
# allows for generating a clean compile_commands on each configure.
file(REMOVE ${CMAKE_BINARY_DIR}/tablegen_compile_commands.yml)

# VAST_TARGET_DEFINITIONS must contain the name of the .td file to process,
# while VAST_TARGET_DEPENDS may contain additional file dependencies.
# Extra parameters for `tablegen' may come after `ofn' parameter.
# Adds the name of the generated file to VAST_TABLEGEN_OUTPUT.

# Reimplements LLVM's tablegen function from `TableGen.cmake`
function(vast_tablegen_impl project ofn)
    cmake_parse_arguments(ARG "" "" "DEPENDS;EXTRA_INCLUDES" ${ARGN})

    # Override ${project} with ${project}_TABLEGEN_PROJECT
    if (NOT "${${project}_TABLEGEN_PROJECT}" STREQUAL "")
        set(project ${${project}_TABLEGEN_PROJECT})
    endif()

    # Validate calling context.
    if (NOT ${project}_TABLEGEN_EXE)
        message(FATAL_ERROR "${project}_TABLEGEN_EXE not set")
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
    # ("${${project}_TABLEGEN_TARGET}" STREQUAL "${${project}_TABLEGEN_EXE}")
    # but lets us having smaller and cleaner code here.
    get_directory_property(vast_tablegen_includes INCLUDE_DIRECTORIES)
    list(APPEND vast_tablegen_includes ${ARG_EXTRA_INCLUDES})
    # Filter out empty items before prepending each entry with -I
    list(REMOVE_ITEM vast_tablegen_includes "")
    list(TRANSFORM vast_tablegen_includes PREPEND -I)

    set(tablegen_exe ${${project}_TABLEGEN_EXE})
    set(tablegen_depends ${${project}_TABLEGEN_TARGET} ${tablegen_exe})

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
endfunction()

# Reimplements LLVM's mlir_tablegen function from `AddMLIR.cmake`
function(vast_tablegen ofn)
    vast_tablegen_impl(VAST ${ARGV})
    set(TABLEGEN_OUTPUT
        ${TABLEGEN_OUTPUT} ${CMAKE_CURRENT_BINARY_DIR}/${ofn}
        PARENT_SCOPE
    )

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
endfunction()

function(add_vast_dialect dialect dialect_namespace)
    set(VAST_TARGET_DEFINITIONS ${dialect}.td)
    vast_tablegen(${dialect}.h.inc -gen-op-decls)
    vast_tablegen(${dialect}.cpp.inc -gen-op-defs)
    vast_tablegen(${dialect}Types.h.inc -gen-typedef-decls -typedefs-dialect=${dialect_namespace})
    vast_tablegen(${dialect}Types.cpp.inc -gen-typedef-defs -typedefs-dialect=${dialect_namespace})
    vast_tablegen(${dialect}Dialect.h.inc -gen-dialect-decls -dialect=${dialect_namespace})
    vast_tablegen(${dialect}Dialect.cpp.inc -gen-dialect-defs -dialect=${dialect_namespace})
    add_public_tablegen_target(VAST${dialect}IncGen)
    add_dependencies(vast-headers VAST${dialect}IncGen)
    vast_tablegen(${dialect}Attributes.h.inc -gen-attrdef-decls)
    vast_tablegen(${dialect}Attributes.cpp.inc -gen-attrdef-defs)
    add_public_tablegen_target(VAST${dialect}AttributesIncGen)
    add_dependencies(vast-headers VAST${dialect}AttributesIncGen)
endfunction()

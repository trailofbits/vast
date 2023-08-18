# Copyright (c) 2022-present, Trail of Bits, Inc.

if (ENABLE_PDLL_CONVERSIONS)

    set(VAST_PDLL_TABLEGEN_EXE ${MLIR_INSTALL_PREFIX}/bin/${MLIR_PDLL_TABLEGEN_EXE})

    # Clear out any pre-existing compile_commands file before processing. This
    # allows for generating a clean compile_commands on each configure.
    file(REMOVE ${CMAKE_BINARY_DIR}/pdll_compile_commands.yml)

    # Declare a helper function/copy of tablegen rule for using tablegen without
    # additional tblgen specific flags when invoking PDLL generator.
    function(vast_pdll_tablegen out)
        cmake_parse_arguments(ARG "" "" "DEPENDS;EXTRA_INCLUDES" ${ARGN})

        # Use depfile instead of globbing arbitrary *.td(s) for Ninja.
        if(CMAKE_GENERATOR MATCHES "Ninja")
            # Make output path relative to build.ninja, assuming located on
            # ${CMAKE_BINARY_DIR}.
            # CMake emits build targets as relative paths but Ninja doesn't identify
            # absolute path (in *.d) as relative path (in build.ninja)
            # Note that tblgen is executed on ${CMAKE_BINARY_DIR} as working directory.
            file(RELATIVE_PATH out_rel ${CMAKE_BINARY_DIR} ${CMAKE_CURRENT_BINARY_DIR}/${out})
            set(additional_cmdline -o ${out_rel} -d ${out_rel}.d
                WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
                DEPFILE ${CMAKE_CURRENT_BINARY_DIR}/${out}.d
            )

            set(local_tds)
            set(global_tds)
        else()
            file(GLOB local_tds "*.td")
            file(GLOB_RECURSE global_tds "${LLVM_MAIN_INCLUDE_DIR}/llvm/*.td")
            set(additional_cmdline -o ${CMAKE_CURRENT_BINARY_DIR}/${out})
        endif()

        if (IS_ABSOLUTE ${LLVM_TARGET_DEFINITIONS})
            set(LLVM_TARGET_DEFINITIONS_ABSOLUTE ${LLVM_TARGET_DEFINITIONS})
        else()
            set(LLVM_TARGET_DEFINITIONS_ABSOLUTE
            ${CMAKE_CURRENT_SOURCE_DIR}/${LLVM_TARGET_DEFINITIONS})
        endif()

        if (CMAKE_GENERATOR MATCHES "Visual Studio")
            # Visual Studio has problems with llvm-tblgen's native --write-if-changed
            # behavior. Since it doesn't do restat optimizations anyway, just don't
            # pass --write-if-changed there.
            set(tblgen_change_flag)
        else()
            set(tblgen_change_flag "--write-if-changed")
        endif()

        get_directory_property(tblgen_includes INCLUDE_DIRECTORIES)
        list(APPEND tblgen_includes ${ARG_EXTRA_INCLUDES})
        # Filter out empty items before prepending each entry with -I
        list(REMOVE_ITEM tblgen_includes "")
        list(TRANSFORM tblgen_includes PREPEND -I)

        add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${out}
            COMMAND ${VAST_PDLL_TABLEGEN_EXE} ${ARG_UNPARSED_ARGUMENTS} -x=cpp -I ${CMAKE_CURRENT_SOURCE_DIR}
                ${tblgen_includes}
                ${LLVM_TABLEGEN_FLAGS}
                ${LLVM_TARGET_DEFINITIONS_ABSOLUTE}
                ${tblgen_change_flag}
                ${additional_cmdline}
            # The file in LLVM_TARGET_DEFINITIONS may be not in the current
            # directory and local_tds may not contain it, so we must
            # explicitly list it here:
            DEPENDS ${ARG_DEPENDS}
                ${local_tds} ${global_tds}
                ${LLVM_TARGET_DEFINITIONS_ABSOLUTE}
                ${LLVM_TARGET_DEPENDS}
            COMMENT "Building ${out}..."
        )

        # `make clean' must remove all those generated files:
        set_property(DIRECTORY APPEND PROPERTY ADDITIONAL_MAKE_CLEAN_FILES ${out})

        set(TABLEGEN_OUTPUT ${TABLEGEN_OUTPUT} ${CMAKE_CURRENT_BINARY_DIR}/${out} PARENT_SCOPE)
        set_source_files_properties(${CMAKE_CURRENT_BINARY_DIR}/${out} PROPERTIES GENERATED 1)
    endfunction()

    function(add_vast_pdll_library target input out)
        set(LLVM_TARGET_DEFINITIONS ${input})

        vast_pdll_tablegen(${out} ${ARGN})
        set(TABLEGEN_OUTPUT ${TABLEGEN_OUTPUT} ${CMAKE_CURRENT_BINARY_DIR}/${out}
            PARENT_SCOPE)

        # Get the current set of include paths for this pdll file.
        cmake_parse_arguments(ARG "" "" "DEPENDS;EXTRA_INCLUDES" ${ARGN})
        get_directory_property(tblgen_includes INCLUDE_DIRECTORIES)
        list(APPEND tblgen_includes ${ARG_EXTRA_INCLUDES})
        # Filter out any empty include items.
        list(REMOVE_ITEM tblgen_includes "")

        # Build the absolute path for the current input file.
        if (IS_ABSOLUTE ${LLVM_TARGET_DEFINITIONS})
            set(LLVM_TARGET_DEFINITIONS_ABSOLUTE ${input})
        else()
            set(LLVM_TARGET_DEFINITIONS_ABSOLUTE ${CMAKE_CURRENT_SOURCE_DIR}/${input})
        endif()

        # Append the includes used for this file to the pdll_compilation_commands
        # file.
        file(APPEND ${CMAKE_BINARY_DIR}/pdll_compile_commands.yml
            "--- !FileInfo:\n"
            "  filepath: \"${LLVM_TARGET_DEFINITIONS_ABSOLUTE}\"\n"
            "  includes: \"${CMAKE_CURRENT_SOURCE_DIR};${tblgen_includes}\"\n"
        )

        add_public_tablegen_target(${target})
    endfunction()


    add_vast_pdll_library( VASTHLToFuncIncGen
        HLToFunc.pdll
        HLToFunc.h.inc

        EXTRA_INCLUDES
            ${PROJECT_SOURCE_DIR}/include/vast
    )

endif() # ENABLE_PDLL_CONVERSIONS

set(VAST_TARGET_DEFINITIONS Passes.td)
vast_tablegen(Passes.h.inc -gen-pass-decls -name Conversion)
vast_tablegen(Passes.capi.h.inc -gen-pass-capi-header --prefix Conversion)
vast_tablegen(Passes.capi.cpp.inc -gen-pass-capi-impl --prefix Conversion)
add_public_vast_tablegen_target(VASTConversionPassIncGen)
add_dependencies(vast-headers VASTConversionPassIncGen)
add_mlir_doc(Passes ConversionPasses ./ -gen-pass-doc)

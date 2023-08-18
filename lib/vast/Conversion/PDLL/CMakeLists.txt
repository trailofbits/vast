# Copyright (c) 2022-present, Trail of Bits, Inc.

if (ENABLE_PDLL_CONVERSIONS)

    add_vast_conversion_library( VASTHLToFunc
        HLToFunc.cpp

        ADDITIONAL_HEADER_DIRS
            ${PROJECT_SOURCE_DIR}/include/vast

        DEPENDS
            VASTHLToFuncIncGen
            VASTConversionPassIncGen

        LINK_COMPONENTS
            Core

        LINK_LIBS PUBLIC
            MLIRIR
            MLIRPass
            MLIRSupport
            MLIRTransformUtils

            VASTMeta
            VASTHighLevel

            vast::settings
    )

endif() # ENABLE_PDLL_CONVERSIONS

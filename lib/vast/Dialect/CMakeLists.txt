# Copyright (c) 2021-present, Trail of Bits, Inc.

add_subdirectory(ABI)
add_subdirectory(HighLevel)
add_subdirectory(LowLevel)
add_subdirectory(Meta)
add_subdirectory(Core)

# TODO: Move to vast_add_dialect
set_property(GLOBAL APPEND PROPERTY VAST_INSTALL_TARGETS
    VASTCore
    VASTMeta
    VASTHighLevel
    MLIRLowLevel
)

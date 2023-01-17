# Copyright (c) 2022-present, Trail of Bits, Inc.

add_library(vast_frontend
    GenAction.cpp
    Options.cpp
)

target_link_libraries(vast_frontend PRIVATE vast::settings)

add_library(vast::frontend ALIAS vast_frontend)

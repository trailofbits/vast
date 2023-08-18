# Copyright (c) 2022-present, Trail of Bits, Inc.

add_vast_library(Frontend
    GenAction.cpp
    Options.cpp

    LINK_LIBS PUBLIC
    VASTCodeGen
)

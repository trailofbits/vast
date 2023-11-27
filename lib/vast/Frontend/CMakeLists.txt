# Copyright (c) 2022-present, Trail of Bits, Inc.

add_vast_library(Frontend
    Action.cpp
    Consumer.cpp
    Options.cpp
    Pipelines.cpp
    Targets.cpp

    LINK_LIBS PUBLIC
    VASTCodeGen
)

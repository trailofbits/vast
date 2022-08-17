// Copyright (c) 2022-present, Trail of Bits, Inc.

#include "vast/Dialect/LowLevel/LowLevelDialect.hpp"
#include "vast/Dialect/LowLevel/LowLevelOps.hpp"

namespace vast::ll
{
    void LowLevelDialect::initialize()
    {
        addOperations<
            #define GET_OP_LIST
            #include "vast/Dialect/LowLevel/LowLevel.cpp.inc"
        >();
    }
} // namespace vast::ll

#include "vast/Dialect/LowLevel/LowLevelDialect.cpp.inc"

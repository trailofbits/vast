// Copyright (c) 2022-present, Trail of Bits, Inc.

#include "vast/Dialect/ABI/ABIDialect.hpp"
#include "vast/Dialect/ABI/ABIOps.hpp"

namespace vast::abi
{
    void ABIDialect::initialize()
    {
        addOperations<
            #define GET_OP_LIST
            #include "vast/Dialect/ABI/ABI.cpp.inc"
        >();
    }
} // namespace vast::abi

#include "vast/Dialect/ABI/ABIDialect.cpp.inc"

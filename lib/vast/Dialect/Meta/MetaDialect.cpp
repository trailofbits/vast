// Copyright (c) 2022-present, Trail of Bits, Inc.

#include "vast/Dialect/Meta/MetaDialect.hpp"

namespace vast::meta
{
    void MetaDialect::initialize() {
        registerTypes();
        registerAttributes();

        addOperations<
            #define GET_OP_LIST
            #include "vast/Dialect/Meta/Meta.cpp.inc"
        >();
    }

} // namespace vast::meta

#include "vast/Dialect/Meta/MetaDialect.cpp.inc"

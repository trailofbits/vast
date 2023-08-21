// Copyright (c) 2023-present, Trail of Bits, Inc.

#include "vast/Util/Common.hpp"
#include "vast/Util/Warnings.hpp"

#include "vast/Dialect/Unsupported/UnsupportedDialect.hpp"
#include "vast/Dialect/Unsupported/UnsupportedOps.hpp"

namespace vast::unsup {
    void UnsupportedDialect::initialize() {
        registerTypes();

        addOperations<
            #define GET_OP_LIST
            #include "vast/Dialect/Unsupported/Unsupported.cpp.inc"
        >();
    }
} // namespace vast::unsup

#include "vast/Dialect/Unsupported/UnsupportedDialect.cpp.inc"

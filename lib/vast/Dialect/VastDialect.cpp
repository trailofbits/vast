// Copyright (c) 2021-present, Trail of Bits, Inc.

#include "vast/Dialect/VastDialect.hpp"
#include "vast/Dialect/VastOps.hpp"

//===----------------------------------------------------------------------===//
// Vast dialect.
//===----------------------------------------------------------------------===//

namespace vast::hl {
    void VastDialect::initialize() {
        addOperations<
            #define GET_OP_LIST
            #include "vast/Dialect/VastOps.cpp.inc"
        >();
    }
} // namespace vast::hl
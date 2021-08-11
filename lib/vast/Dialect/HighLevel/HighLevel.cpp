// Copyright (c) 2021-present, Trail of Bits, Inc.

#include "vast/Dialect/HighLevel/HighLevel.hpp"
#include "vast/Dialect/HighLevel/HighLevelTypes.hpp"

#include <mlir/IR/Builders.h>
#include <mlir/IR/DialectImplementation.h>
#include <llvm/ADT/TypeSwitch.h>

namespace vast::hl
{
    void HighLevelDialect::initialize()
    {
        addOperations<
            #define GET_OP_LIST
            #include "vast/Dialect/HighLevel/HighLevel.cpp.inc"
        >();
        addTypes<
            #define GET_TYPEDEF_LIST
            #include "vast/Dialect/HighLevel/HighLevelTypes.cpp.inc"
        >();
    }

    void HighLevelDialect::registerTypes()
    {
        addTypes< void_type >();
    }

} // namespace vast::hl
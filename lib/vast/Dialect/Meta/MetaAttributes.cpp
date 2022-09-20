// Copyright (c) 2022-present, Trail of Bits, Inc.

#include "vast/Dialect/Meta/MetaAttributes.hpp"

VAST_RELAX_WARNINGS
#include <llvm/ADT/TypeSwitch.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/DialectImplementation.h>
VAST_RELAX_WARNINGS

#define GET_ATTRDEF_CLASSES
#include "vast/Dialect/Meta/MetaAttributes.cpp.inc"

namespace vast::meta
{
    void MetaDialect::registerAttributes()
    {
        addAttributes<
            #define GET_ATTRDEF_LIST
            #include "vast/Dialect/Meta/MetaAttributes.cpp.inc"
        >();
    }

} // namespace vast::hl

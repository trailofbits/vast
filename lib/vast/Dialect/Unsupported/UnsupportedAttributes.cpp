// Copyright (c) 2023-present, Trail of Bits, Inc.

#include "vast/Util/Common.hpp"
#include "vast/Util/Warnings.hpp"

#include "vast/Dialect/Unsupported/UnsupportedDialect.hpp"
#include "vast/Dialect/Unsupported/UnsupportedAttributes.hpp"

VAST_RELAX_WARNINGS
#include <llvm/ADT/TypeSwitch.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/DialectImplementation.h>
VAST_UNRELAX_WARNINGS

#define GET_ATTRDEF_CLASSES
#include "vast/Dialect/Unsupported/UnsupportedAttributes.cpp.inc"

namespace vast::unsup
{
    void UnsupportedDialect::registerAttributes()
    {
        addAttributes<
            #define GET_ATTRDEF_LIST
            #include "vast/Dialect/Unsupported/UnsupportedAttributes.cpp.inc"
        >();
    }

} // namespace vast::unsup

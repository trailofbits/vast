// Copyright (c) 2022-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <llvm/ADT/APSInt.h>
#include <llvm/Support/Locale.h>
#include <mlir/IR/BuiltinAttributes.h>
VAST_UNRELAX_WARNINGS

#include "vast/Dialect/Core/CoreDialect.hpp"
#include "vast/Interfaces/TypeQualifiersInterfaces.hpp"

#include "vast/Util/Common.hpp"
#include "vast/Util/TypeList.hpp"

#include "vast/Dialect/Core/CoreTraits.hpp"

#define GET_ATTRDEF_CLASSES
#include "vast/Dialect/Core/CoreAttributes.h.inc"

namespace vast::core {

    using typed_attrs = util::type_list<
        BooleanAttr, IntegerAttr, FloatAttr, VoidAttr
    >;

} // namespace vast::core

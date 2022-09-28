// Copyright (c) 2021-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

#include "vast/Util/Common.hpp"

#include "vast/Dialect/HighLevel/HighLevelDialect.hpp"
#include "vast/Interfaces/TypedAttrInterface.hpp"
#include "vast/Interfaces/TypeQualifiersInterfaces.hpp"

#include "mlir/IR/BuiltinAttributes.h"

#define GET_ATTRDEF_CLASSES
#include "vast/Dialect/HighLevel/HighLevelAttributes.h.inc"

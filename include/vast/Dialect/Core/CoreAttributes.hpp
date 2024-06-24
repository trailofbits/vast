// Copyright (c) 2022-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <llvm/ADT/APSInt.h>
#include <llvm/Support/Locale.h>
#include <mlir/IR/BuiltinAttributes.h>
VAST_UNRELAX_WARNINGS

#include "vast/Dialect/Core/CoreDialect.hpp"
#include "vast/Dialect/Core/CoreTraits.hpp"

#include "vast/Interfaces/SymbolInterface.hpp"
#include "vast/Interfaces/SymbolRefInterface.hpp"
#include "vast/Interfaces/TypeQualifiersInterfaces.hpp"

#include "vast/Util/Common.hpp"
#include "vast/Util/TypeList.hpp"

#define GET_ATTRDEF_CLASSES
#include "vast/Dialect/Core/CoreAttributes.h.inc"

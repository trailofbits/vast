// Copyright (c) 2022-present, Trail of Bits, Inc.

#pragma once

#include "vast/Dialect/Core/CoreDialect.hpp"
#include "vast/Dialect/Core/CoreTraits.hpp"
#include "vast/Dialect/Core/SymbolTable.hpp"
#include "vast/Dialect/Core/Interfaces/SymbolInterface.hpp"
#include "vast/Dialect/Core/Interfaces/SymbolTableInterface.hpp"

#include "vast/Util/Common.hpp"
#include "vast/Util/TypeList.hpp"

#include <mlir/Interfaces/FunctionInterfaces.h>
#include <mlir/Interfaces/CallInterfaces.h>
#include <mlir/Interfaces/InferTypeOpInterface.h>
#include <mlir/Interfaces/DataLayoutInterfaces.h>

#include <mlir/Dialect/DLTI/Traits.h>


#define GET_OP_CLASSES
#include "vast/Dialect/Core/Core.h.inc"

namespace vast::core {
    using module      = core::ModuleOp;
} // namespace vast::core
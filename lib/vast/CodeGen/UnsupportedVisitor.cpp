// Copyright (c) 2024-present, Trail of Bits, Inc.

#include "vast/CodeGen/UnsupportedVisitor.hpp"

#include "vast/Dialect/Unsupported/UnsupportedDialect.hpp"
#include "vast/Dialect/Unsupported/UnsupportedOps.hpp"
#include "vast/Dialect/Unsupported/UnsupportedTypes.hpp"
#include "vast/Dialect/Unsupported/UnsupportedAttributes.hpp"

namespace vast::cg
{
    mlir_type unsup_type_visitor::make_type(const clang_type *type) {
        return unsup::UnsupportedType::get(&self.mcontext(), type->getTypeClassName());
    }

} // namespace vast::cg

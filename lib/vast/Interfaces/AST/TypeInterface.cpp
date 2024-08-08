// Copyright (c) 2024-present, Trail of Bits, Inc.

#include "vast/Interfaces/AST/TypeInterface.hpp"

//===----------------------------------------------------------------------===//
// ASTNodes Op Interface
//===----------------------------------------------------------------------===//

/// Include the generated interface.
#include "vast/Interfaces/AST/TypeInterface.cpp.inc"

namespace vast::ast {

    TypeInterface *QualTypeInterface::operator->() {
        return nullptr;
    }

} // namespace vast::ast

// Copyright (c) 2021-present, Trail of Bits, Inc.

#include "vast/Dialect/HighLevel/HighLevel.hpp"

#include <mlir/IR/DialectImplementation.h>

#include <llvm/ADT/TypeSwitch.h>
#include <llvm/ADT/StringRef.h>

namespace vast::hl
{
    std::string to_string(integer_kind kind) noexcept
    {
        switch (kind) {
            case integer_kind::Char:     return "char";
            case integer_kind::Short:    return "short";
            case integer_kind::Int:      return "int";
            case integer_kind::Long:     return "long";
            case integer_kind::LongLong: return "longlong";
        }
    }

} // namespace vast::hl

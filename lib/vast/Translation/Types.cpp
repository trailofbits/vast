// Copyright (c) 2021-present, Trail of Bits, Inc.

#include "vast/Translation/Types.hpp"
#include "vast/Dialect/HighLevel/HighLevelTypes.hpp"

#include "clang/AST/Type.h"

namespace vast::hl
{

    mlir::Type TypeConverter::convert(const clang::BuiltinType *ty)
    {
        // TODO(Heno) qualifiers
        if (ty->isVoidType())
            return void_type::get(&ctx);

        llvm_unreachable("unknown builtin type");
    }

} // namseapce vast::hl
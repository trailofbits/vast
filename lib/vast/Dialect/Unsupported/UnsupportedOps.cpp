// Copyright (c) 2022-present, Trail of Bits, Inc.

#include "vast/Util/Common.hpp"
#include "vast/Util/Region.hpp"
#include "vast/Util/Warnings.hpp"

#include "vast/Dialect/Unsupported/UnsupportedDialect.hpp"
#include "vast/Dialect/Unsupported/UnsupportedOps.hpp"

VAST_RELAX_WARNINGS
#include <mlir/IR/Builders.h>
VAST_UNRELAX_WARNINGS

#define GET_OP_CLASSES
#include "vast/Dialect/Unsupported/Unsupported.cpp.inc"

namespace vast::us {

    void UnsupportedDecl::build(
        Builder &bld, State &st, mlir::StringRef name, BuilderCallback fields
    ) {
        st.addAttribute("name", bld.getStringAttr(name));
        InsertionGuard guard(bld);
        build_region(bld, st, fields);
    }

    void UnsupportedExpr::build(
        Builder &bld, State &st, mlir::StringRef name, Type rty,
        std::unique_ptr< Region > &&region
    ) {
        InsertionGuard guard(bld);
        st.addRegion(std::move(region));
        st.addTypes(rty);
        st.addAttribute(getNameAttrName(st.name), bld.getStringAttr(name));
    }

} // namespace vast::us

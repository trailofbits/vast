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

namespace vast::unsup {

    void UnsupportedDecl::build(
        Builder &bld, State &st, llvm::StringRef name, maybe_builder_callback_ref body
    ) {
        st.addAttribute(getNameAttrName(st.name), bld.getStringAttr(name));
        InsertionGuard guard(bld);
        build_region(bld, st, body);
    }

    void UnsupportedStmt::build(
        Builder &bld, State &st, llvm::StringRef name, Type rty,
        const std::vector< builder_callback > &builders
    ) {
        InsertionGuard guard(bld);
        // Optional, add a check if rty exist.
        if (rty) {
            st.addTypes(rty);
        }
        st.addAttribute(getNameAttrName(st.name), bld.getStringAttr(name));
        for (builder_callback_ref child : builders) {
            build_region(bld, st, child);
        }
    }

} // namespace vast::unsup

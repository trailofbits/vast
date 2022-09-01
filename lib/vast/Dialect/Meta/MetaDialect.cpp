// Copyright (c) 2022-present, Trail of Bits, Inc.

#include "vast/Dialect/Meta/MetaDialect.hpp"
#include "vast/Dialect/Meta/MetaAttributes.hpp"

#include "vast/Util/Symbols.hpp"

namespace vast::meta
{
    void MetaDialect::initialize() {
        registerTypes();
        registerAttributes();

        addOperations<
            #define GET_OP_LIST
            #include "vast/Dialect/Meta/Meta.cpp.inc"
        >();
    }

    static constexpr std::string_view identifier_name = "meta_identifier";

    void add_identifier(mlir::Operation *op, identifier_t id) {
        auto ctx = op->getContext();
        auto attr = IdentifierAttr::get(ctx, id);
        op->setAttr(identifier_name, attr);
    }

    void remove_identifier(mlir::Operation *op) {
        op->removeAttr(identifier_name);
    }

    bool has_identifier(mlir::Operation *op, identifier_t id) {
        if (auto attr = op->getAttr(identifier_name)) {
            if (attr.cast< IdentifierAttr >().getValue() == id) {
                return true;
            }
        }

        return false;
    }

    std::vector< mlir::Operation * > get_with_identifier(mlir::Operation *scope, identifier_t id) {
        std::vector< mlir::Operation * > result;
        util::symbols(scope, [&] (auto symbol) {
            if (has_identifier(symbol, id)) {
                result.push_back(symbol);
            }
        });
        return result;
    }

    std::vector< mlir::Operation * > get_with_meta_location(mlir::Operation *scope, IdentifierAttr id) {
        std::vector< mlir::Operation * > result;
        scope->walk([&](mlir::Operation *op) {
            if (auto loc = op->getLoc().dyn_cast< mlir::FusedLoc >()) {
                if (id == loc.getMetadata()) {
                    result.push_back(op);
                }
            }
        });
        return result;
    }

    std::vector< mlir::Operation * > get_with_meta_location(mlir::Operation *scope, identifier_t id) {
        auto ctx = scope->getContext();
        return get_with_meta_location(scope, IdentifierAttr::get(ctx, id));
    }

} // namespace vast::meta

#include "vast/Dialect/Meta/MetaDialect.cpp.inc"

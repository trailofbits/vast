// Copyright (c) 2022-present, Trail of Bits, Inc.

#include "vast/Dialect/Meta/MetaDialect.hpp"
#include "vast/Dialect/Meta/MetaAttributes.hpp"

#include "vast/Dialect/Core/Interfaces/SymbolInterface.hpp"

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

    void add_identifier(operation op, identifier_t id) {
        auto ctx = op->getContext();
        auto attr = IdentifierAttr::get(ctx, id);
        op->setAttr(identifier_name, attr);
    }

    void remove_identifier(operation op) {
        op->removeAttr(identifier_name);
    }

    bool has_identifier(operation op, identifier_t id) {
        if (auto attr = op->getAttr(identifier_name)) {
            return mlir::cast< IdentifierAttr >(attr).getValue() == id;
        }

        return false;
    }

    std::vector< operation  > get_with_identifier(operation scope, identifier_t id) {
        std::vector< operation  > result;
        core::symbols< core::symbol >(scope, [&](operation op) {
            if (has_identifier(op, id)) {
                result.push_back(op);
            }
        });
        return result;
    }

    std::vector< operation  > get_with_meta_location(operation scope, IdentifierAttr id) {
        std::vector< operation  > result;
        scope->walk([&](operation op) {
            if (auto loc = mlir::dyn_cast< mlir::FusedLoc >(op->getLoc())) {
                if (id == loc.getMetadata()) {
                    result.push_back(op);
                }
            }
        });
        return result;
    }

    std::vector< operation  > get_with_meta_location(operation scope, identifier_t id) {
        auto ctx = scope->getContext();
        return get_with_meta_location(scope, IdentifierAttr::get(ctx, id));
    }

} // namespace vast::meta

#include "vast/Dialect/Meta/MetaDialect.cpp.inc"

// Copyright (c) 2021-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <mlir/IR/BuiltinDialect.h>
#include <mlir/Interfaces/FunctionInterfaces.h>
#include <mlir/Interfaces/InferTypeOpInterface.h>
VAST_UNRELAX_WARNINGS

#include <gap/coro/generator.hpp>

#include "vast/Dialect/HighLevel/HighLevelAttributes.hpp"
#include "vast/Dialect/HighLevel/HighLevelDialect.hpp"
#include "vast/Dialect/HighLevel/HighLevelTypes.hpp"

#include "vast/Interfaces/SymbolInterface.hpp"
#include "vast/Interfaces/AggregateTypeDefinitionInterface.hpp"
#include "vast/Interfaces/ClangAST/ASTDeclInterface.hpp"


namespace vast::hl
{
    static constexpr auto external_storage = "is_external";
    static constexpr auto static_storage   = "is_static";
    static constexpr auto auto_storage     = "is_auto";
    static constexpr auto register_storage = "is_register";
    static constexpr auto thread_storage   = "is_thread_local";

    template< typename Self >
    void set_unit_attr(Self &self, std::string_view attr) {
        self->setAttr(attr, mlir::UnitAttr::get(self.getContext()));
    }

    template< typename Self >
    void set_external_storage(Self &self) {
        set_unit_attr(self, external_storage);
    }

    template< typename Self >
    void set_static_storage(Self &self) {
        set_unit_attr(self, static_storage);
    }

    template< typename Self >
    void set_auto_storage(Self &self) {
        set_unit_attr(self, auto_storage);
    }

    template< typename Self >
    void set_register_storage(Self &self) {
        set_unit_attr(self, register_storage);
    }

    template< typename Self >
    void set_thread_local_storage(Self &self) {
        set_unit_attr(self, thread_storage);
    }

    template< typename Self >
    bool has_unit_attr(const Self &self, std::string_view attr) {
        return self->hasAttr(attr);
    }

    template< typename Self >
    bool has_external_storage(const Self &self) {
        return has_unit_attr(self, external_storage);
    }

    template< typename Self >
    bool has_static_storage(const Self &self) {
        return has_unit_attr(self, static_storage);
    }

    template< typename Self >
    bool has_auto_storage(const Self &self) {
        return has_unit_attr(self, auto_storage);
    }

    template< typename Self >
    bool has_register_storage(const Self &self) {
        return has_unit_attr(self, register_storage);
    }

    template< typename Self >
    bool has_thread_local_storage(const Self &self) {
        return has_unit_attr(self, thread_storage);
    }

} // namespace vast::hl

#define GET_OP_CLASSES
#include "vast/Dialect/HighLevel/HighLevel.h.inc"

namespace vast::hl
{
    FuncOp getCallee(CallOp call);
}

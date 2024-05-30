// Copyright (c) 2024, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <llvm/ADT/DenseMap.h>
#include <mlir/IR/OpDefinition.h>
VAST_UNRELAX_WARNINGS

#include "vast/Dialect/Core/CoreTraits.hpp"
#include "vast/Util/Common.hpp"

namespace vast::core {

    template< typename... SymbolInterfaces >
    struct ShadowingSymbolTable
    {
        template< typename ConcreteType >
        struct Impl : op_trait_base< ConcreteType, Impl >
        {
            using string_attr   = mlir::StringAttr;
            using symbol_ref    = mlir::SymbolRefAttr;

            static logical_result verifyRegionTrait(operation /* op */) {
                return mlir::success();
            }

            operation lookup_symbol(string_attr /* name */) {
                VAST_UNIMPLEMENTED;
            }

            template< typename T >
            T lookup_symbol(string_attr name) {
                return mlir::dyn_cast_or_null< T >(lookup_symbol(name));
            }

            operation lookup_symbol(symbol_ref /* symbol */) {
                VAST_UNIMPLEMENTED;
            }

            template< typename T >
            T lookup_symbol(symbol_ref symbol) {
                return dyn_cast_or_null< T >(lookup_symbol(symbol));
            }

            operation lookup_symbol(string_ref /* name */) {
                VAST_UNIMPLEMENTED;
            }

            template< typename T >
            T lookup_symbol(string_ref name) {
                return mlir::dyn_cast_or_null< T >(lookup_symbol(name));
            }
        };
    };

    string_ref symbol_attr_name();

} // namespace vast::core

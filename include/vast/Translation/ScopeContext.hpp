// Copyright (c) 2023, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

#include <gap/core/generator.hpp>

#include <functional>
#include <queue>

namespace vast::cg
{
    template< typename From, typename To >
    struct scoped_table : llvm::ScopedHashTable< From, To >
    {
        using value_type = To;

        using base = llvm::ScopedHashTable< From, To >;
        using base::base;

        using base::count;
        using base::insert;

        logical_result declare(const From &from, const To &to) {
            if (count(from)) {
                return mlir::failure();
            }

            insert(from, to);
            return mlir::success();
        }
    };

    struct scope_context {
        using action_t = std::function< void() >;

        ~scope_context() {
            for (const auto &action : deferred()) {
                action();
            }

            VAST_ASSERT(deferred_codegen_actions.empty());
        }

        gap::generator< action_t > deferred() {
            while (!deferred_codegen_actions.empty()) {
                co_yield deferred_codegen_actions.front();
                deferred_codegen_actions.pop();
            }
        }

        void defer(action_t action) {
            deferred_codegen_actions.emplace(std::move(action));
        }

        std::queue< action_t > deferred_codegen_actions;
    };

    // Refers to block scope §6.2.1 of C standard
    //
    // If the declarator or type specifier that declares the identifier appears
    // inside a block or within the list of parameter declarations in a function
    // definition, the identifier has block scope, which terminates at the end
    // of the associated block.
    struct block_scope : scope_context {

    };


    // refers to function scope §6.2.1 of C standard
    struct function_scope : block_scope {
        // label scope
    };

    // Refers to function prototype scope §6.2.1 of C standard
    //
    // If the declarator or type specifier that declares the identifier appears
    // within the list of parameter declarations in a function prototype (not
    // part of a function definition), the identifier has function prototype
    // scope, which terminates at the end of the function declarator
    struct prototype_scope : scope_context {

    };

    // Refers to file scope §6.2.1 of C standard
    //
    // If the declarator or type specifier that declares the identifier appears
    // outside of any block or list of parameters, the identifier has file
    // scope, which terminates at the end of the translation unit.
    struct module_scope : scope_context {

    };

    // Scope of member names for structures and unions

    struct members_scope : scope_context {

    };

} // namespace vast::cg

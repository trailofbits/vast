// Copyright (c) 2023, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <llvm/ADT/ScopedHashTable.h>
VAST_UNRELAX_WARNINGS

#include <gap/coro/generator.hpp>

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

    using deferred_action_t = std::function< void() >;

    enum class emition_kind {
        immediate, deferred
    };

    struct scope_context {

        scope_context() = default;

        virtual ~scope_context() {
            children.clear();
            commit();
            VAST_ASSERT(!has_deferred_actions());
        }

        scope_context(const scope_context &) = delete;
        scope_context(scope_context &&other)
            : deferred_actions(std::move(other.deferred_actions))
        {}

        scope_context &operator=(const scope_context &) = delete;
        scope_context &operator=(scope_context &&) = default;

        gap::generator< deferred_action_t > deferred() {
            while (!deferred_actions.empty()) {
                co_yield deferred_actions.front();
                deferred_actions.pop_front();
            }
        }

        void defer(deferred_action_t action) {
            deferred_actions.push_back(std::move(action));
        }

        void commit() {
            for (const auto &action : deferred()) {
                action();
            }
        }

        void hook(std::unique_ptr< scope_context > &&scope) {
            children.push_back(std::move(scope));
        }

        bool has_deferred_actions() const { return !deferred_actions.empty(); }

        std::deque< deferred_action_t > deferred_actions;
        std::vector< std::unique_ptr< scope_context > > children;
    };

    // Refers to block scope §6.2.1 of C standard
    //
    // If the declarator or type specifier that declares the identifier appears
    // inside a block or within the list of parameter declarations in a function
    // definition, the identifier has block scope, which terminates at the end
    // of the associated block.
    struct block_scope : scope_context {
        virtual ~block_scope() = default;
    };


    // Refers to function scope §6.2.1 of C standard
    struct function_scope : block_scope {
        virtual ~function_scope() = default;
        // label scope
    };

    // Refers to function prototype scope §6.2.1 of C standard
    //
    // If the declarator or type specifier that declares the identifier appears
    // within the list of parameter declarations in a function prototype (not
    // part of a function definition), the identifier has function prototype
    // scope, which terminates at the end of the function declarator
    struct prototype_scope : scope_context {
        virtual ~prototype_scope() = default;
    };

    // Refers to file scope §6.2.1 of C standard
    //
    // If the declarator or type specifier that declares the identifier appears
    // outside of any block or list of parameters, the identifier has file
    // scope, which terminates at the end of the translation unit.
    struct module_scope : scope_context {
        virtual ~module_scope() = default;
    };

    // Scope of member names for structures and unions

    struct members_scope : scope_context {
        virtual ~members_scope() = default;
    };

} // namespace vast::cg

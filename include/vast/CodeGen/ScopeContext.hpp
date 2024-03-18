// Copyright (c) 2023, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <llvm/ADT/ScopedHashTable.h>
VAST_UNRELAX_WARNINGS

#include "vast/CodeGen/Mangler.hpp"
#include "vast/CodeGen/VisitorView.hpp"

#include <functional>
#include <queue>

namespace vast::cg
{
    using vast_function = vast::hl::FuncOp;

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


    using functions_scope_table = scoped_table< string_ref, vast_function >;


    struct scope_tables
    {
        functions_scope_table functions;
    };


    template< typename From, typename To >
    using symbol_table_scope = llvm::ScopedHashTableScope< From, To >;


    struct scope_context {
        using deferred_task = std::function< void() >;

        explicit scope_context(scope_tables &scopes, scope_context *parent)
            : scopes(&scopes), parent(parent)
        {}

        virtual ~scope_context() { finalize(); }

        void finalize() {
            for (auto &child : children) {
                child->finalize();
            }

            while (!deferred.empty()) {
                deferred.front()();
                deferred.pop_front();
            }
        }

        scope_context(const scope_context &) = delete;
        scope_context(scope_context &&other) noexcept = default;

        scope_context &operator=(const scope_context &) = delete;
        scope_context &operator=(scope_context &&) noexcept = default;

        void declare(vast_function function) {
            scopes->functions.insert(function.getName(), function);
        }

        void hook_child(std::unique_ptr< scope_context > &&child) {
            children.push_back(std::move(child));
        }

        void defer(deferred_task task) {
            deferred.push_back(std::move(task));
        }

        std::deque< deferred_task > deferred;

        scope_tables *scopes = nullptr;
        scope_context *parent = nullptr;
        std::vector< std::unique_ptr< scope_context > > children;
    };

    // Refers to block scope §6.2.1 of C standard
    //
    // If the declarator or type specifier that declares the identifier appears
    // inside a block or within the list of parameter declarations in a function
    // definition, the identifier has block scope, which terminates at the end
    // of the associated block.
    struct block_scope : scope_context {
        using scope_context::scope_context;
        virtual ~block_scope() = default;
    };


    // Refers to function scope §6.2.1 of C standard
    struct function_scope : block_scope {
        using block_scope::block_scope;
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
        using scope_context::scope_context;
        virtual ~prototype_scope() = default;
    };

    // Refers to file scope §6.2.1 of C standard
    //
    // If the declarator or type specifier that declares the identifier appears
    // outside of any block or list of parameters, the identifier has file
    // scope, which terminates at the end of the translation unit.
    struct module_scope : scope_context {
        module_scope(
              scope_tables &scopes
            , scope_context *parent
        )
            : scope_context(scopes, parent)
            , functions(scopes.functions)
        {}

        virtual ~module_scope() = default;

        symbol_table_scope< string_ref, vast_function > functions;
    };

    // Scope of member names for structures and unions

    struct members_scope : scope_context {
        using scope_context::scope_context;
        virtual ~members_scope() = default;
    };

    template< typename context >
    struct scope_generator : context {
        scope_generator(visitor_view visitor, auto &&...args)
            : context(std::forward< decltype(args) >(args)...), visitor(visitor)
        {}

        virtual ~scope_generator() = default;

        template< typename generator_t, typename what_t >
        auto generate_child(what_t &&what) {
            auto child = generate< generator_t >(std::forward< what_t >(what), this, visitor);
            auto result = child->result();
            this->hook_child(std::move(child));
            return result;
        }

        visitor_view visitor;
    };

} // namespace vast::cg

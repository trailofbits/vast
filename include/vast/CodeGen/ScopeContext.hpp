// Copyright (c) 2023, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <llvm/ADT/ScopedHashTable.h>
VAST_UNRELAX_WARNINGS

#include "vast/CodeGen/Mangler.hpp"

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

    // TODO why is this name and not function?
    using funs_scope_table = scoped_table< string_ref, operation >;
    using vars_scope_table = scoped_table< string_ref, mlir_value >;

    struct symbol_tables
    {
        funs_scope_table funs;
        vars_scope_table vars;
    };


    template< typename From, typename To >
    using symbol_table_scope = llvm::ScopedHashTableScope< From, To >;


    struct scope_context {
        using deferred_task = std::function< void() >;

        explicit scope_context(scope_context *parent)
            : symbols(parent->symbols), parent(parent)
        {}

        explicit scope_context(symbol_tables &symbols)
            : symbols(symbols)
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
        scope_context(scope_context &&other) noexcept = delete;

        scope_context &operator=(const scope_context &) = delete;
        scope_context &operator=(scope_context &&) noexcept = delete;

        void declare(vast_function function) {
            symbols.funs.insert(function.getName(), function);
        }

        void declare(string_ref name, mlir_value value) {
            symbols.vars.insert(name, value);
        }


        void declare(hl::VarDeclOp var) {
            declare(var.getName(), var);
        }

        void hook_child(std::unique_ptr< scope_context > child) {
            child->parent = this;
            children.push_back(std::move(child));
        }

        template< typename scope_generator_t >
        scope_generator_t& last_child() {
            return *static_cast< scope_generator_t* >(children.back().get());
        }

        void defer(deferred_task task) {
            deferred.push_back(std::move(task));
        }

        std::deque< deferred_task > deferred;

        // scope contents
        symbol_tables &symbols;

        // links between scopes
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
        explicit block_scope(scope_context *parent)
            : scope_context(parent)
            // , vars(parent->symbols.vars)
        {}

        virtual ~block_scope() = default;

//        symbol_table_scope< clang_var_decl*, mlir_value > vars;
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
        explicit module_scope(symbol_tables &symbols)
            : scope_context(symbols)
            , functions(symbols.funs)
            , globals(symbols.vars)
        {}

        virtual ~module_scope() = default;

        symbol_table_scope< string_ref, operation > functions;
        symbol_table_scope< string_ref, mlir_value > globals;
    };

    // Scope of member names for structures and unions

    struct members_scope : scope_context {
        using scope_context::scope_context;
        virtual ~members_scope() = default;
    };

} // namespace vast::cg

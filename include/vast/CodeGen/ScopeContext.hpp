// Copyright (c) 2023, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <llvm/ADT/ScopedHashTable.h>
VAST_UNRELAX_WARNINGS

#include "vast/CodeGen/Mangler.hpp"
#include "vast/Util/TypeList.hpp"

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
    };

    // TODO why is this name and not function?
    using funs_scope_table     = scoped_table< string_ref, operation >;
    using vars_scope_table     = scoped_table< string_ref, mlir_value >;
    using types_scope_table    = scoped_table< string_ref, operation >;
    using enum_constants_table = scoped_table< string_ref, operation >;

    struct symbol_tables
    {
        funs_scope_table funs;
        vars_scope_table vars;
        types_scope_table types;
        enum_constants_table enum_constants;
    };


    namespace symbol {
        using var_decls  = util::type_list< hl::VarDeclOp >;
        using fun_decls  = util::type_list< hl::FuncOp >;
        using type_decls = util::type_list< hl::TypeDeclOp >;

        template< typename op_t >
        concept var_decl_like = var_decls::contains< op_t >;

        template< typename op_t >
        concept fun_decl_like = fun_decls::contains< op_t >;

        template< typename op_t >
        concept type_decl_like = type_decls::contains< op_t >;

    } // namespace symbol

    struct symbols_view {

        template< typename builder_t >
        auto declare(builder_t &&bld) -> decltype(bld()) {
            return declare(bld());
        }

        explicit symbols_view(symbol_tables &symbols)
            : symbols(symbols)
        {}

        auto declare(hl::FuncOp op) {
            return symbols.funs.insert(op.getName(), op), op;
        }

        auto declare(hl::VarDeclOp op) {
            return symbols.vars.insert(op.getName(), op), op;
        }

        auto declare_function_param(string_ref name, mlir_value value) {
            return symbols.vars.insert(name, value), value;
        }

        auto declare(hl::TypeDeclOp op) {
            return symbols.types.insert(op.getName(), op), op;
        }

        mlir_value lookup_var(string_ref name) const {
            return symbols.vars.lookup(name);
        }

        operation lookup_fun(string_ref name) const {
            return symbols.funs.lookup(name);
        }

        operation lookup_type(string_ref name) const {
            return symbols.types.lookup(name);
        }

        template< typename op_t >
        op_t lookup(string_ref name) {
            return mlir::dyn_cast< op_t >(lookup_impl< op_t >(name));
        }

        symbol_tables &symbols;

      private:
        template< symbol::var_decl_like op_t >
        auto lookup_impl(string_ref name) { return lookup_var(name); }

        template< symbol::fun_decl_like op_t >
        auto lookup_impl(string_ref name) { return lookup_fun(name); }

        template< symbol::type_decl_like op_t >
        auto lookup_impl(string_ref name) { return lookup_type(name); }
    };


    template< typename From, typename To >
    using symbol_table_scope = llvm::ScopedHashTableScope< From, To >;

    struct scope_context : symbols_view {
        using deferred_task = std::function< void() >;

        explicit scope_context(scope_context *parent)
            : symbols_view(parent->symbols), parent(parent)
        {}

        explicit scope_context(symbol_tables &symbols)
            : symbols_view(symbols)
        {}

        virtual ~scope_context() { finalize(); }

        void finalize() {
            while (!deferred.empty()) {
                deferred.front()();
                deferred.pop_front();
            }

            for (auto &child : children) {
                child->finalize();
            }

            children.clear();
        }

        scope_context(const scope_context &) = delete;
        scope_context(scope_context &&other) noexcept = delete;

        scope_context &operator=(const scope_context &) = delete;
        scope_context &operator=(scope_context &&) noexcept = delete;

        template< typename child_scope_type >
        scope_context &mk_child() {
            children.push_back(std::make_unique< child_scope_type >(this));
            return *children.back();
        }

        void defer(deferred_task task) {
            deferred.push_back(std::move(task));
        }

        std::deque< deferred_task > deferred;

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
            , vars(parent->symbols.vars)
        {}

        virtual ~block_scope() = default;

        symbol_table_scope< string_ref, mlir_value > vars;
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
            , types(symbols.types)
            , globals(symbols.vars)
        {}

        virtual ~module_scope() = default;

        symbol_table_scope< string_ref, operation >  functions;
        symbol_table_scope< string_ref, operation >  types;
        symbol_table_scope< string_ref, mlir_value > globals;
    };

    // Scope of member names for structures and unions

    struct members_scope : scope_context {
        using scope_context::scope_context;
        explicit members_scope(scope_context *parent)
            : scope_context(parent)
            , vars(parent->symbols.vars)
        {}

        virtual ~members_scope() = default;

        symbol_table_scope< string_ref, mlir_value > vars;
    };

} // namespace vast::cg

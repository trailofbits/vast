// Copyright (c) 2023, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <llvm/ADT/ScopedHashTable.h>
VAST_UNRELAX_WARNINGS

#include "vast/CodeGen/Mangler.hpp"
#include "vast/Util/TypeList.hpp"
#include "vast/Util/Symbols.hpp"

#include "vast/Interfaces/SymbolInterface.hpp"

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

    using funs_scope_table     = scoped_table< string_ref, operation >;
    using vars_scope_table     = scoped_table< string_ref, mlir_value >;
    using types_scope_table    = scoped_table< string_ref, operation >;
    using members_scope_table  = scoped_table< string_ref, operation >;
    using labels_scope_table   = scoped_table< string_ref, operation >;
    using enum_constants_table = scoped_table< string_ref, operation >;

    struct symbol_tables {
        funs_scope_table funs;
        vars_scope_table vars;
        types_scope_table types;
        members_scope_table members;
        labels_scope_table labels;
        enum_constants_table enum_constants;
    };


    struct symbols_view {

        explicit symbols_view(symbol_tables &symbols)
            : symbols(symbols)
        {}

        operation declare(operation op) {
            llvm::TypeSwitch< operation >(op)
                .Case< VarSymbolOpInterface >([&] (auto &op) {
                    symbols.vars.insert(op.getSymbolName(), op->getResult(0));
                })
                .Case< TypeSymbolOpInterface >([&] (auto &op) {
                    symbols.types.insert(op.getSymbolName(), op);
                })
                .Case< FuncSymbolOpInterface >([&] (auto &op) {
                    symbols.funs.insert(op.getSymbolName(), op);
                })
                .Case< MemberVarSymbolOpInterface >([&] (auto &op) {
                    symbols.members.insert(op.getSymbolName(), op);
                })
                .Case< LabelSymbolOpInterface >([&] (auto &op) {
                    symbols.labels.insert(op.getSymbolName(), op);
                })
                .Case< EnumConstantSymbolOpInterface >([&] (auto &op) {
                    symbols.enum_constants.insert(op.getSymbolName(), op);
                })
                .Default([] (auto &op){
                    VAST_UNREACHABLE("Unknown operation declaration type");
                });

            return op;
        }

        template< typename builder_t >
        auto maybe_declare(builder_t &&bld) -> decltype(bld()) {
            if (auto val = bld()) {
                return mlir::dyn_cast< decltype(bld()) >(declare(val));
            } else {
                return val;
            }
        }

        auto declare_function_param(string_ref name, mlir_value value) {
            return symbols.vars.insert(name, value), value;
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

        operation lookup_label(string_ref name) const {
            return symbols.labels.lookup(name);
        }

        bool is_declared_fun(string_ref name) const {
            return lookup_fun(name);
        }

        bool is_declared_type(string_ref name) const {
            return lookup_type(name);
        }

        symbol_tables &symbols;
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

            while (!children.empty()) {
                children.back()->finalize();
                children.pop_back();
            }
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
        std::deque< std::unique_ptr< scope_context > > children;
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
            , types(parent->symbols.types)
            , enum_constants(parent->symbols.enum_constants)
        {}

        virtual ~block_scope() = default;

        symbol_table_scope< string_ref, mlir_value > vars;

        symbol_table_scope< string_ref, operation > types;
        symbol_table_scope< string_ref, operation > enum_constants;
    };


    // Refers to function scope §6.2.1 of C standard
    struct function_scope : block_scope {
        using block_scope::block_scope;
        explicit function_scope(scope_context *parent)
            : block_scope(parent)
            , labels(parent->symbols.labels)
        {}

        virtual ~function_scope() = default;

        symbol_table_scope< string_ref, operation > labels;
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
            , enum_constants(symbols.enum_constants)
        {}

        virtual ~module_scope() = default;

        symbol_table_scope< string_ref, operation >  functions;
        symbol_table_scope< string_ref, operation >  types;
        symbol_table_scope< string_ref, mlir_value > globals;
        symbol_table_scope< string_ref, operation >  enum_constants;
    };

    // Scope of member names for structures and unions

    struct members_scope : scope_context {
        using scope_context::scope_context;
        explicit members_scope(scope_context *parent)
            : scope_context(parent)
            , members(parent->symbols.members)
        {}

        virtual ~members_scope() = default;

        symbol_table_scope< string_ref, operation > members;
    };

} // namespace vast::cg

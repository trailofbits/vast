// Copyright (c) 2024-present, Trail of Bits, Inc.

#include "vast/CodeGen/CodeGenFunction.hpp"

VAST_RELAX_WARNINGS
#include <clang/AST/GlobalDecl.h>
#include <clang/Basic/TargetInfo.h>
VAST_UNRELAX_WARNINGS

#include "vast/CodeGen/CodeGenBlock.hpp"
#include "vast/CodeGen/CodeGenModule.hpp"
#include "vast/CodeGen/Util.hpp"

#include "vast/Util/Maybe.hpp"

#include "vast/Dialect/Core/Linkage.hpp"

namespace vast::cg
{
    //
    // function generation
    //

    operation function_generator::emit_in_scope(region_t &scope, const clang_function *decl) {
        auto _ = bld.insertion_guard();
        bld.set_insertion_point_to_end(&scope);
        return emit(decl);
    }

    operation function_generator::emit(const clang_function *decl) {
        auto ctx = dynamic_cast< module_context* >(parent);
        VAST_CHECK(ctx, "function context must be a child of a module context");

        auto &pg = mk_child< prototype_generator >(bld, visitor);
        auto prototype = pg.emit_in_scope(ctx->mod->getBodyRegion(), decl);

        if (auto fn = mlir::dyn_cast< vast_function >(prototype)) {
            if (decl->hasBody()) {
                declare_function_params(decl, fn);

                defer([=] {
                    if (auto fn = mlir::dyn_cast< vast_function >(prototype)) {
                        auto &bg = mk_child< body_generator >(bld, visitor);
                        bg.emit_in_scope(fn.getBody(), decl, fn);
                    } else {
                        VAST_REPORT("can not emit function body for unknown prototype");
                    }
                });
            }
        }

        return prototype;
    }

    void function_generator::declare_function_params(const clang_function *decl, vast_function fn) {
        auto *entry_block = fn.addEntryBlock();
        auto params = llvm::zip(decl->parameters(), entry_block->getArguments());
        for (const auto &[param, earg] : params) {
            // TODO set alignment

            earg.setLoc(visitor.location(param));
            if (auto name = visitor.symbol(param)) {
                // TODO set name
                scope_context::declare(name.value(), earg);
            }
        }
    }

    //
    // function prototype generation
    //

    operation prototype_generator::emit_in_scope(region_t &scope, const clang_function *decl) {
        auto _ = bld.insertion_guard();
        bld.set_insertion_point_to_end(&scope);
        return emit(decl);
    }

    operation prototype_generator::lookup_or_declare(const clang_function *decl, module_context *mod) {
        if (auto symbol = visitor.symbol(decl)) {
            if (auto fn = mod->lookup_global(symbol.value())) {
                return fn;
            }
        }

        if (auto op = visitor.visit_prototype(decl)) {
            if (auto fn = mlir::dyn_cast< vast_function >(op)) {
                scope_context::declare(fn);
            }

            return op;
        }

        return {};
    }

    operation prototype_generator::emit(const clang_function *decl) {
        auto ctx = dynamic_cast< function_scope* >(parent);
        VAST_CHECK(ctx, "prototype generator must be a child of a function context");

        auto mod = dynamic_cast< module_context* >(ctx->parent);
        VAST_CHECK(mod, "function context must be a child of a module context");

        // TODO create a new function prototype scope here

        return lookup_or_declare(decl, mod);
    }

    //
    // function body generation
    //

    void body_generator::emit_in_scope(region_t &scope, const clang_function *decl, vast_function fn) {
        auto _ = bld.insertion_guard();
        bld.set_insertion_point_to_end(&scope);
        emit(decl, fn);
    }

    void body_generator::emit(const clang_function *decl, vast_function fn) {
        auto body = decl->getBody();

        if (clang::isa< clang::CoroutineBodyStmt >(body)) {
            VAST_REPORT("coroutines are not supported");
            return;
        }

        if (auto stmt = clang::dyn_cast< clang_compound_stmt >(body)) {
            auto &sg = mk_child< block_generator >(bld, visitor);
            sg.emit_in_scope(fn.getBody(), stmt);
        } else {
            visitor.visit(body);
        }

        VAST_ASSERT(mlir::succeeded(fn.verifyBody()));

        emit_epilogue(decl, fn);
    }

    insertion_guard body_generator::insert_at_end(vast_function fn) {
        auto guard = bld.insertion_guard();
        bld.set_insertion_point_to_end(&fn.getBody().back());
        return guard;
    }

    void body_generator::emit_implicit_return_zero(const clang_function *decl) {
        auto loc = visitor.location(decl);
        auto rty = visitor.visit(decl->getFunctionType()->getReturnType());

        auto value = bld.constant(loc, rty, apsint(0));
        bld.create< core::ImplicitReturnOp >(loc, value);
    }

    void body_generator::emit_implicit_void_return(const clang_function *decl) {
        VAST_CHECK( decl->getReturnType()->isVoidType(),
            "Can't emit implicit void return in non-void function."
        );

        auto loc = visitor.location(decl);
        auto value = bld.constant(loc);
        bld.create< core::ImplicitReturnOp >(loc, value);
    }

    void body_generator::emit_trap(const clang_function *decl) {
        // TODO fix when we support builtin function (emit enreachable for now)
        emit_unreachable(decl);
    }

    void body_generator::emit_unreachable(const clang_function *decl) {
        bld.create< hl::UnreachableOp >(visitor.location(decl));
    }

    bool may_drop_function_return(clang_qual_type rty, acontext_t &actx) {
        // We can't just disard the return value for a record type with a
        // complex destructor or a non-trivially copyable type.
        if (const auto *recorrd_type = rty.getCanonicalType()->getAs< clang::RecordType >()) {
            VAST_UNIMPLEMENTED;
        }

        return rty.isTriviallyCopyableType(actx);
    }

    bool body_generator::should_final_emit_unreachable(const clang_function *decl) const {
        auto ctx  = static_cast< function_generator* >(parent);
        auto rty  = decl->getReturnType();
        auto & actx = decl->getASTContext();
        return ctx->opts.has_strict_return || may_drop_function_return(rty, actx);
    }

    void body_generator::deal_with_missing_return(const clang_function *decl, vast_function fn) {
        auto ctx = static_cast< function_generator* >(parent);
        auto rty = decl->getReturnType();

        auto _ = insert_at_end(fn);

        if (rty->isVoidType()) {
            emit_implicit_void_return(decl);
        } else if (decl->hasImplicitReturnZero()) {
            emit_implicit_return_zero(decl);
        } else if (should_final_emit_unreachable(decl)) {
            // C++11 [stmt.return]p2:
            //   Flowing off the end of a function [...] results in undefined behavior
            //   in a value-returning function.
            // C11 6.9.1p12:
            //   If the '}' that terminates a function is reached, and the value of the
            //   function call is used by the caller, the behavior is undefined.

            // TODO: skip if SawAsmBlock
            if (ctx->opts.optimization_level == 0) {
                emit_trap(decl);
            } else {
                emit_unreachable(decl);
            }
        } else {
            VAST_UNIMPLEMENTED_MSG("unknown missing return case");
        }
    }

    operation get_last_effective_operation(block_t &block) {
        if (block.empty()) {
            return {};
        }
        auto last = &block.back();
        if (auto scope = mlir::dyn_cast< core::ScopeOp >(last)) {
            return get_last_effective_operation(scope.getBody().back());
        }

        return last;
    }

    bool has_return(block_t &block) {
        if (auto op = get_last_effective_operation(block)) {
            return op->template hasTrait< core::return_trait >();
        }
        return false;
    }

    void body_generator::emit_epilogue(const clang_function *decl, vast_function fn) {
        auto &last_block = fn.getBody().back();
        if (!has_return(last_block)) {
            deal_with_missing_return(decl, fn);
        }

        // Emit the standard function epilogue.
        // TODO: finishFunction(BodyRange.getEnd());

        // TODO: If we haven't marked the function nothrow through other means, do a
        // quick pass now to see if we can.
        // if (!decl->doesNotThrow()) TryMarkNoThrow(decl);
    }

} // namespace vast::cg

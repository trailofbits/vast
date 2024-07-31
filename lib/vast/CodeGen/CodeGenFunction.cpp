// Copyright (c) 2024-present, Trail of Bits, Inc.

#include "vast/CodeGen/CodeGenFunction.hpp"

VAST_RELAX_WARNINGS
#include <clang/AST/GlobalDecl.h>
#include <clang/Basic/TargetInfo.h>
#include <clang/AST/Stmt.h>
#include <clang/AST/StmtCXX.h>
VAST_UNRELAX_WARNINGS

#include "vast/CodeGen/CodeGenBlock.hpp"
#include "vast/CodeGen/CodeGenModule.hpp"
#include "vast/CodeGen/Util.hpp"

#include "vast/Util/Maybe.hpp"

#include "vast/Dialect/Core/Linkage.hpp"

#include "vast/Dialect/Builtin/Ops.hpp"

namespace vast::cg
{
    mlir_visibility get_function_visibility(const clang_function *decl, std::optional< linkage_kind > linkage) {
        if (!linkage) {
            return mlir_visibility::Private;
        }
        if (decl->isThisDeclarationADefinition()) {
            return core::get_visibility_from_linkage(linkage.value());
        }
        if (decl->doesDeclarationForceExternallyVisibleDefinition()) {
            return mlir_visibility::Public;
        }
        return mlir_visibility::Private;
    }

    vast_function set_visibility(const clang_function *decl, vast_function fn) {
        auto visibility = get_function_visibility(decl, fn.getLinkage());
        mlir::SymbolTable::setSymbolVisibility(fn, visibility);
        return fn;
    }

    //
    // function generation
    //

    operation mk_prototype(auto &parent, const clang_function *decl) {
        auto gen = mk_scoped_generator< prototype_generator >(parent);
        return gen.emit(decl);
    }

    operation function_generator::emit(const clang_function *decl) {
        auto prototype = [&] {
            if (auto symbol = visitor.symbol(decl)) {
                if (auto op = visitor.scope.lookup_fun(symbol.value())) {
                    auto fn        = mlir::cast< vast_function >(op);
                    // Function declaration that is not a prototype will have zero arguments
                    // and we need to fix that when we discover them
                    // This is caused by a deprecated C feature
                    auto fn_args   = fn.getNumArguments();
                    auto decl_args = decl->getNumParams();
                    VAST_CHECK(
                        fn_args == decl_args || fn_args == 0 || decl_args == 0,
                        "Mismatch in number of arguments between function declaration and "
                        "vast function."
                    );
                    if (fn_args == 0 && decl_args != 0) {
                        fn.setType(visitor.visit(decl->getFunctionType()));
                    }

                    // inline function declarations (without definition) are unknown
                    // because clang can not provide the information for them
                    auto linkage = fn->getAttrOfType< core::GlobalLinkageKindAttr >("linkage");
                    if (!linkage) {
                        if (auto new_linkage = core::get_function_linkage(decl)) {
                            auto attr = core::GlobalLinkageKindAttr::get(fn.getContext(), new_linkage.value());
                            fn->setAttr("linkage", attr);
                        }
                        set_visibility(decl, fn);
                    }

                    // Some later declaration might cause an inline function to become
                    // externally available.
                    // Section 6.7.4 of C99 standard provides information on this
                    // Summary is avaialble on https://www.greenend.org.uk/rjk/tech/inline.html

                    // Sadly, we can not use the standard query mechanism because clang does not
                    // allow to query about linkage on inline specified function declaration
                    // that does not have a body. To make things even uglier the
                    // "...ForceExternallyVisible..." query does not allow to ask when the
                    // declaration does have a body for some reason which is beyond me
                    // The linakge querrying is full of asserts some of which are very
                    // questionable
                    if (!decl->doesThisDeclarationHaveABody()
                        && decl->doesDeclarationForceExternallyVisibleDefinition())
                    {
                        fn->setAttr(
                            "linkage",
                            core::GlobalLinkageKindAttr::get(
                                fn.getContext(), core::GlobalLinkageKind::ExternalLinkage
                            )
                        );
                        set_visibility(decl, fn);
                    }
                    return op;
                }
            }

            return mk_prototype(*this, decl);
        } ();

        if (decl->isThisDeclarationADefinition()) {
            // Unsupported functions might produce unsupported decl
            if (auto fn = mlir::dyn_cast< vast_function >(prototype)) {
                defer([parent = *this, decl, fn]() mutable {
                    // If the user implements a function that is also a builtin,
                    // it might be visited multiple times
                    if (fn.getBody().empty()) {
                        set_visibility(decl, fn);
                        if (!decl->hasDefiningAttr()) {
                            parent.declare_function_params(decl, fn);
                            parent.emit_labels(decl, fn);
                            parent.emit_body(decl, fn);
                        }

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

            earg.setLoc(visitor.location(param).value());
            if (auto name = visitor.symbol(param)) {
                // TODO set name
                scope().declare_function_param(name.value(), earg);
            }
        }
    }

    void function_generator::emit_labels(const clang_function *decl, vast_function fn) {
        auto _ = bld.scoped_insertion_at_start(&fn.getBody());
        for (const auto lab : filter< clang::LabelDecl >(decl->decls())) {
            visitor.visit(lab);
        }
    }

    void function_generator::emit_body(const clang_function *decl, vast_function prototype) {
        auto _ = bld.scoped_insertion_at_end(&prototype.getBody());
        auto body = decl->getBody();

        if (clang::isa< clang::CoroutineBodyStmt >(body)) {
            VAST_REPORT("coroutines are not supported");
            return;
        }

        if (!decl->hasTrivialBody()) {
            visitor.visit(body);
        }

        emit_epilogue(decl, prototype);
        VAST_ASSERT(mlir::succeeded(prototype.verifyBody()));
    }


    void function_generator::emit_implicit_return_zero(const clang_function *decl) {
        auto loc = visitor.location(decl).value();
        auto rty = visitor.visit(decl->getFunctionType()->getReturnType());

        auto value = bld.constant(loc, rty, apsint(0));
        bld.create< core::ImplicitReturnOp >(loc, value);
    }

    void function_generator::emit_implicit_void_return(const clang_function *decl) {
        VAST_CHECK( decl->getReturnType()->isVoidType(),
            "Can't emit implicit void return in non-void function."
        );

        auto loc = visitor.location(decl).value();
        auto value = bld.constant(loc);
        bld.create< core::ImplicitReturnOp >(loc, value);
    }

    void function_generator::emit_trap(const clang_function *decl) {
        bld.create< hlbi::TrapOp >(visitor.location(decl).value(), bld.void_type());
    }

    void function_generator::emit_unreachable(const clang_function *decl) {
        bld.create< hl::UnreachableOp >(visitor.location(decl).value());
    }

    bool may_drop_function_return(clang_qual_type rty, acontext_t &actx) {
        // We can't just discard the return value for a record type with a
        // complex destructor or a non-trivially copyable type.
        if (const auto *recorrd_type = rty.getCanonicalType()->getAs< clang::RecordType >()) {
            VAST_UNIMPLEMENTED;
        }

        return rty.isTriviallyCopyableType(actx);
    }

    bool function_generator::should_final_emit_unreachable(const clang_function *decl) const {
        auto rty  = decl->getReturnType();
        auto &actx = decl->getASTContext();
        return emit_strict_function_return || may_drop_function_return(rty, actx);
    }

    void function_generator::deal_with_missing_return(const clang_function *decl, vast_function fn) {
        auto rty = decl->getReturnType();

        auto _ = bld.scoped_insertion_at_end(&fn.getBody().back());

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
            switch (missing_return_policy) {
                case missing_return_policy::emit_trap:
                    emit_trap(decl);
                    break;
                case missing_return_policy::emit_unreachable:
                    emit_unreachable(decl);
                    break;
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
            return core::is_return_like(op);
        }
        return false;
    }

    void function_generator::emit_epilogue(const clang_function *decl, vast_function fn) {
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

    //
    // function prototype generation
    //

    operation prototype_generator::emit(const clang_function *decl) {
        // TODO create a new function prototype scope here
        if (auto op = visitor.visit_prototype(decl)) {
            if (auto fn = mlir::dyn_cast< vast_function >(op)) {
                scope().declare(fn);
            }

            return op;
        }

        return {};
    }

} // namespace vast::cg

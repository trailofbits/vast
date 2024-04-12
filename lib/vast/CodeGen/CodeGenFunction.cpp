// Copyright (c) 2022-present, Trail of Bits, Inc.

#include "vast/CodeGen/CodeGenDriver.hpp"

VAST_RELAX_WARNINGS
#include <clang/AST/GlobalDecl.h>
#include <clang/Basic/TargetInfo.h>
VAST_UNRELAX_WARNINGS

namespace vast::cg
{
    bool codegen_driver::may_drop_function_return(clang::QualType rty) const {
        // We can't just disard the return value for a record type with a complex
        // destructor or a non-trivially copyable type.
        if (const auto *recorrd_type = rty.getCanonicalType()->getAs< clang::RecordType >()) {
            VAST_UNIMPLEMENTED;
        }

        return rty.isTriviallyCopyableType(acontext());
    }

    void codegen_driver::deal_with_missing_return(hl::FuncOp fn, const clang::FunctionDecl *decl) {
        auto rty = decl->getReturnType();

        bool shoud_emit_unreachable = (
            opts.codegen.StrictReturn || may_drop_function_return(rty)
        );

        // if (SanOpts.has(SanitizerKind::Return)) {
        //     VAST_UNIMPLEMENTED;
        // }

        if (rty->isVoidType()) {
            codegen.emit_implicit_void_return(fn, decl);
        } else if (decl->hasImplicitReturnZero()) {
            codegen.emit_implicit_return_zero(fn, decl);
        } else if (shoud_emit_unreachable) {
            // C++11 [stmt.return]p2:
            //   Flowing off the end of a function [...] results in undefined behavior
            //   in a value-returning function.
            // C11 6.9.1p12:
            //   If the '}' that terminates a function is reached, and the value of the
            //   function call is used by the caller, the behavior is undefined.

            // TODO: skip if SawAsmBlock
            if (opts.codegen.OptimizationLevel == 0) {
                codegen.emit_trap(fn, decl);
            } else {
                codegen.emit_unreachable(fn, decl);
            }
        } else {
            VAST_UNIMPLEMENTED_MSG("unknown missing return case");
        }
    }

    operation get_last_effective_operation(auto &block) {
        if (block.empty()) {
            return {};
        }
        auto last = &block.back();
        if (auto scope = mlir::dyn_cast< core::ScopeOp >(last)) {
            return get_last_effective_operation(scope.getBody().back());
        }

        return last;
    }

    hl::FuncOp codegen_driver::emit_function_epilogue(hl::FuncOp fn, clang::GlobalDecl decl) {
        auto function_decl = clang::cast< clang::FunctionDecl >( decl.getDecl() );

        auto &last_block = fn.getBody().back();
        auto missing_return = [&] (auto &block) {
            if (codegen.has_insertion_block()) {
                if (auto op = get_last_effective_operation(block)) {
                    return !core::is_return_like(op);
                }
                return true;
            }

            return false;
        };

        if (missing_return(last_block)) {
            deal_with_missing_return(fn, function_decl);
        }


        // Emit the standard function epilogue.
        // TODO: finishFunction(BodyRange.getEnd());

        // If we haven't marked the function nothrow through other means, do a quick
        // pass now to see if we can.
        // TODO: if (!CurFn->doesNotThrow()) TryMarkNoThrow(CurFn);

        return fn;
    }

    // This function implements the logic from CodeGenFunction::GenerateCode
    hl::FuncOp codegen_driver::build_function_body(hl::FuncOp fn, clang::GlobalDecl decl) {
        fn = codegen.emit_function_prologue(fn, decl, opts);

        if (mlir::failed(fn.verifyBody())) {
            return nullptr;
        }

        return emit_function_epilogue(fn, decl);
    }

} // namespace vast::cg

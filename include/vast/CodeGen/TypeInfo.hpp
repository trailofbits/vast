// Copyright (c) 2023, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <clang/AST/GlobalDecl.h>
#include <clang/CodeGen/CGFunctionInfo.h>
#include <mlir/IR/BuiltinTypes.h>
#include <llvm/ADT/SmallPtrSet.h>
VAST_UNRELAX_WARNINGS

#include "vast/CodeGen/ABIInfo.hpp"
#include "vast/CodeGen/CallingConv.hpp"
#include "vast/CodeGen/FunctionInfo.hpp"

namespace vast::cg {
    struct codegen_driver;
    struct target_info_t;

    // This class organizes the cross-module state that is used while lowering
    // AST types to VAST high-level types.
    struct type_info_t {
        type_info_t(codegen_driver &codegen);

        ~type_info_t() {
            // clean up function infos
            std::vector< std::unique_ptr< function_info_t > > ptrs;
            for (auto &info : function_infos) {
                ptrs.emplace_back(&info);
            }
            function_infos.clear();
        }

        // Convert clang calling convention to LLVM calling convention.
        calling_conv to_vast_calling_conv(clang::CallingConv cc);

        // The arrangement methods are split into three families:
        //   - those meant to drive the signature and prologue/epilogue
        //     of a function declaration or definition,
        //   - those meant for the computation of the VAST type for an abstract
        //     appearance of a function, and
        //   - those meant for performing the VAST-generation of a call.
        // They differ mainly in how they deal with optional (i.e. variadic)
        // arguments, as well as unprototyped functions.
        //
        // Key points:
        // - The function_info_t for emitting a specific call site must include
        //   entries for the optional arguments.
        // - The function type used at the call site must reflect the formal
        // signature
        //   of the declaration being called, or else the call will go away.
        // - For the most part, unprototyped functions are called by casting to a
        //   formal signature inferred from the specific argument types used at the
        //   call-site. However, some targets (e.g. x86-64) screw with this for
        //   compatability reasons.
        const function_info_t &arrange_global_decl(clang::GlobalDecl decl, target_info_t &target_info);
        const function_info_t &arrange_function_decl(const clang::FunctionDecl *fn, target_info_t &target_info);

        // C++ methods have some special rules and also have implicit parameters.
        const function_info_t &arrange_cxx_method_decl(
            const clang::CXXMethodDecl *decl
        );

        const function_info_t &arrange_cxx_structor_decl(
            clang::GlobalDecl decl
        );

        const function_info_t & arrange_cxx_method_type(
            const clang::CXXRecordDecl *record,
            const clang::FunctionProtoType *prototype,
            const clang::CXXMethodDecl *method
        );

        // const function_info_t &arrange_free_function_call(
        //     const CallArgList &args,
        //     const clang::FunctionType *ty,
        //     bool chain_call
        // );

        const function_info_t &arrange_free_function_type(
            const clang::FunctionProtoType *function_type,
            target_info_t &target_info
        );

        // "Arrange" the vast information for a call or type with the given
        // signature. This is largely an internal method; other clients should use
        // one of the above routines, which ultimatley defer to this.
        //
        // \param arg_types - must all actually be canonical as params
        const function_info_t &arrange_function_info(
            qual_type rty, bool instance_method, bool chain_call,
            qual_types_span arg_types,
            ext_info info,
            ext_parameter_info_span params,
            required_args args,
            target_info_t &target_info
        );

        codegen_driver &codegen;

    private:

        // Hold memoized function_info_t results
        llvm::FoldingSet< function_info_t > function_infos;
    };

} // namespace vast::cg

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

    struct codegen_module;

    // This class organizes the cross-module state that is used while lowering
    // AST types to VAST high-level types.
    struct types_generator {
        types_generator(codegen_module &cgm);

        // Convert clang calling convention to LLVM calling convention.
        calling_conv to_vast_calling_conv(clang::CallingConv cc);

        using type_cache_t = llvm::DenseMap< const clang::Type *, mlir_type >;
        type_cache_t type_cache;

        const abi_info_t &get_abi_info() const { return abi_info; }

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
        const function_info_t &arrange_global_decl(clang::GlobalDecl decl);
        const function_info_t &arrange_function_decl(const clang::FunctionDecl *fn);

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
            clang::CanQual<clang::FunctionProtoType> type
        );

        mlir::FunctionType get_function_type(clang::GlobalDecl decl);
        mlir::FunctionType get_function_type(const function_info_t &info);

        // Convert type into a mlir_type.
        mlir_type convert_type(qual_type type);
        mlir_type convert_type_impl(const clang::Type *type);

        // "Arrange" the vast information for a call or type with the given
        // signature. This is largely an internal method; other clients should use
        // one of the above routines, which ultimatley defer to this.
        //
        // \param arg_types - must all actually be canonical as params
        const function_info_t &arrange_function_info(
            can_qual_type rty, bool instance_method, bool chain_call,
            can_qual_types_span arg_types,
            ext_info info,
            ext_parameter_info_span params,
            required_args args
        );
    private:
        codegen_module &cgm;

        // This should not be moved earlier, since its initialization depends on some
        // of the previous reference members being already initialized
        const abi_info_t &abi_info;

        // Hold memoized function_info_t results
        llvm::FoldingSet< function_info_t > function_infos;

        llvm::SmallPtrSet< const function_info_t*, 4 > functions_being_processed;
    };

} // namespace vast::cg

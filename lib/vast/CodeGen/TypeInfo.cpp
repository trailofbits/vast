// Copyright (c) 2023-present, Trail of Bits, Inc.

#include "vast/CodeGen/TypeInfo.hpp"
#include "vast/CodeGen/CallingConv.hpp"

#include "vast/Frontend/Common.hpp"

#include "vast/Translation/CodeGenDriver.hpp"
#include "vast/CodeGen/TypeInfo.hpp"
#include "vast/CodeGen/ABIInfo.hpp"

namespace vast::cg
{
    type_info_t::type_info_t(codegen_driver &codegen)
        : codegen{codegen}
    {}

    calling_conv type_info_t::to_vast_calling_conv(clang::CallingConv cc) {
        if (cc != clang::CC_C) {
            cc::compiler_error("No other calling conventions implemented.");
        }

        return calling_conv::C;
    }

    const function_info_t &type_info_t::arrange_global_decl(
        clang::GlobalDecl glob, target_info_t &target_info
    ) {
        auto decl = glob.getDecl();
        assert(!llvm::dyn_cast< clang::ObjCMethodDecl >(decl)
            && "This is reported as a FIXME in codegen"
        );

        const auto *fn = llvm::cast< clang::FunctionDecl >(decl);

        assert(!llvm::isa< clang::CXXConstructorDecl >(decl) && "NYI");
        assert(!llvm::isa< clang::CXXDestructorDecl >(decl) && "NYI");

        return arrange_function_decl(fn, target_info);
    }

    const function_info_t &type_info_t::arrange_function_decl(
        const clang::FunctionDecl *fn, target_info_t &target_info
    ) {
        if (const auto *method = llvm::dyn_cast< clang::CXXMethodDecl >(fn)) {
            if (method->isInstance()) {
                return arrange_cxx_method_decl(method);
            }
        }

        auto fty = fn->getType()->getCanonicalTypeUnqualified();

        assert(llvm::isa< clang::FunctionType >(fty));
        // TODO: setCUDAKernelCallingConvention

        // When declaring a function without a prototype, always use a
        // non-variadic type.
        if (auto noproto = fty.getAs< clang::FunctionNoProtoType >()) {
            return arrange_function_info(
                noproto->getReturnType(),
                /* instance_method */ false,
                /* chain_call */ false, {},
                noproto->getExtInfo(), {},
                require_all_args,
                target_info
            );
        }

        return arrange_free_function_type(fty.castAs< clang::FunctionProtoType >(), target_info);
    }

    const function_info_t &type_info_t::arrange_cxx_method_decl(
        const clang::CXXMethodDecl * /* decl */
    ) {
        throw cc::compiler_error("arrange_cxx_method_decl not implemented");
    }

    const function_info_t &type_info_t::arrange_cxx_structor_decl(
        clang::GlobalDecl /* decl */
    ) {
        throw cc::compiler_error("arrange_cxx_structor_decl not implemented");
    }

    const function_info_t &type_info_t::arrange_cxx_method_type(
        const clang::CXXRecordDecl * /* record */,
        const clang::FunctionProtoType * /* prototype */,
        const clang::CXXMethodDecl * /* method */
    ) {
        throw cc::compiler_error("arrange_cxx_method_type not implemented");
    }

    // const function_info_t &type_info_t::arrange_free_function_call(
    //     const CallArgList &Args,
    //     const clang::FunctionType *Ty,
    //     bool ChainCall
    // ) {
    //    throw cc::compiler_error("arrange_free_function_call not implemented");
    // }

    // Adds the formal parameters in FPT to the given prefix. If any parameter in
    // FPT has pass_object_size_attrs, then we'll add parameters for those, too.
    static void append_parameter_types(
        const type_info_t &type_info,
        llvm::SmallVectorImpl< clang::CanQualType > &prefix,
        llvm::SmallVectorImpl< clang::FunctionProtoType::ExtParameterInfo > &param_infos,
        clang::CanQual< clang::FunctionProtoType > function_type
    ) {
        // Fast path: don't touch param info if we don't need to.
        if (!function_type->hasExtParameterInfos()) {
            assert(param_infos.empty() && "We have paramInfos, but the prototype doesn't?");
            prefix.append(function_type->param_type_begin(), function_type->param_type_end());
            return;
        }

        VAST_UNREACHABLE("params NYI");
    }

    const function_info_t &arrange_function_info(
        type_info_t &type_info, bool instance_method,
        llvm::SmallVectorImpl< clang::CanQualType > &prefix,
        clang::CanQual< clang::FunctionProtoType > function_type,
        target_info_t &target_info
    ) {
        llvm::SmallVector< clang::FunctionProtoType::ExtParameterInfo, 16 > param_infos;
        auto required = required_args::for_prototype_plus(function_type, prefix.size());

        // FIXME: Kill copy. -- from codegen
        append_parameter_types(type_info, prefix, param_infos, function_type);
        auto resultType = function_type->getReturnType().getUnqualifiedType();

        return type_info.arrange_function_info(
            resultType, instance_method,
            /* chain call=*/ false, prefix,
            function_type->getExtInfo(),
            param_infos, required,
            target_info
        );
    }


    const function_info_t &type_info_t::arrange_free_function_type(
        clang::CanQual< clang::FunctionProtoType > function_type,
        target_info_t &target_info
    ) {
        llvm::SmallVector< clang::CanQualType, 16 > arg_types;
        return ::vast::cg::arrange_function_info(
            *this,
            /* instance method */ false,
            arg_types,
            function_type,
            target_info
        );
    }

    const function_info_t &type_info_t::arrange_function_info(
        can_qual_type rty,
        bool instance_method,
        bool chain_call,
        can_qual_types_span arg_types,
        ext_info info,
        ext_parameter_info_span params,
        required_args args,
        target_info_t &target_info
    ) {
        assert(llvm::all_of(arg_types, [] (can_qual_type ty) {
            return ty.isCanonicalAsParam(); })
        );

        // Lookup or create unique function info.
        llvm::FoldingSetNodeID id;
        function_info_t::Profile(
            id, instance_method, chain_call, info, params, args, rty, arg_types
        );

        void *insert_pos = nullptr;
        if (auto *fninfo = function_infos.FindNodeOrInsertPos(id, insert_pos)) {
            return *fninfo;
        }

        auto cc = to_vast_calling_conv(info.getCC());

        // Construction the function info. We co-allocate the ArgInfos.
        auto fninfo = function_info_t::create(
            cc, instance_method, chain_call, info, params, rty, arg_types, args
        );

        function_infos.InsertNode(fninfo, insert_pos);

        auto lock = codegen.make_lock(fninfo);

        const auto &abi = target_info.get_abi_info();
        // FIXME: remove and make vast pass: compute ABI inforamtion.
        assert(info.getCC() != clang::CallingConv::CC_SpirFunction && "not supported");
        assert(info.getCC() != clang::CC_Swift && "Swift not supported");
        assert(info.getCC() != clang::CC_SwiftAsync && "Swift not supported");
        abi.compute_info(*fninfo);

        // FIXME: deal with type coersion later in the vast pipeline
        // Loop over all of the computed argument and return value info. If any of
        // them are direct or extend without a specified coerce type, specify the
        // default now.

        return *fninfo;
    }
} // namespace vast::cg

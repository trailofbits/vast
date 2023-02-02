// Copyright (c) 2023-present, Trail of Bits, Inc.

#include "vast/CodeGen/TypeInfo.hpp"
#include "vast/CodeGen/CallingConv.hpp"

#include "vast/Frontend/Common.hpp"

namespace vast::cg
{
    type_info_t::type_info_t(codegen_driver &codegen)
        : codegen{codegen}
        // , abi_info(cgm.get_target_info().get_abi_info())
    {}

    calling_conv type_info_t::to_vast_calling_conv(clang::CallingConv cc) {
        if (cc != clang::CC_C) {
            cc::compiler_error("No other calling conventions implemented.");
        }

        return calling_conv::C;
    }

    const function_info_t &type_info_t::arrange_global_decl(
        clang::GlobalDecl glob
    ) {
        auto decl = glob.getDecl();
        assert(!llvm::dyn_cast< clang::ObjCMethodDecl >(decl)
            && "This is reported as a FIXME in codegen"
        );

        const auto *fn = llvm::cast< clang::FunctionDecl >(decl);

        assert(!llvm::isa< clang::CXXConstructorDecl >(decl) && "not implemented");
        assert(!llvm::isa< clang::CXXDestructorDecl >(decl) && "not implemented");

        return arrange_function_decl(fn);
    }

    const function_info_t &type_info_t::arrange_function_decl(
        const clang::FunctionDecl *fn
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
                require_all_args
            );
        }

        return arrange_free_function_type(fty.castAs< clang::FunctionProtoType >());
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
        throw cc::compiler_error("arrange_free_function_type not implemented");
    }

    // const function_info_t &type_info_t::arrange_free_function_call(
    //     const CallArgList &Args,
    //     const clang::FunctionType *Ty,
    //     bool ChainCall
    // ) {
    //    throw cc::compiler_error("arrange_free_function_call not implemented");
    // }

    const function_info_t &type_info_t::arrange_free_function_type(
        clang::CanQual< clang::FunctionProtoType > /* type */
    ) {
        throw cc::compiler_error("arrange_free_function_type not implemented");
    }

    const function_info_t &type_info_t::arrange_function_info(
        can_qual_type rty,
        bool instance_method,
        bool chain_call,
        can_qual_types_span arg_types,
        ext_info info,
        ext_parameter_info_span params,
        required_args args
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

        if (!functions_being_processed.insert(fninfo).second) {
            throw cc::compiler_error("trying to process a function recursively");
        }

        // Compute ABI inforamtion.
        assert(info.getCC() != clang::CallingConv::CC_SpirFunction && "not supported");
        assert(info.getCC() != clang::CC_Swift && "Swift not supported");
        assert(info.getCC() != clang::CC_SwiftAsync && "Swift not supported");
        // abi_info.compute_info(*fninfo);

        throw cc::compiler_error("arrange_function_info not implemented");

        // Loop over all of the computed argument and return value info. If any of
        // them are direct or extend without a specified coerce type, specify the
        // default now.
        // auto convert = [&] (auto &info, auto &&type) {
        //     if (info.can_have_coerce_to_type() && info.get_coerce_to_type() == nullptr) {
        //         info.set_coerce_to_type(convert_type(type));
        //     }
        // };

        // convert(fninfo->get_return_info(), fninfo->get_return_type());

        // for (auto &i : fninfo->arguments()) {
        //     convert(i.info, i.type);
        // }

        // if (functions_being_processed.erase(fninfo)) {
        //     throw cc::compiler_error("function info not being processed");
        // }
        (void)codegen;

        return *fninfo;
    }
} // namespace vast::cg

// Copyright (c) 2023-present, Trail of Bits, Inc.

#include "vast/CodeGen/Generator.hpp"

namespace vast::cg {

    void vast_generator::anchor() {}

    void vast_generator::Initialize(acontext_t &actx) {
        this->acontext = &actx;
        this->mcontext = std::make_unique< mcontext_t >();

        codegen_driver_options options {
            .verbose_diagnostics = true,
            // forwarded options form clang codegen
            .coverage_mapping   = bool(cgo.CoverageMapping),
            .keep_static_consts = bool(cgo.KeepStaticConsts)
        };

        // TODO initialize dialects here
        this->codegen = std::make_unique< codegen_driver >(
            *acontext, *mcontext, options, get_target_info()
        );
    }

    std::unique_ptr< mcontext_t > vast_generator::take_context() {
        return std::move(mcontext);
    }

    bool vast_generator::HandleTopLevelDecl(clang::DeclGroupRef decls) {
        if (diags.hasErrorOccurred())
            return true;

        return codegen->handle_top_level_decl(decls), true;
    }

    void vast_generator::HandleTranslationUnit(acontext_t &acontext) {
        codegen->handle_translation_unit(acontext);
    }

    void vast_generator::HandleInlineFunctionDefinition(clang::FunctionDecl */* decl */) {
        throw cc::compiler_error("HandleInlineFunctionDefinition not implemented");
    }

    void vast_generator::HandleTagDeclDefinition(clang::TagDecl */* decl */) {
        throw cc::compiler_error("HandleTagDeclDefinition not implemented");
    }

    void vast_generator::HandleTagDeclRequiredDefinition(const clang::TagDecl */* decl */) {
        throw cc::compiler_error("HandleTagDeclRequiredDefinition not implemented");
    }

    bool vast_generator::verify_module() const { return codegen->verify_module(); }

    owning_module_ref vast_generator::freeze() { return codegen->freeze(); }

    x86_avx_abi_level avx_level(const clang::TargetInfo &target) {
        const auto &abi = target.getABI();

        if (abi == "avx512")
            return x86_avx_abi_level::avx512;
        if (abi == "avx")
            return x86_avx_abi_level::avx;
        return x86_avx_abi_level::none;
    }

    using target_info_ptr = std::unique_ptr< target_info_t >;
    target_info_ptr get_target_info_impl(
        const clang::TargetInfo &target, const type_info_t &type_info
    ) {
        const auto &triple = target.getTriple();
        const auto &abi = target.getABI();

        auto aarch64_info = [&] {
            if (abi == "aapcs" || abi == "darwinpcs") {
                cc::compiler_error("Only Darwin supported for aarch64");
            }

            auto abi_kind = aarch64_abi_info::abi_kind::darwin_pcs;
            return target_info_ptr(new aarch64_target_info(type_info, abi_kind));
        };

        auto x86_64_info = [&] {
            auto os = triple.getOS();
            if (os == llvm::Triple::Linux) {
                return target_info_ptr(
                    new x86_64_target_info(type_info, avx_level(target))
                );
            }

            if (os == llvm::Triple::Darwin) {
                return target_info_ptr(
                    new darwin_x86_64_target_info(type_info, avx_level(target))
                );
            }

            throw cc::compiler_error("Unsupported x86_64 OS type.");
        };

        switch (triple.getArch()) {
            case llvm::Triple::aarch64: return aarch64_info();
            case llvm::Triple::x86_64:  return x86_64_info();
            default: throw cc::compiler_error("Target not yet supported.");
        }
    }

    const target_info_t &vast_generator::get_target_info() {
        if (!target_info) {
            const auto &target = acontext->getTargetInfo();
            target_info = get_target_info_impl(target, codegen->get_type_info());
        }

        return *target_info;
    }

} // namespace vast::cc

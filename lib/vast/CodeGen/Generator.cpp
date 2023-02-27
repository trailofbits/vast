// Copyright (c) 2023-present, Trail of Bits, Inc.

#include "vast/CodeGen/Generator.hpp"

namespace vast::cg {

    void vast_generator::anchor() {}

    void vast_generator::Initialize(acontext_t &actx) {
        this->acontext = &actx;
        this->mcontext = std::make_unique< mcontext_t >();

        codegen_options options {
            .verbose_diagnostics = true,
            // forwarded options form clang codegen
            .coverage_mapping                = bool(cgo.CoverageMapping),
            .keep_static_consts              = bool(cgo.KeepStaticConsts),
            .patchable_function_entry_count  = cgo.PatchableFunctionEntryCount,
            .patchable_function_entry_offset = cgo.PatchableFunctionEntryOffset,
            .no_use_jump_tables              = bool(cgo.NoUseJumpTables),
            .no_inline_line_tables           = bool(cgo.NoInlineLineTables),
            .packed_stack                    = bool(cgo.PackedStack),
            .warn_stack_size                 = cgo.WarnStackSize,
            .strict_return                   = bool(cgo.StrictReturn),
            .optimization_level              = cgo.OptimizationLevel,
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

    // HandleTagDeclDefinition - This callback is invoked each time a TagDecl to
    // (e.g. struct, union, enum, class) is completed. This allows the client hack
    // on the type, which can occur at any point in the file (because these can be
    // defined in declspecs).
    void vast_generator::HandleTagDeclDefinition(clang::TagDecl *decl) {
        if (diags.hasErrorOccurred()) {
            return;
        }

        // Don't allow re-entrant calls to generator triggered by PCH
        // deserialization to emit deferred decls.
        defer_handle_of_top_level_decl handling_decl(*codegen, /* emit deferred */false);

        codegen->update_completed_type(decl);

        // For MSVC compatibility, treat declarations of static data members with
        // inline initializers as definitions.
        if (acontext->getTargetInfo().getCXXABI().isMicrosoft()) {
            llvm_unreachable("NYI");
        }
        // For OpenMP emit declare reduction functions, if required.
        if (acontext->getLangOpts().OpenMP) {
            llvm_unreachable("NYI");
        }
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

            if (os == llvm::Triple::Win32) {
                throw cg::unimplemented( "target info for Win32" );
            } else {
                return target_info_ptr(
                    new x86_64_target_info(type_info, avx_level(target))
                );
            }

            throw cc::compiler_error(std::string("Unsupported x86_64 OS type: ") + triple.getOSName().str());
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

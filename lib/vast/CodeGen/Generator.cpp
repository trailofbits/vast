// Copyright (c) 2023-present, Trail of Bits, Inc.

#include "vast/CodeGen/Generator.hpp"
#include "vast/CodeGen/CodeGenContext.hpp"
#include <memory>

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

        this->cgcontext = std::make_unique< CodeGenContext >(*this->mcontext, *this->acontext);
        // TODO initialize dialects here
        this->codegen = std::make_unique< codegen_driver >(*this->cgcontext, options);
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
        VAST_UNIMPLEMENTED;
    }

    void vast_generator::CompleteTentativeDefinition(clang::VarDecl *decl){
        codegen->handle_top_level_decl(decl);
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
            VAST_UNIMPLEMENTED;
        }

        // For OpenMP emit declare reduction functions, if required.
        if (acontext->getLangOpts().OpenMP) {
            VAST_UNIMPLEMENTED;
        }
    }

    void vast_generator::HandleTagDeclRequiredDefinition(const clang::TagDecl */* decl */) {
        VAST_UNIMPLEMENTED;
    }

    bool vast_generator::verify_module() const { return codegen->verify_module(); }

    owning_module_ref vast_generator::freeze() { return std::move(cgcontext->mod); }

    type_info_t   &vast_generator::get_type_info()   { return codegen->get_type_info(); }
    target_info_t &vast_generator::get_target_info() { return codegen->get_target_info(); }

} // namespace vast::cc

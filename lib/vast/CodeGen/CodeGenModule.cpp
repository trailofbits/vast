// Copyright (c) 2024-present, Trail of Bits, Inc.

#include "vast/CodeGen/CodeGenModule.hpp"

VAST_RELAX_WARNINGS
#include <clang/Basic/SourceManager.h>
#include <mlir/IR/Verifier.h>
VAST_UNRELAX_WARNINGS

#include "vast/CodeGen/DataLayout.hpp"
#include "vast/CodeGen/CodeGenFunction.hpp"

#include "vast/CodeGen/DefaultTypeVisitor.hpp"
#include "vast/CodeGen/FallBackVisitor.hpp"

namespace vast::cg
{
    //
    // Module Context
    //

    owning_module_ref module_context::freeze() { return std::move(mod); }

    void set_target_triple(owning_module_ref &mod, std::string triple) {
        mlir::OpBuilder bld(mod.get());
        auto attr = bld.getAttr< mlir::StringAttr >(triple);
        mod.get()->setAttr(core::CoreDialect::getTargetTripleAttrName(), attr);
    }

    void set_source_language(owning_module_ref &mod, source_language lang) {
        mlir::OpBuilder bld(mod.get());
        auto attr = bld.getAttr< core::SourceLanguageAttr >(lang);
        mod.get()->setAttr(core::CoreDialect::getLanguageAttrName(), attr);
    }

    operation get_global_value(const module_context *ctx, clang_global decl) {
        return get_global_value(ctx, get_mangled_name(ctx, decl));
    }

    operation get_global_value(const module_context *ctx, mangled_name_ref name) {
        if (auto global = mlir::SymbolTable::lookupSymbolIn(ctx->mod.get(), name.name))
            return global;
        return {};
    }

    string_ref get_path_to_source(acontext_t &actx) {
        // Set the module name to be the name of the main file. TranslationUnitDecl
        // often contains invalid source locations and isn't a reliable source for the
        // module location.
        auto main_file_id = actx.getSourceManager().getMainFileID();
        const auto &main_file = *actx.getSourceManager().getFileEntryForID(main_file_id);
        return main_file.tryGetRealPathName();
    }

    //
    // Module Generator
    //

    namespace detail
    {
        std::pair< loc_t, std::string > module_loc_name(mcontext_t &mctx, acontext_t &actx) {
            if (auto path = get_path_to_source(actx); !path.empty()) {
                return { mlir::FileLineColLoc::get(&mctx, path, 0, 0), path.str() };
            }

            return { mlir::UnknownLoc::get(&mctx), "unknown" };
        }
    } // namespace detail

    owning_module_ref mk_module(acontext_t &actx, mcontext_t &mctx) {
        auto [loc, name] = detail::module_loc_name(mctx, actx);
        auto mod = owning_module_ref(vast_module::create(loc));
        mod->setSymName(name);
        return mod;
    }

    owning_module_ref mk_module_with_attrs(acontext_t &actx, mcontext_t &mctx, source_language lang) {
        auto mod = mk_module(actx, mctx);

        set_target_triple(mod, actx.getTargetInfo().getTriple().str());
        set_source_language(mod, lang);

        return mod;
    }

    const target_info &get_target_info(const module_context *mod) {
        return mod->actx.getTargetInfo();
    }

    const std::string &get_module_name_hash(const module_context *mod) {
        /* FIXME for UniqueInternalLinkageNames */
        VAST_UNIMPLEMENTED;
    }

    mangled_name_ref get_mangled_name(const module_context *mod, clang_global decl) {
        return mod->mangler.get_mangled_name(decl, get_target_info(mod), "" /* get_module_name_hash(mod) */);
    }

    void module_generator::emit(clang::DeclGroupRef decls) {
        for (auto &decl : decls) { emit(decl); }
    }

    void module_generator::emit(clang::Decl *decl) {
        switch (decl->getKind()) {
            case clang::Decl::Kind::Typedef:
                return emit(cast<clang::TypedefDecl>(decl));
            case clang::Decl::Kind::Enum:
                return emit(cast<clang::EnumDecl>(decl));
            case clang::Decl::Kind::Record:
                return emit(cast<clang::RecordDecl>(decl));
            case clang::Decl::Kind::Function:
                return emit(cast<clang::FunctionDecl>(decl));
            case clang::Decl::Kind::Var:
                return emit(cast<clang::VarDecl>(decl));
            default:
                VAST_FATAL("unhandled decl kind: {}", decl->getDeclKindName());
        }
    }

    void module_generator::emit(clang_global */* decl */) {
        VAST_UNIMPLEMENTED;
    }

    void module_generator::emit(clang::TypedefDecl */* decl */) {
        VAST_UNIMPLEMENTED;
    }

    void module_generator::emit(clang::EnumDecl */* decl */) {
        VAST_UNIMPLEMENTED;
    }

    void module_generator::emit(clang::RecordDecl */* decl */) {
        VAST_UNIMPLEMENTED;
    }

    void module_generator::emit(clang::FunctionDecl *decl) {
        generate_child< function_generator >(decl);
    }

    void module_generator::emit(clang::VarDecl */* decl */) {
        VAST_UNIMPLEMENTED;
    }

    // TODO this should not be needed the data layout should be emitted from cached types directly
    dl::DataLayoutBlueprint emit_datalayout_blueprint(
        const clang::ASTContext &actx, const cached_default_type_visitor &type_visitor
    ) {
        dl::DataLayoutBlueprint dl;

        auto store_layout = [&] (const clang_type *orig, mlir_type vast_type) {
            if (orig->isFunctionType()) {
                return;
            }

            // skip forward declared types
            if (auto tag = orig->getAsTagDecl()) {
                if (!tag->isThisDeclarationADefinition()) {
                    return;
                }
            }

            dl.try_emplace(vast_type, orig, actx);
        };

        for (auto [orig, vast_type] : type_visitor.cache) {
            store_layout(orig, vast_type);
        }

        for (auto [qual_type, vast_type] : type_visitor.qual_cache) {
            auto [orig, quals] = qual_type.split();
            store_layout(orig, vast_type);
        }

        return dl;
    }

    void module_generator::emit_data_layout() {
        auto mctx = mod.get()->getContext();

        auto default_visitor = [&] {
            if (auto fallback = dynamic_cast< fallback_visitor* >(visitor.raw())) {
                return fallback->front();
            }

            return visitor;
        } ();


        if (auto type_visitor = dynamic_cast< cached_default_type_visitor* >(default_visitor.raw())) {
            auto dl = emit_datalayout_blueprint(actx, *type_visitor);
            vast::cg::emit_data_layout(*mctx, mod, dl);
        }
    }

    void module_generator::finalize() {
        scope_context::finalize();
        emit_data_layout();
    }

    bool module_generator::verify() {
        return mlir::verify(mod.get()).succeeded();
    }

} // namespace vast::cg

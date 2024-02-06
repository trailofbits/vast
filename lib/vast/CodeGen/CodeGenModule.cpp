// Copyright (c) 2024-present, Trail of Bits, Inc.

#include "vast/CodeGen/CodeGenModule.hpp"

VAST_RELAX_WARNINGS
#include <clang/Basic/SourceManager.h>
#include <mlir/IR/Verifier.h>
VAST_UNRELAX_WARNINGS

#include "vast/CodeGen/DataLayout.hpp"

namespace vast::cg
{
    //
    // Module Context
    //

    owning_module_ref module_context::freeze() {
        VAST_ASSERT(!frozen);
        frozen = true;
        return std::move(mod);
    }

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


    //
    // Module Generator
    //

    namespace detail
    {
        std::pair< loc_t, std::string > module_loc_name(mcontext_t &mctx, acontext_t &actx) {
            // Set the module name to be the name of the main file. TranslationUnitDecl
            // often contains invalid source locations and isn't a reliable source for the
            // module location.
            auto main_file_id = actx.getSourceManager().getMainFileID();
            const auto &main_file = *actx.getSourceManager().getFileEntryForID(main_file_id);
            auto path = main_file.tryGetRealPathName();
            if (!path.empty()) {
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

    void module_generator::emit(clang::DeclGroupRef decls) {
        VAST_ASSERT(!frozen);
        for (auto &decl : decls) {
            emit(decl);
        }
    }

    void module_generator::emit(clang::Decl *decl) {
        VAST_ASSERT(!frozen);
        switch (decl->getKind()) {
            case clang::Decl::Kind::Typedef:
                emit(cast<clang::TypedefDecl>(decl));
                break;
            case clang::Decl::Kind::Enum:
                emit(cast<clang::EnumDecl>(decl));
                break;
            case clang::Decl::Kind::Record:
                emit(cast<clang::RecordDecl>(decl));
                break;
            case clang::Decl::Kind::Function:
                emit(cast<clang::FunctionDecl>(decl));
                break;
            case clang::Decl::Kind::Var:
                emit(cast<clang::VarDecl>(decl));
                break;
            default:
                VAST_FATAL("unhandled decl kind: {}", decl->getDeclKindName());
        }
    }

    void module_generator::emit(clang::GlobalDecl */* decl */) {
        VAST_ASSERT(!frozen);
        VAST_UNIMPLEMENTED;
    }

    void module_generator::emit(clang::TypedefDecl */* decl */) {
        VAST_ASSERT(!frozen);
        VAST_UNIMPLEMENTED;
    }

    void module_generator::emit(clang::EnumDecl */* decl */) {
        VAST_ASSERT(!frozen);
        VAST_UNIMPLEMENTED;
    }

    void module_generator::emit(clang::RecordDecl */* decl */) {
        VAST_ASSERT(!frozen);
        VAST_UNIMPLEMENTED;
    }

    void module_generator::emit(clang::FunctionDecl */* decl */) {
        VAST_ASSERT(!frozen);
        VAST_UNIMPLEMENTED;
    }

    void module_generator::emit(clang::VarDecl */* decl */) {
        VAST_ASSERT(!frozen);
        VAST_UNIMPLEMENTED;
    }

    void module_generator::emit_data_layout() {
        VAST_ASSERT(!frozen);
        auto mctx = mod.get()->getContext();
        vast::cg::emit_data_layout(*mctx, mod, dl);
    }

    void module_generator::finalize() {
        VAST_ASSERT(!frozen);
        emit_data_layout();
    }

    bool module_generator::verify() {
        VAST_ASSERT(!frozen);
        return mlir::verify(mod.get()).succeeded();
    }

} // namespace vast::cg

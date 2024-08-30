// Copyright (c) 2022-present, Trail of Bits, Inc.

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <clang/AST/GlobalDecl.h>
#include <clang/Basic/SourceManager.h>
#include <clang/Basic/TargetInfo.h>
#include <mlir/IR/Verifier.h>
VAST_UNRELAX_WARNINGS

#include "vast/CodeGen/AttrVisitorProxy.hpp"
#include "vast/CodeGen/CodeGenDriver.hpp"
#include "vast/CodeGen/CodeGenFunction.hpp"
#include "vast/CodeGen/CodeGenModule.hpp"
#include "vast/CodeGen/CodeGenVisitorList.hpp"
#include "vast/CodeGen/DataLayout.hpp"
#include "vast/CodeGen/DefaultCodeGenPolicy.hpp"
#include "vast/CodeGen/DefaultMetaGenerator.hpp"
#include "vast/CodeGen/DefaultVisitor.hpp"
#include "vast/CodeGen/IdMetaGenerator.hpp"
#include "vast/CodeGen/InvalidMetaGenerator.hpp"
#include "vast/CodeGen/TypeCachingProxy.hpp"
#include "vast/CodeGen/UnreachableVisitor.hpp"
#include "vast/CodeGen/UnsupportedVisitor.hpp"

#include "vast/Dialect/Core/CoreOps.hpp"

namespace vast::cg {

    void driver::emit(clang::DeclGroupRef decls) { generator.emit(decls); }

    void driver::emit(clang::Decl *decl) { generator.emit(decl); }

    owning_mlir_module_ref driver::freeze() { return std::move(top); }

    // TODO this should not be needed the data layout should be emitted from cached types directly
    dl::DataLayoutBlueprint emit_data_layout_blueprint(
        const acontext_t &actx, const type_caching_proxy &types
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

        for (auto [orig, vast_type] : types.cache) {
            store_layout(orig, vast_type);
        }

        for (auto [qual_type, vast_type] : types.qual_cache) {
            auto [orig, quals] = qual_type.split();
            store_layout(orig, vast_type);
        }

        return dl;
    }

    void driver::finalize() {
        generator.finalize();

        emit_data_layout();

        if (enabled_verifier) {
            if (!verify()) {
                VAST_FATAL("codegen: module verification error before running vast passes");
            }
        }
    }

    void driver::emit_data_layout() {
        auto list = std::dynamic_pointer_cast< visitor_list >(visitor);
        for (auto node = list->head; node; node = node->next) {
            if (auto types = std::dynamic_pointer_cast< type_caching_proxy >(node)) {
                ::vast::cg::emit_data_layout(mctx, mod, emit_data_layout_blueprint(actx, *types));
            }
        }
    }

    bool driver::verify() { return mlir::verify(mod).succeeded(); }

    std::unique_ptr< codegen_builder > mk_codegen_builder(mcontext_t &mctx) {
        return std::make_unique< codegen_builder >(&mctx);
    }

    std::shared_ptr< meta_generator > mk_meta_generator(
        acontext_t *actx, mcontext_t *mctx, const cc::vast_args &vargs
    ) {
        if (vargs.has_option(cc::opt::locs_as_meta_ids)) {
            return std::make_shared< id_meta_gen >(actx, mctx);
        }
        return std::make_shared< default_meta_gen >(actx, mctx);
    }

    std::shared_ptr< meta_generator > mk_invalid_meta_generator(mcontext_t *mctx) {
        return std::make_shared< invalid_meta_gen >(mctx);
    }

    std::shared_ptr< symbol_generator > mk_symbol_generator(acontext_t &actx) {
        return std::make_shared< default_symbol_generator >(actx.createMangleContext());
    }

    std::shared_ptr< codegen_policy > mk_codegen_policy(cc::action_options &opts) {
        return std::make_shared< default_policy >(opts);
    }

    std::unique_ptr< driver > mk_default_driver(
        cc::action_options &opts, const cc::vast_args &vargs, acontext_t &actx, mcontext_t &mctx
    ) {
        auto bld = mk_codegen_builder(mctx);

        // setup visitor list
        const bool enable_unsupported = !vargs.has_option(cc::opt::disable_unsupported);

        auto mg = mk_meta_generator(&actx, &mctx, vargs);
        auto invalid_mg = mk_invalid_meta_generator(&mctx);
        auto sg = mk_symbol_generator(actx);
        auto policy = mk_codegen_policy(opts);

        auto visitors = std::make_shared< visitor_list >()
            | as_node_with_list_ref< attr_visitor_proxy >()
            | as_node< type_caching_proxy >()
            | as_node_with_list_ref< default_visitor >(
                mctx, actx, *bld, std::move(mg), std::move(sg), std::move(policy)
            )
            | optional(enable_unsupported,
                as_node_with_list_ref< unsup_visitor >(
                    mctx, *bld, std::move(invalid_mg)
                )
            )
            | as_node< unreach_visitor >();

        // setup driver
        auto drv = std::make_unique< driver >(
            actx, mctx, std::move(bld), visitors
        );

        drv->enable_verifier(!vargs.has_option(cc::opt::disable_vast_verifier));
        return drv;
    }

    void set_target_triple(core::module mod, std::string triple) {
        mlir::OpBuilder bld(mod);
        auto attr = bld.getAttr< mlir::StringAttr >(triple);
        mod->setAttr(core::CoreDialect::getTargetTripleAttrName(), attr);
    }

    void set_source_language(core::module mod, cc::source_language lang) {
        mlir::OpBuilder bld(mod);
        auto attr = bld.getAttr< core::SourceLanguageAttr >(lang);
        mod->setAttr(core::CoreDialect::getLanguageAttrName(), attr);
    }

    string_ref get_path_to_source(acontext_t &actx) {
        // Set the module name to be the name of the main file. TranslationUnitDecl
        // often contains invalid source locations and isn't a reliable source for the
        // module location.
        auto main_file_id     = actx.getSourceManager().getMainFileID();
        const auto &main_file = *actx.getSourceManager().getFileEntryForID(main_file_id);
        return main_file.tryGetRealPathName();
    }

    namespace detail {
        std::pair< loc_t, std::string > module_loc_name(mcontext_t &mctx, acontext_t &actx) {
            // TODO use meta generator
            if (auto path = get_path_to_source(actx); !path.empty()) {
                return { mlir::FileLineColLoc::get(&mctx, path, 0, 0), path.str() };
            }
            return { mlir::UnknownLoc::get(&mctx), "unknown" };
        }
    } // namespace detail

    owning_mlir_module_ref mk_wrapping_module(mcontext_t &mctx) {
        return mlir::ModuleOp::create(mlir::UnknownLoc::get(&mctx));
    }

    core::module mk_module(acontext_t &actx, mlir_module top) {
        mlir::OpBuilder bld(top);
        bld.setInsertionPointToStart(top.getBody());

        // TODO use symbol generator
        auto mctx = top.getContext();
        auto [loc, name] = detail::module_loc_name(*mctx, actx);
        return bld.create< core::module >(loc, name);
    }

    core::module mk_module_with_attrs(
        acontext_t &actx, mlir_module top,
        cc::source_language lang
    ) {
        auto mod = mk_module(actx, top);

        set_target_triple(mod, actx.getTargetInfo().getTriple().str());
        set_source_language(mod, lang);

        return mod;
    }

} // namespace vast::cg

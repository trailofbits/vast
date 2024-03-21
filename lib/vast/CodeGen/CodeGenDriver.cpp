// Copyright (c) 2022-present, Trail of Bits, Inc.

#include "vast/CodeGen/CodeGenDriver.hpp"

VAST_RELAX_WARNINGS
#include <clang/AST/GlobalDecl.h>
#include <clang/Basic/TargetInfo.h>
VAST_UNRELAX_WARNINGS

#include "vast/Frontend/Options.hpp"

#include "vast/CodeGen/CodeGenVisitor.hpp"
#include "vast/CodeGen/DefaultVisitor.hpp"
#include "vast/CodeGen/UnreachableVisitor.hpp"
#include "vast/CodeGen/UnsupportedVisitor.hpp"

namespace vast::cg {
    void driver::emit(clang::DeclGroupRef decls) { generator.emit(decls); }
    void driver::emit(clang::Decl *decl)         { generator.emit(decl); }

    owning_module_ref driver::freeze() { return generator.freeze(); }

    void driver::finalize(const cc::vast_args &vargs) {
        if (!vargs.has_option(cc::opt::disable_vast_verifier)) {
            if (!generator.verify()) {
                VAST_FATAL("codegen: module verification error before running vast passes");
            }
        }

        generator.finalize();
    }

    std::unique_ptr< codegen_builder > mk_codegen_builder(mcontext_t &mctx) {
        return std::make_unique< codegen_builder >(&mctx);
    }

    std::unique_ptr< meta_generator > mk_meta_generator(
        acontext_t &actx, mcontext_t &mctx, const cc::vast_args &vargs
    ) {
        if (vargs.has_option(cc::opt::locs_as_meta_ids))
            return std::make_unique< id_meta_gen >(&actx, &mctx);
        return std::make_unique< default_meta_gen >(&actx, &mctx);
    }

    std::unique_ptr< mcontext_t > mk_mcontext() {
        auto mctx = std::make_unique< mcontext_t >();
        mlir::registerAllDialects(*mctx);
        vast::registerAllDialects(*mctx);
        mctx->loadAllAvailableDialects();
        return mctx;
    }

    std::unique_ptr< visitor_base > mk_visitor(
          const cc::vast_args &vargs
        , mcontext_t &mctx
        , codegen_builder &bld
        , meta_generator &mg
        , symbol_generator &sg
    ) {
        // TODO pick the right visitors based on the command line args
        fallback_visitor::visitor_stack visitors;
        visitors.push_back(std::make_unique< default_visitor >(mctx, bld, mg, sg));
        visitors.push_back(std::make_unique< unsup_visitor   >(mctx, bld, mg, sg));
        visitors.push_back(std::make_unique< unreach_visitor >(mctx, mg, sg));
        return std::make_unique< codegen_visitor >(mctx, mg, sg, std::move(visitors));
    }

    std::unique_ptr< symbol_generator > mk_symbol_generator(acontext_t &actx) {
        return std::make_unique< default_symbol_mangler >(actx.createMangleContext());
    }

    std::unique_ptr< driver > mk_driver(cc::action_options &opts, const cc::vast_args &vargs, acontext_t &actx) {
        auto mctx    = mk_mcontext();
        auto bld     = mk_codegen_builder(*mctx);
        auto mg      = mk_meta_generator(actx, *mctx, vargs);
        auto sg      = mk_symbol_generator(actx);
        auto visitor = mk_visitor(vargs, *mctx, *bld, *mg, *sg);

        return std::make_unique< driver >(
            actx, std::move(mctx), std::move(bld), std::move(mg), std::move(sg), std::move(visitor)
        );
    }

    module_generator driver::mk_module_generator() {
        return module_generator(
            actx, *mctx, cc::get_source_language(actx.getLangOpts()), *bld, visitor_view(*visitor), scopes
        );
    }

} // namespace vast::cg

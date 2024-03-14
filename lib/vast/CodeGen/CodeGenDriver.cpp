// Copyright (c) 2022-present, Trail of Bits, Inc.

#include "vast/CodeGen/CodeGenDriver.hpp"

VAST_RELAX_WARNINGS
#include <clang/AST/GlobalDecl.h>
#include <clang/Basic/TargetInfo.h>
VAST_UNRELAX_WARNINGS

#include "vast/Frontend/Options.hpp"

#include "vast/CodeGen/CodeGenVisitor.hpp"

namespace vast::cg
{
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

    std::unique_ptr< visitor_base > mk_visitor(const cc::vast_args &vargs) {
        // TODO pick the right visitor based on the command line args
        return std::make_unique< default_visitor >();
    }

} // namespace vast::cg

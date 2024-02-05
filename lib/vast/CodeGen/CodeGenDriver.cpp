// Copyright (c) 2022-present, Trail of Bits, Inc.

#include "vast/CodeGen/CodeGenDriver.hpp"

VAST_RELAX_WARNINGS
#include <clang/AST/GlobalDecl.h>
#include <clang/Basic/TargetInfo.h>
VAST_UNRELAX_WARNINGS

#include "vast/Frontend/Options.hpp"

namespace vast::cg
{
    void driver::emit(clang::DeclGroupRef decls) {
        generator.emit(decls);
    }

    void driver::emit(clang::Decl *decl) {
        generator.emit(decl);
    }

    owning_module_ref driver::freeze() {
        return generator.freeze();
    }

    void driver::finalize(const cc::vast_args &vargs) {
        if (!vargs.has_option(cc::opt::disable_vast_verifier)) {
            if (!generator.verify()) {
                VAST_FATAL("codegen: module verification error before running vast passes");
            }
        }

        generator.finalize();
    }

    meta_generator_ptr driver::mk_meta_generator(const cc::vast_args &vargs) {
        if (vargs.has_option(cc::opt::locs_as_meta_ids))
            return std::make_unique< id_meta_gen >(&actx, mctx.get());
        return std::make_unique< default_meta_gen >(&actx, mctx.get());
    }

} // namespace vast::cg

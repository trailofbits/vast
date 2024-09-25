// Copyright (c) 2024-present, Trail of Bits, Inc.

#include "vast/Util/Warnings.hpp"

#include "vast/Conversion/Passes.hpp"

VAST_RELAX_WARNINGS
#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
VAST_UNRELAX_WARNINGS

#include "../PassesDetails.hpp"

#include "vast/Util/Common.hpp"
#include "vast/Conversion/Common/Mixins.hpp"
#include "vast/Conversion/Common/Patterns.hpp"

namespace vast::conv {
    namespace pattern {
        struct move_static_local : operation_conversion_pattern< hl::VarDeclOp >
        {
            using base = operation_conversion_pattern< hl::VarDeclOp >;
            using base::base;

            using adaptor_t = hl::VarDeclOp::Adaptor;

            logical_result matchAndRewrite(
                hl::VarDeclOp op, adaptor_t adaptor, conversion_rewriter &rewriter
            ) const override {
                auto parent_fn = op->getParentOfType< mlir::FunctionOpInterface >();
                if (!parent_fn)
                    return mlir::failure();

                auto guard = insertion_guard(rewriter);
                auto &module_block = parent_fn->getParentOfType< core::ModuleOp >().getBody().front();
                rewriter.setInsertionPoint(&module_block, module_block.begin());

                auto new_decl = rewriter.create< hl::VarDeclOp >(
                        op.getLoc(),
                        op.getType(),
                        (parent_fn.getName() + "." + op.getSymName()).str(),
                        op.getStorageClass(),
                        op.getThreadStorageClass(),
                        std::optional(core::GlobalLinkageKind::InternalLinkage)
                );

                // Save current context informationinto the op to make sure the information stays valid
                new_decl->setAttr("context", core::DeclContextKindAttr::get(op.getContext(), op.getDeclContextKind()));

                new_decl.getInitializer().takeBody(op.getInitializer());
                new_decl.getAllocationSize().takeBody(op.getAllocationSize());

                rewriter.eraseOp(op);

                return mlir::success();
            }

            static void legalize(conversion_target &trg) {
                trg.addDynamicallyLegalOp< hl::VarDeclOp >([] (hl::VarDeclOp op) {
                    return !(op.isStaticLocal() && op->getParentOfType< mlir::FunctionOpInterface >());
                });
            }
        };

        struct update_decl_ref : operation_conversion_pattern< hl::DeclRefOp >
        {
            using base = operation_conversion_pattern< hl::DeclRefOp >;
            using base::base;

            using adaptor_t = hl::DeclRefOp::Adaptor;

            logical_result matchAndRewrite(
                hl::DeclRefOp op, adaptor_t adaptor, conversion_rewriter &rewriter
            ) const override {
                auto var = core::symbol_table::lookup< core::var_symbol >(op, op.getName());
                if (auto decl_storage = mlir::dyn_cast< core::DeclStorageInterface>(var)) {
                    auto parent_fn = op->getParentOfType< mlir::FunctionOpInterface >();

                    if (!parent_fn || !decl_storage.isStaticLocal())
                        return mlir::failure();

                    rewriter.replaceOpWithNewOp< hl::DeclRefOp >(
                            op,
                            op.getType(),
                            (parent_fn.getName() + "." + op.getName()).str()
                    );
                    return mlir::success();
                }
                return mlir::failure();
            }

            static void legalize(conversion_target &trg) {
                trg.addDynamicallyLegalOp< hl::DeclRefOp >([&](hl::DeclRefOp op) {
                    auto var = core::symbol_table::lookup< core::var_symbol >(op, op.getName());
                    if (auto decl_storage = mlir::dyn_cast< core::DeclStorageInterface>(var)) {
                        return !(decl_storage.isStaticLocal() && var->getParentOfType< mlir::FunctionOpInterface >());
                    }
                    return (bool)var;
                });
            }
        };
    }

    struct EvictStaticLocalsPass : ConversionPassMixin< EvictStaticLocalsPass, EvictStaticLocalsBase >
    {
        using base = ConversionPassMixin< EvictStaticLocalsPass, EvictStaticLocalsBase >;

        static conversion_target create_conversion_target(mcontext_t &mctx) {
            return conversion_target(mctx);
        }

        static void populate_conversions(auto &cfg) {
            base::populate_conversions< pattern::move_static_local >(cfg);
            base::populate_conversions< pattern::update_decl_ref >(cfg);
        }
    };
} // namespace vast::conv

std::unique_ptr< mlir::Pass > vast::createEvictStaticLocalsPass() {
    return std::make_unique< vast::conv::EvictStaticLocalsPass >();
}

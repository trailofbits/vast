// Copyright (c) 2023-present, Trail of Bits, Inc.

#include "vast/Conversion/Passes.hpp"

VAST_RELAX_WARNINGS
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>

#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Rewrite/FrozenRewritePatternSet.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
VAST_UNRELAX_WARNINGS

#include "vast/Conversion/Common/Passes.hpp"
#include "vast/Conversion/Common/Patterns.hpp"

#include "vast/Util/Common.hpp"
#include "vast/Util/DialectConversion.hpp"

#include "vast/Conversion/Common/Rewriter.hpp"

#include "vast/Dialect/HighLevel/HighLevelDialect.hpp"

#include "../PassesDetails.hpp"

namespace vast::conv
{
    namespace
    {
        using type_map = std::map< mlir_type, mlir_type >;

        struct typedef_resolver
        {
            vast_module mod;

            mlir_type nested_type(mlir_type type)
            {
                return hl::getBottomTypedefType(type, mod);
            }

            auto get_mapper()
            {
                return [=](hl::ElaboratedType type) -> std::optional< mlir_type >
                {
                    return { nested_type(type) };
                };
            }

            auto do_conversion(mlir_type type)
            {
                if (auto subelements = mlir::dyn_cast< mlir::SubElementTypeInterface >(type))
                {
                    return subelements.replaceSubElements(get_mapper());
                }
                return nested_type(type);
            }

            void fixup_entry_block(mlir::Block *block)
            {
                for (std::size_t i = 0; i < block->getNumArguments(); ++i)
                {
                    auto arg = block->getArgument(i);
                    arg.setType(do_conversion(arg.getType()));
                }
            }

            auto get_process()
            {
                return [=](mlir::Operation *nested)
                {
                    // Special case for functions, it may be that we can unify it with
                    // the generic one.
                    if (auto fn = mlir::dyn_cast< hl::FuncOp >(nested))
                    {
                        auto type = fn.getFunctionType();
                        fn.setType(do_conversion(type));
                        if (nested->getNumRegions() != 0)
                            fixup_entry_block(&*nested->getRegions().begin()->begin());

                        return;
                    }

                    // Generic conversion of only result types.
                    std::vector< mlir_type > new_types;
                    for (auto res : nested->getResultTypes())
                    {
                        auto trg_type = do_conversion(res);
                        new_types.push_back(trg_type);
                    }



                    if (nested->getNumRegions() != 0)
                        fixup_entry_block(&*nested->getRegions().begin()->begin());

                    for (std::size_t i = 0; i < new_types.size(); ++i)
                        nested->getResult(i).setType(new_types[i]);
                };
            }

            void run(mlir::Operation *op)
            {
                op->walk(get_process());
            }
        };

    } // namespace

    struct ResolveTypeDefs : ResolveTypeDefsBase< ResolveTypeDefs >
    {
        using base = ResolveTypeDefsBase< ResolveTypeDefsBase >;

        void runOnOperation() override
        {
            auto op = this->getOperation();
            typedef_resolver{ op }.run(op.getOperation());

        }

    };

} // namespace vast::conv

std::unique_ptr< mlir::Pass > vast::createResolveTypeDefsPass()
{
    return std::make_unique< vast::conv::ResolveTypeDefs >();
}

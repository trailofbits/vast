// Copyright (c) 2021-present, Trail of Bits, Inc.

#include "vast/Dialect/HighLevel/Passes.hpp"

VAST_RELAX_WARNINGS
#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>

#include <mlir/Rewrite/FrozenRewritePatternSet.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
VAST_UNRELAX_WARNINGS

#include <gap/core/memoize.hpp>
#include <gap/core/generator.hpp>

#include <vast/Dialect/HighLevel/HighLevelUtils.hpp>
#include <vast/Interfaces/AggregateTypeDefinitionInterface.hpp>

#include <vast/Conversion/Common/Passes.hpp>
#include <vast/Util/TypeUtils.hpp>

#include "PassesDetails.hpp"

#include <unordered_set>

namespace vast::hl {

#if !defined(NDEBUG)
    constexpr bool debug_ude_pass = false;
    #define VAST_UDE_DEBUG(...) VAST_REPORT_WITH_PREFIX_IF(debug_ude_pass, "[UDE] ", __VA_ARGS__)
#else
    #define VAST_UDE_DEBUG(...)
#endif

    template< typename ... args_t >
    bool is_one_of( operation op ) { return ( mlir::isa< args_t >( op ) || ... ); }

    constexpr auto keep_only_if_used = [] (auto, auto) { /* do not keep by default */ return false; };

    struct UDE : UDEBase< UDE >
    {
        using base         = UDEBase< UDE >;
        using operations_t = std::vector< operation >;

        operations_t unused;

        // We interupt the walk of users as we know we need to keep the
        // operation because it is used in other kept operation (user)
        static inline auto keep_operation = walk_result::interrupt();
        // If we andvance the walk, we know that the operation is not used
        static inline auto drop_operation = walk_result::advance();

        //
        // aggregate unused definition elimination
        //
        bool keep(aggregate_interface op, auto scope) const { return keep_only_if_used(op, scope); }

        walk_result filtered_users(auto op, auto scope, auto &&yield) const {
            return hl::users(op, scope, [yield = std::forward< decltype(yield) >(yield)] (operation op) {
                // skip module as it always contains value use
                return mlir::isa< vast_module >(op) ? walk_result::advance() : yield(op);
            });
        }

        // keep field if its parent is kept
        bool keep(hl::FieldDeclOp op, auto scope) const {
            return keep(op.getParentAggregate(), scope);
        }

        template< typename yield_t >
        walk_result filtered_users(hl::FieldDeclOp decl, auto scope, yield_t &&yield) const {
            return filtered_users(decl.getParentAggregate(), scope, std::forward< yield_t >(yield));
        }

        //
        // typedef/decl unused definition elimination
        //
        bool keep(hl::TypeDefOp op, auto scope) const { return keep_only_if_used(op, scope); }
        bool keep(hl::TypeDeclOp op, auto scope) const { return keep_only_if_used(op, scope); }

        //
        // function unused definition elimination
        //
        bool keep(hl::FuncOp op, auto scope) const { return !op.isDeclaration(); }

        void process(operation op, vast_module mod) {
            std::unordered_set< operation > seen;
            // we keep the operation if it is resolved to be kept or any of its
            // users is marked as to be kept, otherwise we mark it as unused and
            // erase it
            auto keep = [this, mod, &seen](auto &self, operation op) {
                auto dispatch = [this, mod, &self, &seen] (auto op) {
                    if (const auto [_, inserted] = seen.insert(op); !inserted) {
                        // Already processed, the operation has recursive dependency.
                        // We can safely return false here, as some other user
                        // needs to determine if the operation is to be kept.
                        return false;
                    }
                    VAST_UDE_DEBUG("processing: {0}", *op);
                    if (this->keep(op, mod))
                        return true;

                    auto result = filtered_users(op, mod, [&](auto user) -> walk_result {
                        auto keep_user = self(user);
                        VAST_UDE_DEBUG("user: {0} : {1}", *user, keep_user ? "keep" : "drop");
                        // if any user is to be kept, keep the operation
                        return keep_user ? keep_operation : drop_operation;
                    });

                    return result == keep_operation;
                };

                return llvm::TypeSwitch< operation, bool >(op)
                    .Case([&](aggregate_interface op) { return dispatch(op); })
                    .Case([&](hl::FieldDeclOp op)     { return dispatch(op); })
                    .Case([&](hl::TypeDefOp op)       { return dispatch(op); })
                    .Case([&](hl::TypeDeclOp op)      { return dispatch(op); })
                    .Case([&](hl::FuncOp op)          { return dispatch(op); })
                    .Default([&](operation)           { return true; });
            };

            auto to_keep = gap::recursive_memoize<bool(operation)>(keep);

            if (!to_keep(op)) {
                VAST_UDE_DEBUG("unused: {0}", *op);
                unused.push_back(op);
            }
        }

        void process(vast_module mod) {
            for (auto &op : mod.getOps()) {
                process(&op, mod);
            }
        }

        void runOnOperation() override {
            process(getOperation());
            for (auto &op : unused) {
                op->erase();
            }
        }
    };

    std::unique_ptr< mlir::Pass > createUDEPass() { return std::make_unique< UDE >(); }
} // namespace vast::hl

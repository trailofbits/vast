// Copyright (c) 2021-present, Trail of Bits, Inc.

#include "vast/Dialect/HighLevel/Passes.hpp"

VAST_RELAX_WARNINGS
#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>

#include <mlir/Rewrite/FrozenRewritePatternSet.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
VAST_UNRELAX_WARNINGS

#include <gap/coro/generator.hpp>

#include <vast/Dialect/HighLevel/HighLevelUtils.hpp>
#include <vast/Dialect/Core/Interfaces/TypeDefinitionInterface.hpp>

#include <vast/Conversion/Common/Mixins.hpp>

#include <vast/Util/Attribute.hpp>
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

    constexpr bool keep_only_if_used = false;

    template< typename yield_t >
    walk_result users(hl::FieldDeclOp decl, auto scope, yield_t &&yield) {
        return hl::users(decl.getAggregate(), scope, std::forward< yield_t >(yield));
    }

    struct UDE : UDEBase< UDE >
    {
        using base = UDEBase< UDE >;

        std::unordered_set< operation > unused_cached;

        bool keep(core::aggregate_interface op, auto scope) const {
            return keep_only_if_used;
        }

        bool keep(hl::TypeDefOp op, auto scope)       const { return keep_only_if_used; }
        bool keep(hl::TypeDeclOp op, auto scope)      const { return keep_only_if_used; }

        // Mark field to be kept if the parent aggregate is kept
        bool keep(hl::FieldDeclOp op, auto scope) const { return keep(op.getAggregate(), scope); }

        bool keep(hl::FuncOp op, auto scope) const {
            return !op.isDeclaration() && !util::has_attr< hl::AlwaysInlineAttr >(op);
        }

        bool keep(hl::VarDeclOp op, auto scope) const {
            VAST_CHECK(!op.hasExternalStorage() || op.getInitializer().empty(), "extern variable with initializer");
            return !op.hasExternalStorage();
        }

        bool is_unused_impl(auto op, auto scope, auto &seen) {
            if (keep(op, scope)) {
                VAST_UDE_DEBUG("keep: {0}", *op);
                return false;
            }

            auto result = hl::users(op, scope, [&](operation user) -> walk_result {
                VAST_UDE_DEBUG("user: {0} of {1}", *user, *op);
                // Ignore top-level use
                if (user == scope) {
                    return walk_result::advance();
                }

                if (is_unused(user, scope, seen)) {
                    VAST_UDE_DEBUG("unused user: {0}", *user);
                    return walk_result::advance();
                }

                // If the user is an always inlined function that is not used,
                // we mark it as unused.
                if (auto parent = user->template getParentOfType< hl::FuncOp >()) {
                    if (is_unused(parent, scope, seen)) {
                        VAST_UDE_DEBUG("user in always inlined function: {0}", *user);
                        return walk_result::advance();
                    }
                }

                // We interrupt the walk if the operation is used
                return walk_result::interrupt();
            });

            // Operation is used if the walk was interrupted so we need to keep it.
            return !result.wasInterrupted();
        }

        bool is_unused(operation op, auto scope, auto &seen) {
            if (unused_cached.contains(op)) {
                VAST_UDE_DEBUG("cached: {0}", *op);
                return true;
            }

            if (const auto [_, inserted] = seen.insert(op); !inserted) {
                VAST_UDE_DEBUG("recursive: {0}", *op);
                // Already processed, the operation has recursive dependency.
                // We can safely return true here, as some other user
                // needs to determine if the operation is to be kept.
                return true;
            }

            VAST_UDE_DEBUG("processing: {0}", *op);
            bool result = llvm::TypeSwitch< operation, bool >(op)
                .Case([&](core::aggregate_interface op) {
                    return is_unused_impl(op, scope, seen);
                })
                .Case([&](hl::FieldDeclOp op)     { return is_unused_impl(op, scope, seen); })
                .Case([&](hl::TypeDefOp op)       { return is_unused_impl(op, scope, seen); })
                .Case([&](hl::TypeDeclOp op)      { return is_unused_impl(op, scope, seen); })
                .Case([&](hl::FuncOp op)          { return is_unused_impl(op, scope, seen); })
                .Case([&](hl::VarDeclOp op)       { return is_unused_impl(op, scope, seen); })
                .Default([&](operation)           { return false; });

            if (result) {
                unused_cached.insert(op);
            }

            return result;
        }

        std::vector< operation > gather_unused(auto scope) {
            std::vector< operation > unused_operations;
            for (auto &op : scope.getOps()) {
                std::unordered_set< operation > seen;
                if (is_unused(&op, scope, seen)) {
                    unused_operations.push_back(&op);
                }
            }

            return unused_operations;
        }

        void runOnOperation() override {
            auto mod = getOperation();
            auto unused = gather_unused(mod);

            llvm::DenseSet< mlir_type > unused_types;
            for (auto &op : unused) {
                if (auto td = mlir::dyn_cast< hl::TypeDefOp >(op)) {
                    unused_types.insert(td.getDefinedType());
                }

                if (auto agg = mlir::dyn_cast< core::aggregate_interface >(op)) {
                    unused_types.insert(agg.getDefinedType());
                }

                if (auto td = mlir::dyn_cast< hl::TypeDeclOp >(op)) {
                    unused_types.insert(td.getDefinedType());
                }
            }

            auto contains_unused_subtype = [&] (mlir_type type) {
                return contains_subtype(type, [&] (mlir_type sub) {
                    return unused_types.contains(sub);
                });
            };

            dl::filter_data_layout(mod, [&] (const auto &entry) {
                auto type = entry.getKey().template get< mlir_type >();
                return !contains_unused_subtype(type);
            });

            for (auto op : unused) {
                op->erase();
            }
        }
    };

    std::unique_ptr< mlir::Pass > createUDEPass() { return std::make_unique< UDE >(); }

} // namespace vast::hl

// Copyright (c) 2021-present, Trail of Bits, Inc.

#include "vast/Dialect/HighLevel/Passes.hpp"

VAST_RELAX_WARNINGS
#include <mlir/Analysis/DataLayoutAnalysis.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include <mlir/Transforms/Passes.h>
#include <gap/core/overloads.hpp>
VAST_UNRELAX_WARNINGS

#include "PassesDetails.hpp"

#include "vast/Dialect/Core/CoreOps.hpp"

#include "vast/Dialect/HighLevel/HighLevelOps.hpp"
#include "vast/Dialect/HighLevel/HighLevelUtils.hpp"
#include "vast/Dialect/LowLevel/LowLevelOps.hpp"

#include "vast/Conversion/TypeConverters/TypeConvertingPattern.hpp"

#include "vast/Util/Common.hpp"
#include "vast/Util/DialectConversion.hpp"
#include "vast/Util/Symbols.hpp"
#include "vast/Util/Terminator.hpp"

namespace vast::hl {

    struct enums_info {
        using const_decl = hl::EnumConstantOp;
        using enum_decl = hl::EnumDeclOp;

        llvm::DenseMap< llvm::StringRef, const_decl > id_to_value;
        llvm::DenseMap< llvm::StringRef, enum_decl > name_to_decl;

        enums_info(core::module mod) {
            collect_enum_values(mod);
        }

        void collect_enum_values(core::module root) {
            auto walker = [&](operation op) {
                if (auto decl = mlir::dyn_cast< const_decl >(op)) {
                    VAST_ASSERT(!id_to_value.count(decl.getSymName()));
                    id_to_value[decl.getSymName()] = decl;
                } else if (auto decl = mlir::dyn_cast< enum_decl >(op)) {
                    name_to_decl[decl.getSymName()] = decl;
                }
            };

            root->walk(walker);
        }
    };

    struct enum_type_converter
        : conv::tc::identity_type_converter
        , conv::tc::mixins< enum_type_converter >
    {
        mcontext_t &mctx;
        const enums_info &info;

        enum_type_converter(mcontext_t &mctx, const enums_info &info)
            : mctx(mctx), info(info)
        {
            init();
        }

        auto convert_enum() {
            return [&](hl::EnumType enum_type) -> maybe_type_t {
                auto name = enum_type.getName();
                auto it = info.name_to_decl.find(name);
                if (it == info.name_to_decl.end())
                    return {};

                // Need to be a separate decl to avoid compiler thinking it is a `const`.
                hl::EnumDeclOp decl = it->second;
                auto type = decl.getType();
                VAST_CHECK(type, "enum did not provide `getType()`, {0}", decl);
                return type;
            };
        }

        void init() {
            addConversion(convert_enum());
        }
    };

    struct type_rewriter : pattern_rewriter {
        type_rewriter(mcontext_t *mctx) : pattern_rewriter(mctx) {}
    };

    using type_conversion_pattern = conv::tc::generic_type_converting_pattern< enum_type_converter >;

    struct LowerEnumsPass : LowerEnumsBase< LowerEnumsPass > {
        void runOnOperation() override {
            auto enums = enums_info(this->getOperation());

            // Investigate if we want these as patterns instead. I would be curious
            // to see what's the runtime difference of repeating the traversal vs the
            // bookkeeping the conversion needs to work.
            // For now I am keeping this separate in case some random edge cases needs to be
            // added (this is easier to debug),
            // but eventually we probably should swap these to patterns.
            replace_constants(enums);
            replace_types(enums);

            erase_enums(enums);

            // TODO: Now the IR can be in a state where there are casts of `T -> T`. Should
            //       we simply call canonizer?
        }

        void replace_constants(const enums_info &enums) {
            auto replace_constants = [&](hl::EnumRefOp ref) {
                auto name = ref.getValue();
                auto it = enums.id_to_value.find(name);
                VAST_ASSERT(it != enums.id_to_value.end());

                auto bld = mlir::OpBuilder(ref);
                hl::EnumConstantOp enum_constant = it->second;
                auto value = enum_constant.getValue();

                // Make the integer
                operation i = bld.create< hl::ConstantOp >(ref.getLoc(), ref.getType(), value);
                ref.replaceAllUsesWith(i);
                ref->erase();
            };

            // First let's replace all constants
            this->getOperation()->walk(replace_constants);
        }

        void replace_types(const enums_info &enums) {
            auto tc = enum_type_converter(getContext(), enums);
            auto pattern = type_conversion_pattern(tc, getContext());
            auto replace_types = [&](operation root) {
                type_rewriter bld(&getContext());
                // We don't really care, failure only means that pattern was not applied,
                // which is a valid result.
                [[maybe_unused]] auto status = pattern.replace(root, bld);
            };
            this->getOperation()->walk(replace_types);
        }

        void erase_enums(const enums_info &enums) {
            for (auto &[_, decl] : enums.name_to_decl)
                decl->erase();
        }
    };
} // namespace vast::hl

std::unique_ptr< mlir::Pass > vast::hl::createLowerEnumsPass() {
    return std::make_unique< vast::hl::LowerEnumsPass >();
}

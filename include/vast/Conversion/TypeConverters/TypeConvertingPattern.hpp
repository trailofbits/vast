// Copyright (c) 2022, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <mlir/IR/Types.h>
VAST_UNRELAX_WARNINGS

#include "vast/Conversion/Common/Patterns.hpp"
#include "vast/Conversion/TypeConverters/DataLayout.hpp"
#include "vast/Conversion/TypeConverters/TypeConverter.hpp"
#include "vast/Dialect/HighLevel/HighLevelDialect.hpp"
#include "vast/Dialect/HighLevel/HighLevelOps.hpp"

#include "vast/Util/Common.hpp"

namespace vast::conv::tc {

    // Of `self_t` only `getTypeConverter` is required.
    template< typename self_t, typename type_converter >
    struct do_type_conversion_on_op
    {
      private:
        const auto &self() const { return static_cast< const self_t & >(*this); }

      public:

        auto &get_type_converter() const {
            return static_cast< type_converter & >(*self().getTypeConverter());
        }

        // TODO(conv:tc): This should probably be some interface instead, since
        //                 we are only updating the root?
        logical_result replace(mlir::FunctionOpInterface fn,
                               auto &rewriter) const
        {
            auto old_type = fn.getFunctionType();
            auto trg_type = get_type_converter().convert_type_to_type(old_type);
            VAST_CHECK(trg_type, "Type conversion failed for {0}", old_type);

            auto update = [&]() {
                fn.setType(*trg_type);
                if (fn->getNumRegions() != 0) {
                    fixup_entry_block(fn.front());
                }
            };

            rewriter.updateRootInPlace(fn, update);
            return mlir::success();
        }

        logical_result replace(
            mlir::Operation *op,
            auto &rewriter
        ) const {
            auto &tc = get_type_converter();

            auto update = [&]() {
                // TODO(conv:tc): This is pretty ad-hoc as it seems detection
                //                 of types in attributes is hard.
                mlir::AttrTypeReplacer replacer;
                replacer.addReplacement(conv::tc::convert_type_attr(tc));
                replacer.addReplacement(conv::tc::convert_data_layout_attrs(tc));
                replacer.addReplacement(conv::tc::convert_string_attr(tc));

                replacer.addReplacement([&](mlir_type t) { return tc.convert_type_to_type(t); }
                );

                replacer.recursivelyReplaceElementsIn(
                    op
                    , true /* replace attrs */
                    , false /* replace locs */
                    , true /* replace types */
                );

                // TODO(conv:tc): Is this still needed with the `replacer`?
                if (op->getNumRegions() != 0) {
                    fixup_entry_block(op->getRegion(0));
                }
            };

            rewriter.updateRootInPlace(op, update);

            return mlir::success();
        }

        void fixup_entry_block(mlir::Block &block) const {
            for (auto arg : block.getArguments()) {
                auto trg = get_type_converter().convert_type_to_type(arg.getType());
                VAST_CHECK(trg, "Type conversion failed: {0}", arg);
                arg.setType(*trg);
            }
        }

        void fixup_entry_block(mlir::Region &region) const {
            if (region.empty()) {
                return;
            }

            return fixup_entry_block(region.front());
        }
    };

    template< typename type_converter >
    struct generic_type_converting_pattern
        : generic_conversion_pattern,
          do_type_conversion_on_op< generic_type_converting_pattern< type_converter >,
                                    type_converter >
    {
        using base = generic_conversion_pattern;
        using base::base;

        logical_result matchAndRewrite(
            operation op, mlir::ArrayRef< mlir::Value >,
            conversion_rewriter &rewriter
        ) const override {
            if (auto func_op = mlir::dyn_cast< mlir::FunctionOpInterface >(op))
                return this->replace(func_op, rewriter);
            return this->replace(op, rewriter);
        }
    };
} // namespace vast::conv::tc

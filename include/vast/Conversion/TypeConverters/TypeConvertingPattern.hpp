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

    template< typename derived, typename type_converter >
    struct op_type_conversion
    {
        logical_result replace(operation op, auto &rewriter, const type_converter &tc) const {
            if (auto func_op = mlir::dyn_cast< mlir::FunctionOpInterface >(op))
                return replace_impl(func_op, rewriter, tc);
            return replace_impl(op, rewriter, tc);
        }

        logical_result replace(operation op, auto &rewriter) const {
            auto tc = static_cast< const type_converter & >(*self().getTypeConverter());
            return replace(op, rewriter, tc);
        }

      private:
        const auto &self() const { return static_cast< const derived & >(*this); }

        logical_result replace_impl(core::function_op_interface fn, auto &rewriter, const type_converter &tc) const {
            auto old_type = fn.getFunctionType();
            auto trg_type = tc.convert_type_to_type(old_type);
            VAST_CHECK(trg_type, "Type conversion failed for {0}", old_type);

            auto update = [&]() {
                fn.setType(*trg_type);
                if (!fn.empty() && fn->getNumRegions() != 0) {
                    fixup_entry_block(fn.front(), tc);
                }
            };

            rewriter.modifyOpInPlace(fn, update);
            return mlir::success();
        }

        logical_result replace_impl(operation op, auto &rewriter, const type_converter &tc) const {

            auto update = [&]() {
                // TODO(conv:tc): This is pretty ad-hoc as it seems detection
                //                 of types in attributes is hard.
                mlir::AttrTypeReplacer replacer;
                replacer.addReplacement(conv::tc::convert_type_attr(tc));
                replacer.addReplacement(conv::tc::convert_data_layout_attrs(tc));
                replacer.addReplacement(conv::tc::convert_string_attr(tc));

                replacer.addReplacement([&](mlir_type t) { return tc.convert_type_to_type(t); });

                replacer.recursivelyReplaceElementsIn(
                    op
                    , true /* replace attrs */
                    , false /* replace locs */
                    , true /* replace types */
                );

                // TODO(conv:tc): Is this still needed with the `replacer`?
                if (op->getNumRegions() != 0) {
                    fixup_entry_block(op->getRegion(0), tc);
                }
            };

            rewriter.modifyOpInPlace(op, update);
            return mlir::success();
        }

        void fixup_entry_block(mlir::Block &block, const type_converter &tc) const {
            for (auto arg : block.getArguments()) {
                auto trg = tc.convert_type_to_type(arg.getType());
                VAST_CHECK(trg, "Type conversion failed: {0}", arg);
                arg.setType(*trg);
            }
        }

        void fixup_entry_block(mlir::Region &region, const type_converter &tc) const {
            if (region.empty()) {
                return;
            }

            return fixup_entry_block(region.front(), tc);
        }
    };

    template< typename type_converter >
    struct type_converting_pattern
        : generic_conversion_pattern,
          op_type_conversion<
            type_converting_pattern< type_converter >, type_converter
          >
    {
        using base = generic_conversion_pattern;
        using base::base;

        using conversion = op_type_conversion< type_converting_pattern, type_converter >;
        using conversion::replace;

        logical_result matchAndRewrite(
            operation op, mlir::ArrayRef< mlir::Value >,
            conversion_rewriter &rewriter
        ) const override {
            return replace(op, rewriter);
        }
    };
} // namespace vast::conv::tc

// Copyright (c) 2021-present, Trail of Bits, Inc.

VAST_RELAX_WARNINGS
#include <mlir/Analysis/DataLayoutAnalysis.h>
#include <mlir/IR/BuiltinAttributeInterfaces.h>
#include <mlir/Transforms/DialectConversion.h>
VAST_UNRELAX_WARNINGS

#include "vast/Dialect/HighLevel/HighLevelAttributes.hpp"
#include "vast/Dialect/HighLevel/HighLevelOps.hpp"
#include "vast/Dialect/HighLevel/HighLevelTypes.hpp"

#include "vast/Dialect/Core/CoreTypes.hpp"
#include "vast/Dialect/Core/CoreAttributes.hpp"

#include "vast/Conversion/Common/Types.hpp"

#include "vast/Util/Maybe.hpp"

#include <algorithm>
#include <iostream>

namespace vast::conv::tc
{
    struct HLToStd : mlir::TypeConverter
    {
        const mlir::DataLayout &dl;
        mlir::MLIRContext &mctx;

        HLToStd(const mlir::DataLayout &dl, mcontext_t &mctx)
            : mlir::TypeConverter(), dl(dl), mctx(mctx)
        {
            // Fallthrough option - we define it first as it seems the framework
            // goes from the last added conversion.
            addConversion([&](mlir_type t) -> maybe_type_t {
                return Maybe(t)
                    .keep_if([](auto t) { return !hl::isHighLevelType(t); })
                    .take_wrapped< maybe_type_t >();
            });
            addConversion([&](mlir_type t) { return this->try_convert_intlike(t); });
            addConversion([&](mlir_type t) { return this->try_convert_floatlike(t); });

            addConversion([&](hl::LValueType t) { return this->convert_lvalue_type(t); });
            addConversion([&](hl::DecayedType t) { return this->convert_decayed_type(t); });

            // Use provided data layout to get the correct type.
            addConversion([&](hl::PointerType t) { return this->convert_ptr_type(t); });
            addConversion([&](hl::ArrayType t) { return this->convert_arr_type(t); });
            addConversion([&](hl::VoidType t) -> maybe_type_t {
                return { mlir::NoneType::get(&mctx) };
            });
        }

        maybe_types_t convert_type(mlir_type t) {
            types_t out;
            if (mlir::succeeded(convertTypes(t, out))) {
                return { std::move(out) };
            }
            return {};
        }

        // TODO(lukas): Take optional to denote that is may be `Signless`.
        auto int_type(unsigned bitwidth, bool is_signed) {
            auto signedness = [=]() {
                if (is_signed)
                    return mlir::IntegerType::SignednessSemantics::Signed;
                return mlir::IntegerType::SignednessSemantics::Unsigned;
            }();
            return mlir::IntegerType::get(&this->mctx, bitwidth, signedness);
        }

        auto convert_type() {
            return [&](auto t) { return this->convert_type(t); };
        }

        auto convert_type_to_type() {
            return [&](auto t) { return this->convert_type_to_type(t); };
        }

        auto convert_pointer_element_typee() {
            return [&](auto t) -> maybe_type_t {
                if (t.template isa< hl::VoidType >()) {
                    return int_type(8u, mlir::IntegerType::SignednessSemantics::Signless);
                }
                return this->convert_type_to_type(t);
            };
        }

        auto make_int_type(bool is_signed) {
            return [=, this](auto t) { return int_type(dl.getTypeSizeInBits(t), is_signed); };
        }

        auto make_float_type() {
            return [&](auto t) {
                auto target_bw = dl.getTypeSizeInBits(t);
                switch (target_bw) {
                    case 16:
                        return mlir::FloatType::getF16(&mctx);
                    case 32:
                        return mlir::FloatType::getF32(&mctx);
                    case 64:
                        return mlir::FloatType::getF64(&mctx);
                    case 80:
                        return mlir::FloatType::getF80(&mctx);
                    case 128:
                        return mlir::FloatType::getF128(&mctx);
                    default:
                        VAST_UNREACHABLE("Cannot lower float bitsize {0}", target_bw);
                }
            };
        }

        auto make_ptr_type(auto quals) {
            return [=](auto t) { return hl::PointerType::get(t.getContext(), t, quals); };
        }

        auto make_lvalue_type() {
            return [&](auto t) { return hl::LValueType::get(t.getContext(), t); };
        }

        maybe_types_t convert_type_to_types(mlir_type t, std::size_t count = 1) {
            return Maybe(t)
                .and_then(convert_type())
                .keep_if([&](const auto &ts) { return ts->size() == count; })
                .take_wrapped< maybe_types_t >();
        }

        maybe_type_t convert_type_to_type(mlir_type t) {
            return Maybe(t)
                .and_then([&](auto t) { return this->convert_type_to_types(t, 1); })
                .and_then([&](auto ts) { return *ts->begin(); })
                .take_wrapped< maybe_type_t >();
        }

        maybe_type_t try_convert_intlike(mlir_type t) {
            // For now `bool` behaves the same way as any other integer type.
            if (!hl::isIntegerType(t) && !hl::isBoolType(t)) {
                return {};
            }

            return Maybe(t)
                .and_then(make_int_type(hl::isSigned(t)))
                .take_wrapped< maybe_type_t >();
        }

        maybe_type_t try_convert_floatlike(mlir_type t) {
            return Maybe(t)
                .keep_if(hl::isFloatingType)
                .and_then(make_float_type())
                .take_wrapped< maybe_type_t >();
        }

        maybe_type_t convert_decayed_type(hl::DecayedType t) {
            return Maybe(t.getElementType())
                .and_then(convert_type_to_type())
                .unwrap()
                .take_wrapped< maybe_type_t >();
        }

        maybe_type_t convert_ptr_type(hl::PointerType t) {
            return Maybe(t.getElementType())
                .and_then(convert_pointer_element_typee())
                .unwrap()
                .and_then(make_ptr_type(t.getQuals()))
                .take_wrapped< maybe_type_t >();
        }

        maybe_type_t convert_lvalue_type(hl::LValueType t) {
            return Maybe(t.getElementType())
                .and_then(convert_type_to_type())
                .unwrap()
                .and_then(make_lvalue_type())
                .take_wrapped< maybe_type_t >();
        }

        maybe_type_t convert_arr_type(hl::ArrayType arr) {
            auto [dims, nested_ty] = arr.dim_and_type();
            std::vector< int64_t > coerced_dim;
            for (auto dim : dims) {
                if (dim.has_value()) {
                    coerced_dim.push_back(dim.value());
                } else {
                    coerced_dim.push_back(mlir::ShapedType::kDynamic);
                }
            }

            return Maybe(convert_type_to_type(nested_ty))
                .and_then([&](auto t) { return mlir::MemRefType::get({ coerced_dim }, *t); })
                .take_wrapped< maybe_type_t >();
        }
    };
} // namespace vast::conv::tc

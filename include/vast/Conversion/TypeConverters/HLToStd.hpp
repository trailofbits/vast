// Copyright (c) 2021-present, Trail of Bits, Inc.

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <mlir/Analysis/DataLayoutAnalysis.h>
#include <mlir/IR/BuiltinAttributeInterfaces.h>
#include <mlir/Transforms/DialectConversion.h>
VAST_UNRELAX_WARNINGS

#include "vast/Dialect/HighLevel/HighLevelAttributes.hpp"
#include "vast/Dialect/HighLevel/HighLevelOps.hpp"
#include "vast/Dialect/HighLevel/HighLevelTypes.hpp"

#include "vast/Dialect/Core/CoreAttributes.hpp"
#include "vast/Dialect/Core/CoreTypes.hpp"

#include "vast/Conversion/Common/Types.hpp"
#include "vast/Conversion/TypeConverters/TypeConverter.hpp"

#include "vast/Util/Maybe.hpp"

#include <algorithm>
#include <iostream>

namespace vast::conv::tc {
    // TODO(conv:tc): Encode as concept.
    // Requires of `self_t`
    // * inherits/implements `mixins`
    // * implements `int_type()`
    template< typename self_t >
    struct HLAggregates
    {
      private:
        self_t &self() { return static_cast< self_t & >(*this); }

      public:
        // This is not a ctor so that users can position it as they want
        // in their initialization.
        void init() {
            self().addConversion(convert_decayed_type());
            self().addConversion(convert_lvalue_type());
            self().addConversion(convert_pointer_type());
            self().addConversion(convert_elaborated_type());
            self().addConversion(convert_paren_type());
        }

        auto convert_decayed_type() {
            return [&](hl::DecayedType type) {
                return Maybe(type.getElementType())
                    .and_then(self().convert_type_to_type())
                    .unwrap()
                    .template take_wrapped< maybe_type_t >();
            };
        }

        auto convert_lvalue_type() {
            return [&](hl::LValueType type) {
                return Maybe(type.getElementType())
                    .and_then(self().convert_type_to_type())
                    .unwrap()
                    .and_then(self().template make_aggregate_type< hl::LValueType >())
                    .template take_wrapped< maybe_type_t >();
            };
        }

        auto convert_elaborated_type() {
            return [&](hl::ElaboratedType type) {
                using raw = hl::ElaboratedType;

                return Maybe(type.getElementType())
                    .and_then(self().convert_type_to_type())
                    .unwrap()
                    .and_then(self().template make_aggregate_type< raw >(type.getQuals()))
                    .template take_wrapped< maybe_type_t >();
            };
        }

        auto convert_pointer_type() {
            return [&](hl::PointerType type) {
                using raw = hl::PointerType;

                return Maybe(type.getElementType())
                    .and_then(self().convert_pointer_element_type())
                    .unwrap()
                    .and_then(self().template make_aggregate_type< raw >(type.getQuals()))
                    .template take_wrapped< maybe_type_t >();
            };
        }

        auto convert_paren_type() {
            return [&](hl::ParenType type) {
                return Maybe(type.getElementType())
                    .and_then(self().convert_type_to_type())
                    .unwrap()
                    .template take_wrapped< maybe_type_t >();
            };
        }

      protected:
        auto convert_pointer_element_type() {
            return [&](auto t) -> maybe_type_t {
                if (t.template isa< hl::VoidType >()) {
                    auto sign = mlir::IntegerType::SignednessSemantics::Signless;
                    return self().int_type(8u, sign);
                }
                return self().convert_type_to_type(t);
            };
        }

        // `args` is passed by value as I have no idea how to convince clang
        // to not pass in `quals` as `const` when forwarding.
        // They are expected to be lightweight objects anyway.
        template< typename T, typename... Args >
        auto make_aggregate_type(Args... args) {
            return [=](mlir_type elem) { return T::get(elem.getContext(), elem, args...); };
        }
    };

    template< typename self_t >
    struct CoreToStd
    {
      private:
        self_t &self() { return static_cast< self_t & >(*this); }

      public:
        auto convert_fn_type() {
            return [&](core::FunctionType t) -> maybe_type_t {
                // To be consistent with how others use the conversion, we
                // do not convert input types directly.
                auto signature_conversion = self().signature_conversion(t.getInputs());

                auto results = self().convert_types_to_types(t.getResults());

                if (!signature_conversion || !results) {
                    return {};
                }

                return { core::FunctionType::get(
                    t.getContext(), signature_conversion->getConvertedTypes(), *results,
                    t.isVarArg()
                ) };
            };
        }

        void init() { self().addConversion(convert_fn_type()); }
    };

    struct HLToStd
        : base_type_converter
        , mixins< HLToStd >
        , HLAggregates< HLToStd >
        , CoreToStd< HLToStd >
    {
        using Base = mixins< HLToStd >;
        using Base::convert_type_to_type;
        using Base::convert_type_to_types;

        const mlir::DataLayout &dl;
        mlir::MLIRContext &mctx;

        HLToStd(const mlir::DataLayout &dl, mcontext_t &mctx)
            : base_type_converter(), dl(dl), mctx(mctx) {
            // Fallthrough option - we define it first as it seems the framework
            // goes from the last added conversion.
            addConversion([&](mlir_type t) -> maybe_type_t {
                return Maybe(t)
                    .keep_if([](auto t) { return !hl::isHighLevelType(t); })
                    .take_wrapped< maybe_type_t >();
            });
            addConversion([&](mlir_type t) { return this->try_convert_intlike(t); });
            addConversion([&](mlir_type t) { return this->try_convert_floatlike(t); });

            // Use provided data layout to get the correct type.
            addConversion([&](hl::ArrayType t) { return this->convert_arr_type(t); });
            addConversion([&](hl::VoidType t) -> maybe_type_t {
                return { mlir::NoneType::get(&mctx) };
            });

            HLAggregates< HLToStd >::init();
            CoreToStd< HLToStd >::init();
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
                if (is_signed) {
                    return mlir::IntegerType::SignednessSemantics::Signed;
                }
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
                        VAST_FATAL("Cannot lower float bitsize {0}", target_bw);
                }
            };
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

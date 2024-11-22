// Copyright (c) 2022, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <mlir/IR/BuiltinDialect.h>
#include <mlir/IR/Types.h>
#include <mlir/Transforms/DialectConversion.h>
VAST_UNRELAX_WARNINGS

#include <gap/core/crtp.hpp>

#include "vast/Dialect/Core/CoreTypes.hpp"
#include "vast/Dialect/Core/Interfaces/FunctionInterface.hpp"

#include "vast/Dialect/HighLevel/HighLevelDialect.hpp"

#include "vast/Util/Common.hpp"
#include "vast/Util/DataLayout.hpp"
#include "vast/Util/Maybe.hpp"
#include "vast/Util/TypeUtils.hpp"

namespace vast::conv::tc {

    using signature_conversion_t       = mlir::TypeConverter::SignatureConversion;
    using maybe_signature_conversion_t = std::optional< signature_conversion_t >;

    using core_function_type = core::FunctionType;

    struct base_type_converter : mlir::TypeConverter
    {
        using base = mlir::TypeConverter;
        using base::base;

        bool isSignatureLegal(core::FunctionType ty);

        maybe_types_t do_conversion(mlir::Type t) const {
            types_t out;
            if (mlir::succeeded(this->convertTypes(t, out))) {
                return { std::move(out) };
            }
            return {};
        }
    };

    static inline constexpr auto unrealized_materialization = [] (
        mlir::OpBuilder &builder, mlir::Type resultType,
        mlir::ValueRange inputs, mlir::Location loc
    ) -> std::optional< mlir::Value > {
        if (inputs.size() != 1) {
            return std::nullopt;
        }

        return builder
            .create< mlir::UnrealizedConversionCastOp >(loc, resultType, inputs)
            .getResult(0);
    };

    struct identity_type_converter : base_type_converter
    {
        using base = base_type_converter;

        template< typename... args_t >
        identity_type_converter(args_t &&...args) : base(std::forward< args_t >(args)...) {
            addConversion([&](mlir_type t) { return t; });
        }
    };

    template< typename derived >
    struct function_type_converter : gap::core::crtp< derived, function_type_converter >
    {
        using base = gap::core::crtp< derived, function_type_converter >;
        using base::underlying;

        // These can not be moved to the constructor, because we would be downcasting
        // to the derived class while it not yet exists, resulting in UB
        // Calling this in the derived class constructor should be safe
        void init() {
            underlying().addConversion([&](core_function_type t) -> maybe_type_t {
                auto sig = underlying().signature_conversion(t.getInputs());
                auto results = underlying().convert_types_to_types(t.getResults());

                if (!sig || !results) {
                    return std::nullopt;
                }

                return core_function_type::get(
                    t.getContext(), sig->getConvertedTypes(), *results, t.isVarArg()
                );
            });
        }
    };

    struct type_converter_with_dl : base_type_converter
    {
        using base = base_type_converter;
        using dl_t = mlir::DataLayout;

        const dl_t &dl;

        type_converter_with_dl(const dl_t &dl, mcontext_t &mctx) : dl(dl) {}
    };

    template< typename R, typename UnaryPred >
    bool all_of_subtypes(R &&types, UnaryPred &&pred) {
        mlir::AttrTypeWalker walker;

        walker.addWalk([pred = std::forward< UnaryPred >(pred)](mlir_type t) {
            if (!pred(t)) {
                return mlir::WalkResult::interrupt();
            }

            return mlir::WalkResult::advance();
        });

        for (auto type : types) {
            if (walker.walk(type).wasInterrupted()) {
                return false;
            }
        }

        return true;
    }

    template< typename derived >
    struct mixins
    {
        const derived &self() const { return static_cast< const derived & >(*this); }
        derived &self() { return static_cast< derived & >(*this); }

        auto convert_type() const {
            return [&](auto t) { return self().do_conversion(t); };
        }

        auto convert_type_to_type() const {
            return [&](auto t) { return self().convert_type_to_type(t); };
        }

        maybe_types_t convert_type_to_types(mlir_type t, std::size_t count = 1) const {
            return Maybe(t)
                .and_then(self().convert_type())
                .keep_if([&](const auto &ts) { return ts->size() == count; })
                .template take_wrapped< maybe_types_t >();
        }

        maybe_type_t convert_type_to_type(mlir_type t) const {
            return Maybe(t)
                .and_then([&](auto t) { return self().convert_type_to_types(t, 1); })
                .and_then([&](auto ts) { return *ts->begin(); })
                .template take_wrapped< maybe_type_t >();
        }

        maybe_type_t convert_arg_type(mlir_type t, unsigned long /* idx */) const {
            return convert_type_to_type(t);
        }

        auto appender(types_t &out) const {
            return [&](auto collection) {
                out.insert(
                    out.end(), std::move_iterator(collection.begin()),
                    std::move_iterator(collection.end())
                );
            };
        }

        maybe_types_t convert_types_to_types(auto types) const {
            types_t out;
            auto append = appender(out);

            for (auto t : types) {
                if (auto c = convert_type_to_types(t)) {
                    append(std::move(*c));
                } else {
                    return {};
                }
            }

            return { out };
        }

        maybe_signature_conversion_t signature_conversion(const auto &inputs) const {
            signature_conversion_t sc(inputs.size());
            for (auto [i, arg] : llvm::enumerate(inputs)) {
                if (auto trg = self().convert_arg_type(arg, i)) {
                    sc.addInputs(i, *trg);
                } else {
                    return {};
                }
            }
            return { std::move(sc) };
        }

        auto get_is_type_conversion_legal() {
            // We need to check
            //  * result types
            //  * types of attributes
            // types of arguments are result types of a different op.
            return [this](operation op) {
                auto res   = self().isLegal(op->getResults().getTypes());
                auto attrs = contains_subtype(op->getAttrDictionary(), self().get_is_illegal());
                return res && !attrs;
            };
        }

        auto get_is_illegal() const {
            return [&](mlir_type t) { return !self().isLegal(t); };
        }

        auto get_is_legal() const {
            return [&](mlir_type t) { return self().isLegal(t); };
        }

        template< typename op_t >
        auto get_has_only_legal_types() const {
            return [&](op_t op) -> bool {
                return !has_type_somewhere(op, get_is_illegal());
            };
        }

        bool has_legal_return_type(auto op) const {
            return all_of_subtypes(op->getResults().getTypes(), get_is_legal());
        }

        bool has_legal_operand_types(auto op) const {
            return all_of_subtypes(op->getOperands().getTypes(), get_is_legal());
        }

        template< typename op_t >
        auto get_has_legal_return_type() const {
            return [&](op_t op) -> bool {
                auto types = op->getResults().getTypes();
                return !contains_subtype(types, get_is_illegal());
            };
        }

        template< typename op_t >
        auto get_has_legal_operand_types() const {
             return [&](op_t op) -> bool {
                auto types = op->getOperands().getTypes();
                return !contains_subtype(types, get_is_illegal());
            };
        }

        mcontext_t &get_context() const { return self().mctx; }
    };

    // TODO(lukas): `rewriter.convertRegionTypes` should do the job, but it does not.
    //              It' hard to debug, but it seems to leave dangling values
    //              instead of correctly rewiring SSA data flow. Investigate, we
    //              would prefer to use mlir native solutions.
    // NOTE(lukas): This may break the contract that all modifications happen
    //              via rewriter.
    void convert_region_types(auto old_fn, auto new_fn, auto signature_conversion) {
        auto orig_count = new_fn.getBody().getNumArguments();
        auto &block     = new_fn.getBody();
        for (std::size_t i = 0; i < orig_count; ++i) {
            auto new_arg = block.addArgument(
                signature_conversion.getConvertedTypes()[i], block.getArgument(i).getLoc()
            );
            block.getArgument(i).replaceAllUsesWith(new_arg);
        }

        while (block.getNumArguments() != orig_count) {
            block.eraseArgument(0);
        }
    }

    auto convert_type_attr(auto &type_converter) {
        return [&type_converter](mlir::TypeAttr attr) {
            return Maybe(attr.getValue())
                .and_then(type_converter.convert_type_to_type())
                .unwrap()
                .and_then(mlir::TypeAttr::get)
                .template take_wrapped< maybe_attr_t >();
        };
    }

    auto convert_string_attr(auto &type_converter) {
        return [&type_converter](mlir::StringAttr attr) -> maybe_attr_t {
            return Maybe(attr.getType())
                .and_then(type_converter.convert_type_to_type())
                .unwrap()
                .and_then([&](auto type) { return mlir::StringAttr::get(attr.getValue(), type); })
                .template take_wrapped< maybe_attr_t >();
        };
    }

} // namespace vast::conv::tc

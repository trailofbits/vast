// Copyright (c) 2022, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <mlir/IR/BuiltinDialect.h>
#include <mlir/IR/Types.h>
#include <mlir/Transforms/DialectConversion.h>
VAST_UNRELAX_WARNINGS

#include "vast/Dialect/Core/CoreTypes.hpp"
#include "vast/Dialect/HighLevel/HighLevelDialect.hpp"

#include "vast/Util/Common.hpp"
#include "vast/Util/DataLayout.hpp"
#include "vast/Util/Maybe.hpp"

namespace vast::conv::tc {
    using signature_conversion_t       = mlir::TypeConverter::SignatureConversion;
    using maybe_signature_conversion_t = std::optional< signature_conversion_t >;

    struct base_type_converter : mlir::TypeConverter
    {
        using base = mlir::TypeConverter;
        using base::base;

        bool isSignatureLegal(core::FunctionType ty);
    };

    struct identity_type_converter : base_type_converter
    {
        using base = base_type_converter;

        template< typename... args_t >
        identity_type_converter(args_t &&...args) : base(std::forward< args_t >(args)...) {
            addConversion([&](mlir_type t) { return t; });
        }
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
        derived &self() { return static_cast< derived & >(*this); }

        auto convert_type() {
            return [&](auto t) { return self().do_conversion(t); };
        }

        auto convert_type_to_type() {
            return [&](auto t) { return self().convert_type_to_type(t); };
        }

        maybe_types_t convert_type_to_types(mlir_type t, std::size_t count = 1) {
            return Maybe(t)
                .and_then(self().convert_type())
                .keep_if([&](const auto &ts) { return ts->size() == count; })
                .template take_wrapped< maybe_types_t >();
        }

        maybe_type_t convert_type_to_type(mlir_type t) {
            return Maybe(t)
                .and_then([&](auto t) { return self().convert_type_to_types(t, 1); })
                .and_then([&](auto ts) { return *ts->begin(); })
                .template take_wrapped< maybe_type_t >();
        }

        maybe_type_t convert_type_to_type(core::FunctionType fty) {
            auto params = self().convert_types_to_types(fty.getInputs());
            if (!params) {
                return std::nullopt;
            }

            auto results = self().convert_types_to_types(fty.getResults());
            if (!results) {
                return std::nullopt;
            }

            return core::FunctionType::get(fty.getContext(), *params, *results, fty.isVarArg());
        }

        auto appender(types_t &out) {
            return [&](auto collection) {
                out.insert(
                    out.end(), std::move_iterator(collection.begin()),
                    std::move_iterator(collection.end())
                );
            };
        }

        maybe_types_t convert_types_to_types(auto types) {
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

        mcontext_t &get_context() { return self().mctx; }
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
                .and_then([&](auto type) { return type_converter.convertType(type); })
                .and_then(mlir::TypeAttr::get)
                .template take_wrapped< maybe_attr_t >();
        };
    }
} // namespace vast::conv::tc

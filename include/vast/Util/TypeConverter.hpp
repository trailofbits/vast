// Copyright (c) 2022, Trail of Bits, Inc.

#pragma once

VAST_RELAX_WARNINGS
#include <mlir/IR/Types.h>
VAST_UNRELAX_WARNINGS

#include "vast/Dialect/HighLevel/HighLevelDialect.hpp"
#include "vast/Dialect/Core/CoreTypes.hpp"

#include "vast/Util/Maybe.hpp"

namespace vast::util
{
    // Walks all types (including nested types for aggregates) and calls the unary function
    // on each of them.
    template< typename UnaryFn >
    void walk_type(mlir::Type root, UnaryFn &&fn)
    {
        fn(root);
        if (auto aggregate = root.dyn_cast< mlir::SubElementTypeInterface >())
            aggregate.walkSubTypes(std::forward< UnaryFn >(fn));

    }

    // TODO(lukas): `fn` is copied.
    template< typename UnaryFn >
    void walk_type(mlir::TypeRange range, UnaryFn fn)
    {
        for (auto type : range)
            walk_type(type, fn);
    }

    template< typename R, typename UnaryFn, typename Combine >
    bool walk_type(R &&type, UnaryFn &&fn, Combine &&combine, bool init)
    {
        bool out = init;
        auto wrap = [impl = std::forward< UnaryFn >(fn),
                     combine = std::forward< Combine >(combine),
                     &out](auto element_type)
        {
            out = combine(out, impl(element_type));
        };
        walk_type(std::forward< R >(type), std::move(wrap));

        return out;
    }


    template< typename R, typename UnaryPred >
    bool for_each_subtype(R &&type, UnaryPred &&pred)
    {
        return walk_type(std::forward< R >(type),
                         std::forward< UnaryPred >(pred),
                         std::logical_and{}, true);
    }

    template< typename Self >
    struct TCHelpers
    {
        using types_t = mlir::SmallVector< mlir::Type >;
        using maybe_type_t = llvm::Optional< mlir::Type >;
        using maybe_types_t = llvm::Optional< types_t >;

        Self &self() { return static_cast< Self & >(*this); }

        auto convert_type() { return [&](auto t) { return self().do_conversion(t); }; }
        auto convert_type_to_type()
        {
            return [&](auto t) { return self().convert_type_to_type(t); };
        }

        maybe_types_t convert_type_to_types(mlir::Type t, std::size_t count = 1)
        {
            return Maybe(t).and_then(self().convert_type())
                           .keep_if([&](const auto &ts) { return ts->size() == count; })
                           .template take_wrapped< maybe_types_t >();
        }

        maybe_type_t convert_type_to_type(mlir::Type t)
        {
            return Maybe(t).and_then([&](auto t){ return self().convert_type_to_types(t, 1); })
                           .and_then([&](auto ts){ return *ts->begin(); })
                           .template take_wrapped< maybe_type_t >();
        }

        maybe_type_t convert_type_to_type(core::FunctionType fty)
        {
            auto params = self().convert_types_to_types(fty.getInputs());
            if (!params) {
                return std::nullopt;
            }

            auto results = self().convert_types_to_types(fty.getResults());
            if (!results) {
                return std::nullopt;
            }

            return core::FunctionType::get(
                fty.getContext(), *params, *results, fty.isVarArg()
            );
        }

        auto appender(types_t &out)
        {
            return [&](auto collection)
            {
                out.insert(out.end(), std::move_iterator(collection.begin()),
                                      std::move_iterator(collection.end()));
            };
        }

        maybe_types_t convert_types_to_types(auto types)
        {
            types_t out;
            auto append = appender(out);

            for (auto t : types)
                if (auto c = convert_type_to_types(t))
                    append(std::move(*c));
                else
                    return {};

            return { out };
        }
    };

    // Comment out
    template< typename Impl >
    struct TypeConverterWrapper
    {
        using types_t = mlir::SmallVector< mlir::Type >;
        using maybe_type_t = llvm::Optional< mlir::Type >;
        using maybe_types_t = llvm::Optional< types_t >;

        Impl &impl;

        TypeConverterWrapper(Impl &impl_) : impl(impl_) {}

        Impl *operator->() { return &impl; }
        const Impl *operator->() const { return &impl; }

        maybe_types_t do_conversion(mlir::Type t)
        {
            types_t out;
            if (mlir::succeeded(impl.convertTypes(t, out)))
                return { std::move( out ) };
            return {};
        }
    };


    // TODO(lukas): `rewriter.convertRegionTypes` should do the job, but it does not.
    //              It' hard to debug, but it seems to leave dangling values
    //              instead of correctly rewiring SSA data flow. Investigate, we
    //              would prefer to use mlir native solutions.
    // NOTE(lukas): This may break the contract that all modifications happen
    //              via rewriter.
    void convert_region_types(auto old_fn, auto new_fn, auto signature_conversion)
    {
        auto orig_count = new_fn.getBody().getNumArguments();
        auto &block = new_fn.getBody();
        for (std::size_t i = 0; i < orig_count; ++i)
        {
            auto new_arg = block.addArgument(signature_conversion.getConvertedTypes()[i],
                                             block.getArgument(i).getLoc());
            block.getArgument(i).replaceAllUsesWith(new_arg);
        }

        while (block.getNumArguments() != orig_count)
            block.eraseArgument(0);
    }

    struct IdentityTC : mlir::TypeConverter
    {
        template< typename ... Args >
        IdentityTC(Args && ... args) : mlir::TypeConverter(std::forward< Args >(args) ...)
        {
            addConversion([&](mlir::Type t) { return t; });
        }
    };

    template< typename TC >
    struct AttributeConverter
    {
        using type_converter = TC;
        using maybe_attribute = std::optional< mlir::Attribute >;

        mcontext_t &mctx;
        type_converter &tc;

        AttributeConverter(mcontext_t &mctx, type_converter &tc) : mctx(mctx), tc(tc) {}

        template< typename A, typename ... Args >
        auto make_hl_attr(Args && ... args) const
        {
            // Expected cheap values are passed around, otherwise perfectly forward.
            return [=](auto type)
            {
                return A::get(type, args ...);
            };
        }

        template< typename Attr, typename ... Rest >
        maybe_attribute hl_attr_conversion(mlir::Attribute attr) const
        {
            if (auto hl_attr = attr.dyn_cast< Attr >())
            {
                return Maybe(hl_attr.getType())
                    .and_then(tc.convert_type_to_type())
                    .unwrap()
                    .and_then(make_hl_attr< Attr >(hl_attr.getValue()))
                    .template take_wrapped< maybe_attribute >();
            }
            if constexpr (sizeof ... (Rest) != 0)
                return hl_attr_conversion< Rest ... >(attr);
            return {};
        }

        maybe_attribute convertAttr(mlir::Attribute attr) const
        {
            if (auto out = hl_attr_conversion< hl::BooleanAttr,
                                               hl::IntegerAttr,
                                               hl::FloatAttr,
                                               hl::StringAttr,
                                               hl::StringLiteralAttr >(attr))
                return out;

            if (auto type_attr = attr.dyn_cast< mlir::TypeAttr >())
            {
                return Maybe(type_attr.getValue())
                    .and_then(tc.convert_type_to_type())
                    .unwrap()
                    .and_then(mlir::TypeAttr::get)
                    .template take_wrapped< maybe_attribute >();
            }
            return {};
        }

        template< typename T >
        auto get_type_attr_conversion() const
        {
            return [=](T attr) -> maybe_attribute
            {
                auto converted = convertAttr(attr);
                return (converted) ? converted : attr;
            };
        }

        [[nodiscard]] logical_result convert(operation op) const
        {
            auto attrs = op->getAttrDictionary();
            auto nattrs = attrs.replaceSubElements(
                    get_type_attr_conversion< mlir::TypeAttr >(),
                    get_type_attr_conversion< mlir::TypedAttr >()
            );
            op->setAttrs(mlir::dyn_cast< mlir::DictionaryAttr >(nattrs));
            return mlir::success();
        }
    };

    template< typename TC >
    AttributeConverter( mcontext_t &, TC & ) -> AttributeConverter< TC >;

} // namespace vast::util

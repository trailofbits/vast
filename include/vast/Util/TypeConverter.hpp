// Copyright (c) 2022, Trail of Bits, Inc.

#pragma once

VAST_RELAX_WARNINGS
#include <mlir/IR/Types.h>
VAST_UNRELAX_WARNINGS

#include "vast/Util/Maybe.hpp"

namespace vast::util
{
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


} // namespace vast::util

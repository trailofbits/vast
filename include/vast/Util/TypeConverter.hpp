// Copyright (c) 2022, Trail of Bits, Inc.


VAST_RELAX_WARNINGS
#include <mlir/IR/Types.h>
VAST_UNRELAX_WARNINGS

#include "vast/Util/Maybe.hpp"

namespace vast::hl
{
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

        maybe_types_t convert_type(mlir::Type t)
        {
            types_t out;
            if (mlir::succeeded(impl.convertTypes(t, out)))
                return { std::move( out ) };
            return {};
        }

        auto convert_type() { return [&](auto t) { return this->convert_type(t); }; }
        auto convert_type_to_type()
        {
            return [&](auto t) { return this->convert_type_to_type(t); };
        }

        maybe_types_t convert_type_to_types(mlir::Type t, std::size_t count = 1)
        {
            return Maybe(t).and_then(convert_type())
                           .keep_if([&](const auto &ts) { return ts->size() == count; })
                           .template take_wrapped< maybe_types_t >();
        }

        maybe_type_t convert_type_to_type(mlir::Type t)
        {
            return Maybe(t).and_then([&](auto t){ return this->convert_type_to_types(t, 1); })
                           .and_then([&](auto ts){ return *ts->begin(); })
                           .template take_wrapped< maybe_type_t >();
        }
    };
} // namespace vast::hl

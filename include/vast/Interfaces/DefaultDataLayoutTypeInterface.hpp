// Copyright (c) 2023-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/OperationSupport.h>
VAST_RELAX_WARNINGS

#include "vast/Util/DataLayout.hpp"

namespace vast
{
    // Shared utility by `DefaultDataLayoutTypeInterface` to correctly
    // filter data layout entries. Once one is selected it will be casted
    // to `DLEntry` and passed `extract` to produce resulting value.
    // TODO(interface): Return can be generic based on what `extract` returns.
    template< typename ConcreteType, typename Interface, typename Extract >
    unsigned default_dl_query(const Interface &self, Extract &&extract,
                              const mlir::DataLayout &dl,
                              mlir::DataLayoutEntryListRef entries)
    {
        VAST_CHECK(entries.size() != 0, "Data layout query did not match to any dl entry!");

        std::optional< unsigned > out;
        auto handle_entry = [&](const auto &entry)
        {
            auto current = extract(entry);
            if (!out)
                out = current;

            VAST_CHECK(*out == current,
                       "Inconsistent entries {0} != {1}, total number of entries",
                       *out, current, entries.size());
        };

        // First we try to find an exact match in the data layout entries.
        auto casted_self = static_cast< const ConcreteType & >(self);
        for (const auto &entry : entries)
        {
            auto raw = dl::DLEntry(entry);
            if (casted_self == raw.type)
                handle_entry(raw);
        }

        if (out.has_value())
            return *out;

        // Since we did not find the exact entry we search for generalised type.
        for (const auto &entry : entries)
        {
            auto raw = dl::DLEntry(entry);
            if (mlir::isa< ConcreteType >(raw.type))
                handle_entry(raw);
        }

        VAST_CHECK(out.has_value(), "Data layout query of {0} did not produce a value!",
                   casted_self);
        return *out;
    }

} // namespace vast

/// Include the generated interface declarations.
#include "vast/Interfaces/DefaultDataLayoutTypeInterface.h.inc"

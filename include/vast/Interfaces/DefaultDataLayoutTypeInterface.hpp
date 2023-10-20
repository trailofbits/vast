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

        for (const auto &entry : entries)
        {
            auto raw = dl::DLEntry(entry);
            auto casted_self = static_cast< const ConcreteType & >(self);
            if (casted_self == raw.type)
                handle_entry(raw);
        }

        VAST_CHECK(out.has_value(), "Data layout query did not produce a value!");
        return *out;
    }

} // namespace vast

/// Include the generated interface declarations.
#include "vast/Interfaces/DefaultDataLayoutTypeInterface.h.inc"

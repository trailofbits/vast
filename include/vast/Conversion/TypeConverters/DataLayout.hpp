// Copyright (c) 2021-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <mlir/Dialect/DLTI/DLTI.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/IR/BuiltinDialect.h>

#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/TypeSupport.h>
#include <mlir/Interfaces/DataLayoutInterfaces.h>
VAST_UNRELAX_WARNINGS

#include "vast/Util/Common.hpp"
#include "vast/Util/DataLayout.hpp"

#include <array>

namespace vast::conv::tc {
    // For each type remember its data layout information.
    struct data_layout_blueprint
    {
        using dl_entry_interface = mlir::DataLayoutEntryInterface;
        using dl_entry_attr      = mlir::DataLayoutEntryAttr;

        void add(mlir_type type, mlir_attr attr) {
            if (!attr) {
                return;
            }

            auto it = entries.find(type);
            if (it != entries.end()) {
                VAST_CHECK(
                    it->second == attr,
                    "New dl entry for type: {0} would make dl incosistent: {1} != {2}.", type,
                    it->second, attr
                );
                return;
            }

            entries.try_emplace(type, attr);
        }

        mlir_attr wrap(mcontext_t &mctx) const {
            std::vector< dl_entry_interface > flattened;
            for (const auto &[t, e] : entries) {
                auto wrapped = dl_entry_attr::get(t, e);
                VAST_ASSERT(wrapped);
                flattened.push_back(wrapped);
            }

            return mlir::DataLayoutSpecAttr::get(&mctx, flattened);
        }

        llvm::DenseMap< mlir_type, mlir_attr > entries;
    };

    // Each dialect can have its own encoding of data layout entries.
    // Therefore if we do conversion, we need to convert data layout in
    // the correct way.
    // `make( ... )` can return empty attribute which means that the type should
    // not be present in the new data layout.

    // LLVM types should have `abi:pref` where `pref` is not required, but we
    // are going to emit it anyways to make things easier.
    // This whole functionality is 100% dependent on internals of `LLVM::` dialect
    // that are not exposed anywhere and are probably free to change on version bumps.
    struct llvm_dl_entry_helper
    {
        static mlir_attr
        make(mcontext_t &mctx, mlir_type llvm_type, const dl::DLEntry &old_entry) {
            // TODO(conv:tc): Issue #435.
            //                This has more complicated rules, consult `LLVM` dialect
            //                sources.
            if (mlir::isa< mlir::LLVM::LLVMStructType >(llvm_type)) {
                return {};
            }

            auto vector_type = mlir::VectorType::get({ 2 }, mlir::IntegerType::get(&mctx, 32));
            std::array< unsigned, 2 > entries = {
                old_entry.abi_align,
                // We currently do not have preferred option in `DLEntry`.
                old_entry.abi_align
            };

            return mlir::DenseIntElementsAttr::get(vector_type, entries);
        }
    };

    struct vast_dl_entry_helper
    {
        static mlir_attr
        make(mcontext_t &mctx, mlir_type vast_type, const dl::DLEntry &old_entry) {
            auto new_entry = old_entry;
            new_entry.type = vast_type;
            return new_entry.create_raw_attr(mctx);
        }
    };

    struct builtin_dl_entry_helper
    {
        static mlir_attr make(mcontext_t &, mlir_type, const dl::DLEntry &) {
            // Builtin types cannot be present in the data layout.
            return {};
        }
    };

    static inline mlir_attr make_entry(mlir_type trg_type, const dl::DLEntry &old_entry) {
        auto &mctx       = *trg_type.getContext();
        auto trg_dialect = &trg_type.getDialect();

        if (mlir::isa< mlir::BuiltinDialect >(trg_dialect)) {
            return builtin_dl_entry_helper::make(mctx, trg_type, old_entry);
        }

        if (mlir::isa< mlir::LLVM::LLVMDialect >(trg_dialect)) {
            return llvm_dl_entry_helper::make(mctx, trg_type, old_entry);
        }

        // TODO(conv): Add helper function to check if dialect belongs
        //             to VAST suite and then assert on fallthrough?
        return vast_dl_entry_helper::make(mctx, trg_type, old_entry);
    }

    // This is leaky abstraction of our data layout implementation, so maybe
    // move this to `Util/DataLayout.hpp`?
    static inline auto convert_data_layout_attrs(auto &type_converter) {
        return [&type_converter](mlir::DataLayoutSpecInterface spec) {
            data_layout_blueprint bp;
            for (auto e : spec.getEntries()) {
                auto dl_entry = dl::DLEntry(e);
                auto trg_type = type_converter.convert_type_to_type(dl_entry.type);
                // What does this imply?
                if (!trg_type) {
                    continue;
                }

                bp.add(*trg_type, make_entry(*trg_type, std::move(dl_entry)));
            }
            return bp.wrap(type_converter.get_context());
        };
    }

} // namespace vast::conv::tc

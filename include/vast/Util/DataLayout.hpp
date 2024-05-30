// Copyright (c) 2021-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <clang/AST/ASTContext.h>
#include <mlir/Dialect/DLTI/DLTI.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/TypeSupport.h>
#include <mlir/Interfaces/DataLayoutInterfaces.h>
VAST_UNRELAX_WARNINGS

#include "vast/Util/Common.hpp"
#include "vast/Dialect/Core/CoreOps.hpp"

#include <type_traits>

namespace vast::dl {
    // We are currently using `DLTI` dialect to help encoding data layout information,
    // however in the future custom attributes will be probably preferable.
    // Each entry is mapping `hl::Type -> uint32_t` and in the IR it is encoded as
    // attribute of `ModuleOp`.
    // TODO(lukas): Add alignment information & possibly ABI lowering relevant info?
    struct DLEntry
    {
        using bitwidth_t = uint32_t;
        using type_size_t = llvm::TypeSize;

        mlir_type type;
        bitwidth_t bw        = 0;
        bitwidth_t abi_align = 0;

        DLEntry(mlir_type type, bitwidth_t bw, bitwidth_t abi_align)
            : type(type), bw(bw), abi_align(abi_align) {}

        DLEntry(mlir_type type, mlir::DictionaryAttr dict_attr)
            : type(type)
            , bw(extract(dict_attr, bw_key()))
            , abi_align(extract(dict_attr, abi_align_key())) {}

        DLEntry(const mlir::DataLayoutEntryInterface &attr)
            : DLEntry(
                mlir::dyn_cast< mlir_type >(attr.getKey()),
                mlir::dyn_cast< mlir::DictionaryAttr >(attr.getValue())
            ) {
            VAST_ASSERT(type);
        }

      private:
        static mlir_type bw_type(mcontext_t &mctx) { return mlir::IntegerType::get(&mctx, 32); }

        static llvm::StringRef bw_key() { return "vast.dl.bw"; }

        static llvm::StringRef abi_align_key() { return "vast.abi_align.key"; }

        static bitwidth_t extract(mlir::DictionaryAttr dict_attr, llvm::StringRef key) {
            // Defensive check as there is path in ctors that does not guarantee this.
            VAST_ASSERT(dict_attr);

            auto named_attr = dict_attr.getNamed(key);
            VAST_ASSERT(named_attr);

            auto int_attr = mlir::dyn_cast< mlir::IntegerAttr >(named_attr->getValue());
            return static_cast< bitwidth_t >(int_attr.getInt());
        }

        template< typename T >
        static T extract_as(mlir::DictionaryAttr dict_attr, llvm::StringRef key) {
            return cast< T >(extract(dict_attr, key));
        }

        mlir::StringAttr wrap_str(mcontext_t &mctx, llvm::StringRef str_value) const {
            return mlir::StringAttr::get(&mctx, str_value);
        }

      public:
        mlir::Attribute create_raw_attr(mcontext_t &mctx) const {
            // TODO(lukas): There is `UI64Attr` in `IR/OpBase.td` not sure how to include it
            //              though.
            auto to_attr = [&](auto what) {
                return mlir::IntegerAttr::get(bw_type(mctx), llvm::APInt(32, what));
            };

            std::array< mlir::NamedAttribute, 2 > all = {
                mlir::NamedAttribute(wrap_str(mctx, bw_key()), to_attr(bw)),
                mlir::NamedAttribute(wrap_str(mctx, abi_align_key()), to_attr(abi_align))
            };

            return mlir::DictionaryAttr::get(&mctx, all);
        }

        template< typename T >
        static T cast(bitwidth_t bw) {
            if constexpr (std::is_same_v< T, type_size_t >)
                return type_size_t::getFixed(static_cast< uint64_t >(bw));
            else
                return static_cast< T >(bw);
        }

        // Wrap information in this object as `mlir::Attribute`, which is not attached yet
        // to anything.
        mlir::DataLayoutEntryInterface wrap(mcontext_t &mctx) const {
            return mlir::DataLayoutEntryAttr::get(type, create_raw_attr(mctx));
        }

        bool operator==(const DLEntry &o) const = default;
    };

    void filter_data_layout(core::module mod, auto &&filter) {
        auto dl = mod.getDataLayoutSpec();

        auto filtered_entries = llvm::to_vector(
            llvm::make_filter_range(dl.getEntries(), std::forward< decltype(filter) >(filter))
        );

        mod->setAttr(
            mlir::DLTIDialect::kDataLayoutAttrName,
            mlir::DataLayoutSpecAttr::get(mod.getContext(), filtered_entries)
        );
    }

    // For each type remember its data layout information.
    struct DataLayoutBlueprint
    {
        bool try_emplace(mlir_type mty, const clang::Type *aty, const acontext_t &actx) {
            // For other types this should be good-enough for now
            auto info      = actx.getTypeInfo(aty);
            auto bw        = static_cast< uint32_t >(info.Width);
            auto abi_align = static_cast< uint32_t >(info.Align);
            return std::get< 1 >(entries.try_emplace(mty, dl::DLEntry{ mty, bw, abi_align }));
        }

        void add(mlir_type type, dl::DLEntry entry) {
            auto it = entries.find(type);
            if (it != entries.end()) {
                VAST_CHECK(
                    entry == it->second,
                    "Insertion of dl::DLEntry would make DLBlueprint inconsistent."
                );
            }
            entries.try_emplace(type, entry);
        }

        auto wrap(mcontext_t &mctx) const {
            std::vector< mlir::DataLayoutEntryInterface > flattened;
            for (const auto &[_, e] : entries) {
                flattened.push_back(e.wrap(mctx));
            }
            return mlir::DataLayoutSpecAttr::get(&mctx, flattened);
        }

        llvm::DenseMap< mlir_type, dl::DLEntry > entries;
    };

    template< typename Stream >
    auto operator<<(Stream &os, const DataLayoutBlueprint &dl) -> decltype(os << "") {
        for (const auto &[ty, entry] : dl.entries) {
            os << ty << " ";
            os << llvm::formatv("[ bw: {}, abi_align: {} ]\n", entry.bw, entry.abi_align);
        }
        return os;
    }

} // namespace vast::dl

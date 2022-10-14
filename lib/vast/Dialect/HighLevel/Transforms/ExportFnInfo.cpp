// Copyright (c) 2022-present, Trail of Bits, Inc.

#include "vast/Dialect/HighLevel/Passes.hpp"

VAST_RELAX_WARNINGS
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/JSON.h>
#include <llvm/Support/MemoryBuffer.h>
#include <mlir/Analysis/DataLayoutAnalysis.h>
#include <mlir/Target/LLVMIR/Dialect/All.h>
VAST_UNRELAX_WARNINGS

#include "PassesDetails.hpp"

#include <vast/Dialect/HighLevel/HighLevelDialect.hpp>
#include <vast/Dialect/HighLevel/HighLevelOps.hpp>
#include <vast/Util/Symbols.hpp>
#include <vast/Util/TypeSwitch.hpp>

namespace vast::hl
{
    llvm::json::Object json_type_entry(const mlir::DataLayout &dl, mlir::Type type);

    //
    // generic type entry
    //
    struct TypeEntryBase {
        llvm::json::Object raw;
        mlir::Type type;

        TypeEntryBase(mlir::Type type) : type(type) {}

        llvm::json::Object take() && { return std::move(raw); }

        TypeEntryBase &name(const std::string &name) {
            raw["type"] = name;
            return *this;
        }

        TypeEntryBase &size(const mlir::DataLayout &dl) {
            raw["size"] = dl.getTypeSizeInBits(type);
            return *this;
        }

        TypeEntryBase &size(std::uint64_t s) {
            raw["size"] = s;
            return *this;
        }

        TypeEntryBase &emit() { return *this; }
    };

    TypeEntryBase type_entry(const mlir::DataLayout &dl, mlir::Type type);

    //
    // dialect type entry emits type mnemonic name
    //
    template< typename DialectType >
    struct DialectTypeEntry : TypeEntryBase {
        using Base = TypeEntryBase;
        DialectTypeEntry(DialectType type) : Base(type) {}

        DialectType in_dialect() { return type.cast< DialectType >(); }

        DialectTypeEntry &name() {
            raw["type"] = in_dialect().getMnemonic();
            return *this;
        }

        TypeEntryBase &emit() { return name().Base::emit(); }
    };

    template< typename DialectType >
    DialectTypeEntry(DialectType) -> DialectTypeEntry< DialectType >;

    //
    // void type entry
    //
    template< typename DialectType >
    struct VoidTypeEntry : DialectTypeEntry< DialectType > {
        using Base = DialectTypeEntry< DialectType >;
        VoidTypeEntry(DialectType t) : Base(t) {}

        TypeEntryBase &emit() { return Base::emit().size(0u); }
    };

    template< typename DialectType >
    VoidTypeEntry(DialectType) -> VoidTypeEntry< DialectType >;

    //
    // entry to emit type qualifiers
    //
    template< typename DialectType >
    struct WithModifiersEntry : DialectTypeEntry< DialectType > {
        using Base = DialectTypeEntry< DialectType >;
        WithModifiersEntry(DialectType t) : Base(t) {}

        using Base::in_dialect;
        using Base::raw;

        WithModifiersEntry &qualifiers() {
            // raw["const"]    = in_dialect().isConst();
            // raw["volatile"] = in_dialect().isVolatile();
            // TODO restrict
            // TODO static
            return *this;
        }

        TypeEntryBase &emit() { return qualifiers().Base::emit(); }
    };

    template< typename DialectType >
    WithModifiersEntry(DialectType) -> WithModifiersEntry< DialectType >;

    //
    // scalar values are dialect types with modifiers
    //
    template< typename DialectType >
    struct ScalarTypeEntry : WithModifiersEntry< DialectType > {
        using Base = WithModifiersEntry< DialectType >;
        ScalarTypeEntry(DialectType t) : Base(t) {}

        TypeEntryBase &emit(const mlir::DataLayout &dl) {
            return Base::emit().size(dl);
        }
    };

    template< typename DialectType >
    ScalarTypeEntry(DialectType) -> ScalarTypeEntry< DialectType >;

    //
    // type entry with subelement
    //
    template< typename DialectType >
    struct WithElementType : WithModifiersEntry< DialectType > {
        using Base = WithModifiersEntry< DialectType >;
        WithElementType(DialectType t) : Base(t) {}

        using Base::in_dialect;
        using Base::raw;

        template< typename ElementTypeEntry >
        WithElementType &element_type(ElementTypeEntry &&elem) {
            raw["element_type"] = std::forward< ElementTypeEntry >(elem);
            return *this;
        }

        WithElementType &element_type(const mlir::DataLayout &dl) {
            return element_type(json_type_entry(dl, in_dialect().getElementType()));
        }

        TypeEntryBase &emit(const mlir::DataLayout &dl) {
            return element_type(dl).Base::emit();
        }
    };

    template< typename DialectType >
    WithElementType(DialectType) -> WithElementType< DialectType >;

    //
    // pointer entry is dialect entry with modifiers and sub_element entry
    //
    template< typename DialectType >
    struct PointerTypeEntry : WithElementType< DialectType > {
        using Base = WithElementType< DialectType >;
        PointerTypeEntry(DialectType t) : Base(t) {}

        using Base::emit;
    };

    template< typename DialectType >
    PointerTypeEntry(DialectType) -> PointerTypeEntry< DialectType >;

    //
    // lvalue type entry
    //
    template< typename DialectType >
    struct LValueTypeEntry : DialectTypeEntry< DialectType > {
        using Base = DialectTypeEntry< DialectType >;
        LValueTypeEntry(DialectType t) : Base(t) {}

        using Base::raw;
        using Base::in_dialect;

        TypeEntryBase &emit(const mlir::DataLayout &dl) {
            raw = type_entry(dl, in_dialect().getElementType()).raw;
            return *this;
        }
    };

    //
    // type entry dispatcher
    //
    TypeEntryBase type_entry(const mlir::DataLayout &dl, mlir::Type type) {
        auto ptr_entry    = [&](auto ty) { return PointerTypeEntry(ty).emit(dl); };
        auto lvalue_entry = [&](auto ty) { return LValueTypeEntry(ty).emit(dl); };
        auto void_entry   = [&](auto ty) { return VoidTypeEntry(ty).emit(); };
        auto scalar_entry = [&](auto ty) { return ScalarTypeEntry(ty).emit(dl); };

        return TypeSwitch< mlir::Type, TypeEntryBase >(type)
            .Case< hl::LValueType >(lvalue_entry)
            .Case< hl::PointerType >(ptr_entry)
            .Case< hl::VoidType >(void_entry)
            .Case(scalar_types{}, scalar_entry);
    }

    llvm::json::Object json_type_entry(const mlir::DataLayout &dl, mlir::Type type) {
        return type_entry(dl, type).take();
    }

    struct ExportFnInfo : ExportFnInfoBase< ExportFnInfo > {
        void runOnOperation() override {
            mlir::ModuleOp mod = this->getOperation();

            llvm::json::Object top;

            // TODO use FunctionOpInterface instead of specific operation
            util::functions(mod, [&](FuncOp fn) {
                const auto &dl_analysis = this->getAnalysis< mlir::DataLayoutAnalysis >();
                const auto &dl          = dl_analysis.getAtOrAbove(mod);

                llvm::json::Array args;
                for (auto &arg_type : fn.getArgumentTypes()) {
                    args.push_back(json_type_entry(dl, arg_type));
                }

                llvm::json::Array rets;
                for (auto &ret_type : fn.getResultTypes()) {
                    rets.push_back(json_type_entry(dl, ret_type));
                }

                llvm::json::Object current;
                current["rets"] = std::move(rets);
                current["args"] = std::move(args);

                top[fn.getName().str()] = std::move(current);
            });

            auto value = llvm::formatv("{0:2}", llvm::json::Value(std::move(top)));
            // If destination filename was supplied by the user.
            if (!this->o.empty()) {
                std::error_code ec;

                llvm::raw_fd_ostream out(this->o, ec, llvm::sys::fs::OF_Text);
                VAST_ASSERT(!ec);
                out << std::move(value);
            } else {
                llvm::outs() << std::move(value);
            }
        }
    };

} // namespace vast::hl

std::unique_ptr< mlir::Pass > vast::hl::createExportFnInfoPass() {
    return std::make_unique< ExportFnInfo >();
}

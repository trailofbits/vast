// Copyright (c) 2021-present, Trail of Bits, Inc.

#include "vast/Dialect/HighLevel/HighLevelAttributes.hpp"
#include "vast/Dialect/HighLevel/HighLevelTypes.hpp"
#include "vast/Dialect/HighLevel/HighLevelOps.hpp"
#include "vast/Util/TypeList.hpp"
#include <sstream>

VAST_RELAX_WARNINGS
#include <llvm/ADT/TypeSwitch.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/DialectImplementation.h>
VAST_RELAX_WARNINGS

#include "vast/Dialect/Core/CoreOps.hpp"

namespace vast::hl
{
    mlir_type strip_elaborated(mlir_value v)
    {
        return strip_elaborated(v.getType());
    }

    mlir_type strip_elaborated(mlir_type t)
    {
        if (auto e = mlir::dyn_cast_or_null<hl::ElaboratedType>(t))
            return e.getElementType();
        return t;
    }

    mlir_type strip_value_category(mlir_value v)
    {
        return strip_value_category(v.getType());
    }

    mlir_type strip_value_category(mlir_type t)
    {
        if (auto e = mlir::dyn_cast< hl::LValueType >(t))
            return e.getElementType();
        return t;
    }

    mlir_type strip_complex(mlir_value v)
    {
        return strip_complex(v.getType());
    }

    mlir_type strip_complex(mlir_type t)
    {
        if (auto c = mlir::dyn_cast< hl::ComplexType >(t))
            return c.getElementType();
        return t;
    }

    mlir_type getBottomTypedefType(TypedefType def, core::module mod)
    {
        return getBottomTypedefType(getTypedefType(def, mod), mod);
    }

    mlir_type getBottomTypedefType(mlir_type type, core::module mod)
    {
        if (auto def = mlir::dyn_cast_or_null< TypedefType >(strip_elaborated(type)))
            return getBottomTypedefType(def, mod);
        return type;
    }

    mlir_type getTypedefType(TypedefType type, core::module mod)
    {
        auto name = type.getName();
        mlir_type result;

        // TODO: probably needs scope
        mod.walk([&] (TypeDefOp op) {
            if (op.getSymName() == name) { result = op.getType(); }
        });

        if (result) {
            return result;
        }

        return {};
    }

    auto name_of_record(mlir_type t) -> std::optional< std::string >
    {
        auto naked_type = strip_elaborated(strip_value_category(t));
        if (auto record_type = mlir::dyn_cast< hl::RecordType >(naked_type))
            return record_type.getName().str();
        return {};
    }

    core::FunctionType getFunctionType(mlir_type type, operation from) {
        if (auto ty = type.dyn_cast< core::FunctionType >()) {
            return ty;
        } else if (auto ty = dyn_cast< ElementTypeInterface >(type)) {
            return getFunctionType(ty.getElementType(), from);
        } else if (auto ty = type.dyn_cast< TypedefType >()) {
            auto mod = from->getParentOfType< core::module >();
            return getFunctionType(getTypedefType(ty, mod), from);
        } else {
            return {};
        }
    }

    core::FunctionType getFunctionType(mlir::CallInterfaceCallable callee, operation from) {
        if (!callee) {
            return {};
        }

        if (auto sym = callee.dyn_cast< mlir::SymbolRefAttr >()) {
            auto st = core::get_effective_symbol_table_for< core::func_symbol >(from);
            VAST_CHECK(st, "No effective symbol table found for function type resolution.");

            auto fn = st->lookup< core::func_symbol >(sym.getRootReference());
            VAST_CHECK(fn, "Function {} not present in the symbol table.", sym.getRootReference());

            return mlir::cast< FuncOp >(fn).getFunctionType();
        }

        if (auto value = callee.dyn_cast< mlir_value >()) {
            return getFunctionType(value.getType(), from);
        }

        return {};
    }


    void HighLevelDialect::registerTypes() {
        addTypes<
            #define GET_TYPEDEF_LIST
            #include "vast/Dialect/HighLevel/HighLevelTypes.cpp.inc"
        >();
    }

    bool isBoolType(mlir_type type)
    {
        return type.isa< BoolType >();
    }

    bool isIntegerType(mlir_type type)
    {
        return util::is_one_of< integer_types >(type);
    }

    bool isFloatingType(mlir_type type)
    {
        return util::is_one_of< floating_types >(type);
    }

    bool isSigned(mlir_type type)
    {
        if (isBoolType(type)) {
            return false;
        }

        if (auto builtin_type = type.dyn_cast< mlir::IntegerType >())
            return builtin_type.isSigned();

        VAST_ASSERT(isIntegerType(type));
        return util::dispatch< integer_types, bool >(type, [] (auto ty) {
            auto quals = ty.getQuals();
            return !quals || !quals.hasUnsigned();
        });
    }

    bool isUnsigned(mlir_type type)
    {
        return !(isSigned(type));
    }

    bool isHighLevelType(mlir_type type)
    {
        return util::is_one_of< high_level_types >(type);
    }

} // namespace vast::hl

using StringRef = llvm::StringRef; // to fix missing namespace in generated file

#define  GET_TYPEDEF_CLASSES
#include "vast/Dialect/HighLevel/HighLevelTypes.cpp.inc"

namespace vast::hl
{

    auto ArrayType::dim_and_type() -> std::tuple< dimensions_t, mlir_type >
    {
        dimensions_t dims;
        // If this ever is generalised investigate if `SubElementTypeInterface` can be used
        // do this recursion?
        auto collect = [&](ArrayType arr, auto &self) -> mlir_type {
            dims.push_back(arr.getSize());
            if (auto nested = arr.getElementType().dyn_cast< ArrayType >())
                return self(nested, self);
            return arr.getElementType();
        };
        return { std::move(dims), collect(*this, collect) };
    }

    PointerType PointerType::get(mlir_type element_type) {
        return PointerType::get(element_type.getContext(), element_type);
    }

} // namespace vast::hl

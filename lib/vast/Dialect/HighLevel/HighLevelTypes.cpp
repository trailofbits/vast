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

namespace vast::hl
{
    Type strip_elaborated(mlir::Value v)
    {
        return strip_elaborated(v.getType());
    }

    Type strip_elaborated(Type t)
    {
        if (auto e = mlir::dyn_cast<hl::ElaboratedType>(t))
            return e.getElementType();
        return t;
    }

    Type strip_value_category(mlir::Value v)
    {
        return strip_value_category(v.getType());
    }

    mlir_type strip_value_category(mlir_type t)
    {
        if (auto e = mlir::dyn_cast< hl::LValueType >(t))
            return e.getElementType();
        return t;
    }


    Type getBottomTypedefType(TypedefType def, vast_module mod) {
        auto type = getTypedefType(def, mod);
        if (auto ty = strip_elaborated(type).dyn_cast< TypedefType >()) {
            return getBottomTypedefType(ty, mod);
        }
        return type;
    }

    Type getTypedefType(TypedefType type, vast_module mod) {
        auto name = type.getName();
        for (const auto &op : mod) {
            if (auto def = mlir::dyn_cast< TypeDefOp >(&op)) {
                if (def.getName() == name) {
                    return def.getType();
                }
            }
        }

        VAST_UNREACHABLE("unknown typedef name");
    }

    auto name_of_record(mlir_type t) -> std::optional< std::string >
    {
        auto naked_type = strip_elaborated(strip_value_category(t));
        auto record_type = mlir::dyn_cast< hl::RecordType >(naked_type);
        if (record_type)
            return record_type.getName().str();
        return {};
    }

    mlir::FunctionType getFunctionType(Type type, vast_module mod) {
        if (auto ty = type.dyn_cast< mlir::FunctionType >())
            return ty;
        if (auto ty = type.dyn_cast< LValueType >())
            return getFunctionType(ty.getElementType(), mod);
        if (auto ty = type.dyn_cast< ParenType >())
            return getFunctionType(ty.getElementType(), mod);
        if (auto ty = type.dyn_cast< PointerType >())
            return getFunctionType(ty.getElementType(), mod);
        if (auto ty = type.dyn_cast< TypedefType >())
            return getFunctionType(getTypedefType(ty, mod), mod);
        if (auto ty = type.dyn_cast< ElaboratedType >())
            return getFunctionType(ty.getElementType(), mod);

        VAST_UNREACHABLE("unknown type to extract function type");
    }

    mlir::FunctionType getFunctionType(Value callee) {
        auto op  = callee.getDefiningOp();
        auto mod = op->getParentOfType< vast_module >();
        return getFunctionType(callee.getType(), mod);
    }

    mlir::FunctionType getFunctionType(mlir::CallOpInterface call) {
        auto mod = call->getParentOfType< vast_module >();
        return getFunctionType(call.getCallableForCallee(), mod);
    }

    mlir::FunctionType getFunctionType(mlir::CallInterfaceCallable callee, vast_module mod) {
        if (auto sym = callee.dyn_cast< mlir::SymbolRefAttr >()) {
            return mlir::dyn_cast_or_null< FuncOp >(
                mlir::SymbolTable::lookupSymbolIn(mod, sym)
            ).getFunctionType();
        }

        if (auto value = callee.dyn_cast< Value >()) {
            return getFunctionType(value.getType(), mod);
        }

        VAST_UNREACHABLE("unknown callee type");
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
    template< typename T >
    using walk_fn = llvm::function_ref< void( T ) >;

    using walk_types = walk_fn< mlir_type >;
    using walk_attrs = walk_fn< mlir::Attribute >;


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

} // namespace vast::hl

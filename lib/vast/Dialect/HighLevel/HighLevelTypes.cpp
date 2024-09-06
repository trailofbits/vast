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

    mlir_type getBottomTypedefType(TypedefType def, vast_module mod)
    {
        return getBottomTypedefType(getTypedefType(def, mod), mod);
    }

    mlir_type getBottomTypedefType(mlir_type type, vast_module mod)
    {
        if (auto def = mlir::dyn_cast_or_null< TypedefType >(strip_elaborated(type)))
            return getBottomTypedefType(def, mod);
        return type;
    }

    mlir_type getTypedefType(TypedefType type, vast_module mod)
    {
        auto name = type.getName();
        mlir_type result;

        // TODO: probably needs scope
        mod.walk([&] (TypeDefOp op) {
            if (op.getName() == name) { result = op.getType(); }
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

    core::FunctionType getFunctionType(mlir_type type, vast_module mod) {
        if (auto ty = type.dyn_cast< core::FunctionType >())
            return ty;
        if (auto ty = dyn_cast< ElementTypeInterface >(type))
            return getFunctionType(ty.getElementType(), mod);
        if (auto ty = type.dyn_cast< TypedefType >())
            return getFunctionType(getTypedefType(ty, mod), mod);

        return {};
    }

    core::FunctionType getFunctionType(mlir_value callee) {
        auto op  = callee.getDefiningOp();
        auto mod = op->getParentOfType< vast_module >();
        return getFunctionType(callee.getType(), mod);
    }

    core::FunctionType getFunctionType(mlir::CallOpInterface call) {
        auto mod = call->getParentOfType< vast_module >();
        return getFunctionType(call.getCallableForCallee(), mod);
    }

    core::FunctionType getFunctionType(mlir::CallInterfaceCallable callee, vast_module mod) {
        if (!callee) {
            return {};
        }

        if (auto sym = callee.dyn_cast< mlir::SymbolRefAttr >()) {
            return mlir::dyn_cast_or_null< FuncOp >(
                mlir::SymbolTable::lookupSymbolIn(mod, sym)
            ).getFunctionType();
        }

        if (auto value = callee.dyn_cast< mlir_value >()) {
            return getFunctionType(value.getType(), mod);
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

    bool RecordType::isRecordType() const {
        return true;
    }

    bool VoidType::isVoidType() const {
        return true;
    }

    bool BoolType::isScalarType() const {
        return true;
    }

    bool CharType::isScalarType() const {
        return true;
    }

    bool ShortType::isScalarType() const {
        return true;
    }

    bool IntType::isScalarType() const {
        return true;
    }

    bool LongType::isScalarType() const {
        return true;
    }

    bool LongLongType::isScalarType() const {
        return true;
    }

    bool Int128Type::isScalarType() const {
        return true;
    }

    bool HalfType::isScalarType() const {
        return true;
    }

    bool BFloat16Type::isScalarType() const {
        return true;
    }

    bool FloatType::isScalarType() const {
        return true;
    }

    bool DoubleType::isScalarType() const {
        return true;
    }

    bool LongDoubleType::isScalarType() const {
        return true;
    }

    bool Float128Type::isScalarType() const {
        return true;
    }

    bool ComplexType::isScalarType() const {
        return true;
    }

    bool PointerType::isScalarType() const {
        return true;
    }

    bool VectorType::isVectorType() const {
        return true;
    }

    bool LValueType::isScalarType() const {
        return mlir::dyn_cast< ast::TypeInterface >(getElementType()).isScalarType();
    }

    bool RValueType::isScalarType() const {
        return mlir::dyn_cast< ast::TypeInterface >(getElementType()).isScalarType();
    }

    bool ReferenceType::isScalarType() const {
        return false;
    }
} // namespace vast::hl

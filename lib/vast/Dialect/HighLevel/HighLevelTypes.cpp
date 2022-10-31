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
    Type LValueType::replaceImmediateSubElements(llvm::ArrayRef<mlir::Attribute>, llvm::ArrayRef<mlir::Type>) const {
        VAST_UNIMPLEMENTED;
    }

    Type ElaboratedType::replaceImmediateSubElements(llvm::ArrayRef<mlir::Attribute>, llvm::ArrayRef<mlir::Type>) const {
        VAST_UNIMPLEMENTED;
    }

    Type PointerType::replaceImmediateSubElements(llvm::ArrayRef<mlir::Attribute>, llvm::ArrayRef<mlir::Type>) const {
        VAST_UNIMPLEMENTED;
    }

    Type ArrayType::replaceImmediateSubElements(llvm::ArrayRef<mlir::Attribute>, llvm::ArrayRef<mlir::Type>) const {
        VAST_UNIMPLEMENTED;
    }

    Type getBottomTypedefType(TypedefType def, Module mod) {
        auto type = getTypedefType(def, mod);
        if (auto ty = type.dyn_cast< TypedefType >()) {
            return getBottomTypedefType(ty, mod);
        }
        return type;
    }

    Type getTypedefType(TypedefType type, Module mod) {
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

    mlir::FunctionType getFunctionType(Type type, Module mod) {
        if (auto ty = type.dyn_cast< mlir::FunctionType >())
            return ty;
        if (auto ty = type.dyn_cast< LValueType >())
            return getFunctionType(ty.getElementType(), mod);
        if (auto ty = type.dyn_cast< ParenType >())
            return getFunctionType(ty.getElementType(), mod);
        if (auto ty = type.dyn_cast< PointerType >())
            return getFunctionType(ty.getElementType(), mod);
        if (auto ty = type.dyn_cast< TypedefType >()) {
            return getFunctionType(getTypedefType(ty, mod), mod);
        }

        VAST_UNREACHABLE("unknown type to extract function type");
    }

    mlir::FunctionType getFunctionType(Value callee) {
        auto op  = callee.getDefiningOp();
        auto mod = op->getParentOfType< Module >();
        return getFunctionType(callee.getType(), mod);
    }

    mlir::FunctionType getFunctionType(mlir::CallOpInterface call) {
        auto mod = call->getParentOfType< Module >();
        return getFunctionType(call.getCallableForCallee(), mod);
    }

    mlir::FunctionType getFunctionType(mlir::CallInterfaceCallable callee, Module mod) {
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

    bool isBoolType(mlir::Type type)
    {
        return type.isa< BoolType >();
    }

    bool isIntegerType(mlir::Type type)
    {
        return util::is_one_of< integer_types >(type);
    }

    bool isFloatingType(mlir::Type type)
    {
        return util::is_one_of< floating_types >(type);
    }

    bool isSigned(mlir::Type type)
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

    bool isUnsigned(mlir::Type type)
    {
        return !(isSigned(type));
    }

    bool isHighLevelType(mlir::Type type)
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

    using walk_types = walk_fn< mlir::Type >;
    using walk_attrs = walk_fn< mlir::Attribute >;

    void LValueType::walkImmediateSubElements(walk_attrs, walk_types tys) const {
        tys( this->getElementType() );
    }

    void ElaboratedType::walkImmediateSubElements(walk_attrs, walk_types tys) const {
        tys( this->getElementType() );
    }


    void PointerType::walkImmediateSubElements(walk_attrs, walk_types tys) const {
        tys( this->getElementType() );
    }

    void ArrayType::walkImmediateSubElements(walk_attrs, walk_types tys) const {
        tys( this->getElementType() );
    }

    auto ArrayType::dim_and_type() -> std::tuple< dimensions_t, mlir::Type >
    {
        dimensions_t dims;
        // If this ever is generalised investigate if `SubElementTypeInterface` can be used
        // do this recursion?
        auto collect = [&](ArrayType arr, auto &self) -> mlir::Type {
            dims.push_back(arr.getSize());
            if (auto nested = arr.getElementType().dyn_cast< ArrayType >())
                return self(nested, self);
            return arr.getElementType();
        };
        return { std::move(dims), collect(*this, collect) };
    }

} // namespace vast::hl

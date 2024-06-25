// Copyright (c) 2022-present, Trail of Bits, Inc.

#include "vast/Util/Warnings.hpp"

#include "vast/Dialect/ABI/ABIDialect.hpp"
#include "vast/Dialect/ABI/ABIOps.hpp"

#include "vast/Util/Region.hpp"

VAST_RELAX_WARNINGS
#include <mlir/IR/Builders.h>
VAST_UNRELAX_WARNINGS

#include "vast/Util/Dialect.hpp"

#include "vast/Dialect/Core/Func.hpp"

namespace vast::abi
{
    ParseResult parseFunctionSignatureAndBody(
        Parser &parser, Attribute &funcion_type,
        mlir::NamedAttrList &attr_dict, Region &body
    ) {
        return core::parseFunctionSignatureAndBody( parser, funcion_type, attr_dict, body );
    }


    void printFunctionSignatureAndBody(
        Printer &printer, auto op,
        Attribute attr, mlir::DictionaryAttr dict, Region &body
    ) {
        return core::printFunctionSignatureAndBodyImpl( printer, op, attr, dict, body );
    }

    mlir::CallInterfaceCallable CallOp::getCallableForCallee()
    {
        return core::get_callable_for_callee(*this);
    }

    mlir::Operation::operand_range CallOp::getArgOperands()
    {
        return this->getOperands();
    }

    mlir::MutableOperandRange CallOp::getArgOperandsMutable()
    {
        return mlir::MutableOperandRange(*this, 1, getOperands().size());
    }

    void CallOp::setCalleeFromCallable(mlir::CallInterfaceCallable callee)
    {
        setOperand(0, callee.get< mlir_value >());
    }

    void build_op_with_region(
        Builder &bld, State &st,
        maybe_builder_callback_ref body
    ) {
        Builder::InsertionGuard guard(bld);
        build_region(bld, st, body);
    }

    void PrologueOp::build(
        Builder &bld, State &st, mlir::TypeRange types,
        maybe_builder_callback_ref body
    ) {
        st.addTypes(types);
        return build_op_with_region(bld, st, body);
    }

    void EpilogueOp::build(
        Builder &bld, State &st, mlir::TypeRange types,
        maybe_builder_callback_ref body
    ) {
        st.addTypes(types);
        return build_op_with_region(bld, st, body);
    }

    void CallArgsOp::build(
        Builder &bld, State &st, mlir::TypeRange types,
        maybe_builder_callback_ref body
    ) {
        st.addTypes(types);
        return build_op_with_region(bld, st, body);
    }

    void CallRetsOp::build(
        Builder &bld, State &st, mlir::TypeRange types,
        maybe_builder_callback_ref body
    ) {
        st.addTypes(types);
        return build_op_with_region(bld, st, body);
    }

    mlir::CallInterfaceCallable CallExecutionOp::getCallableForCallee()
    {
        return core::get_callable_for_callee(*this);
    }

    mlir::Operation::operand_range CallExecutionOp::getArgOperands()
    {
        return this->getOperands();
    }

    mlir::MutableOperandRange CallExecutionOp::getArgOperandsMutable()
    {
        return mlir::MutableOperandRange(*this, 1, getOperands().size());
    }

    void CallExecutionOp::setCalleeFromCallable(mlir::CallInterfaceCallable callee)
    {
        setOperand(0, callee.get< mlir_value >());
    }

    SSACFG_REGION_OP( FuncOp );

    logical_result FuncOp::verify() {
        return core::verifyFuncOp(*this);
    }

} // namespace vast::abi

#define GET_OP_CLASSES
#include "vast/Dialect/ABI/ABI.cpp.inc"

// Copyright (c) 2022-present, Trail of Bits, Inc.

#ifndef VAST_DIALECT_IR_ABIOPS
#define VAST_DIALECT_IR_ABIOPS

include "mlir/Interfaces/CastInterfaces.td"

include "mlir/IR/OpBase.td"

include "vast/Dialect/Core/CommonAttrConstraints.td"
include "vast/Dialect/Core/Func.td"

include "vast/Dialect/Core/Interfaces/SymbolInterface.td"
include "vast/Dialect/Core/Interfaces/SymbolTableInterface.td"

def DirectOp
    : ABI_Op< "direct" >
    , Arguments<(ins Variadic<AnyType>:$value)>
    , Results<(outs Variadic<AnyType>:$result)>
{
    let summary = "Pass value directly - usually means by register";
    let description = [{ Pass value directly - usually means by register. }];

    let assemblyFormat = [{ $value attr-dict `:` type($value) `->` type($result) }];
}

def IndirectOp
    : ABI_Op< "indirect" >
    , Arguments<(ins AnyType:$value)>
    , Results<(outs AnyType:$result)>
{
    let summary = "Value is passed indirectly via memory";
    let description = [{ Value is passed indirectly via memory }];

    let assemblyFormat = [{ $value attr-dict `:` type($value) `->` type($result) }];
}

def RetDirectOp
    : ABI_Op< "ret_direct" >
    , Arguments<(ins AnyType:$value)>
    , Results<(outs Variadic<AnyType>:$result)>
{
    let summary = "Value is returned directly.";
    let description = [{ Value is returned directly. }];

    let assemblyFormat = [{ $value attr-dict `:` type($value) `->` type($result) }];
}

class HasBody< string mnemonic >
    : ABI_Op< mnemonic >
    , Results<(outs Variadic<AnyType>:$results)>
{
    let regions = (region AnyRegion:$body);

    let skipDefaultBuilders = 1;
    let builders = [
        OpBuilder<(ins
            "mlir::TypeRange":$types,
            CArg< "maybe_builder_callback_ref", "std::nullopt" >:$bodyBuilder
        )>
    ];

    let assemblyFormat = [{ $body attr-dict `:` type($results) }];
}

def PrologueOp : HasBody< "prologue" >
{
    let summary = "WIP";
    let description = [{ WIP }];
}

def EpilogueOp : HasBody< "epilogue" >
{
    let summary = "WIP";
    let description = [{ WIP }];
}

def CallArgsOp : HasBody< "call_args" >
{
    let summary = "Not implement yet.";
    let description = [{ WIP }];
}

def CallRetsOp : HasBody< "call_rets" >
{
    let summary = "Not implement yet.";
    let description = [{ WIP }];
}

def CallExecutionOp
    : ABI_Op< "call_exec", [DeclareOpInterfaceMethods< CallOpInterface >] >
    , Arguments<(ins
        SymbolRefAttr:$callee,
        Variadic<AnyType>:$args)>
    , Results<(outs AnyType:$result)>
{
    let summary = "WIP";
    let description = [{ WIP }];

    let regions = (region AnyRegion:$body);

    let skipDefaultBuilders = 1;
    let builders = [
        OpBuilder< (ins
            "mlir::FlatSymbolRefAttr":$callee,
            "mlir::TypeRange":$results,
            CArg< "mlir::ValueRange", "{}" >:$operands,
            CArg< "maybe_builder_callback_ref", "std::nullopt" >:$body_builder
        ), [{
            $_state.addTypes(results);
            $_state.addOperands(operands);
            $_state.addAttribute("callee", callee);
            build_op_with_region($_builder, $_state, body_builder);
        }]>,
        OpBuilder< (ins
            "mlir::StringAttr":$callee,
            "mlir::TypeRange":$results,
            CArg<"mlir::ValueRange", "{}">:$operands,
            CArg< "maybe_builder_callback_ref", "std::nullopt" >:$body_builder
        ), [{
            build($_builder, $_state, mlir::FlatSymbolRefAttr::get($_builder.getContext(), callee),
                results, operands, body_builder
            );
        }]>,
        OpBuilder< (ins
            "llvm::StringRef":$callee,
            "mlir::TypeRange":$results,
            CArg<"mlir::ValueRange", "{}">:$operands,
            CArg< "maybe_builder_callback_ref", "std::nullopt" >:$body_builder
        ), [{
            build($_builder, $_state, mlir::FlatSymbolRefAttr::get($_builder.getContext(), callee),
                results, operands, body_builder
            );
        }]>
    ];

    let assemblyFormat = [{
        $callee `(` $args `)` $body attr-dict `:` functional-type($args, $result)
    }];
}

def YieldOp
    : ABI_Op< "yield", [Terminator] >
    , Arguments<(ins Variadic<AnyType>:$values)>
    , Results<(outs Variadic<AnyType>:$result)>
{
    let summary = "WIP";
    let description = [{ WIP }];

    let assemblyFormat = [{ $values attr-dict `:` type($values) `->` type($result) }];
}


def CallOp
  : ABI_Op< "call", [DeclareOpInterfaceMethods<CallOpInterface>] >
  , Arguments<(ins
      FlatSymbolRefAttr:$callee,
      Variadic<AnyType>:$args)>
  , Results<(outs Variadic<AnyType>:$results)>
{
    let summary = "ABI call operation";
    let description = [{ ABI call operation }];

    let skipDefaultBuilders = 1;
    let builders = [
        OpBuilder< (ins
            "mlir::FlatSymbolRefAttr":$callee,
            "mlir::TypeRange":$results,
            CArg< "mlir::ValueRange", "{}" >:$operands
        ), [{
            $_state.addOperands(operands);
            $_state.addAttribute("callee", callee);
            $_state.addTypes(results);
        }]>,
        OpBuilder< (ins
            "mlir::StringAttr":$callee,
            "mlir::TypeRange":$results,
            CArg<"mlir::ValueRange", "{}">:$operands
        ), [{
            build($_builder, $_state, mlir::FlatSymbolRefAttr::get($_builder.getContext(), callee), results, operands);
        }]>,
        OpBuilder< (ins
            "llvm::StringRef":$callee,
            "mlir::TypeRange":$results,
            CArg<"mlir::ValueRange", "{}">:$operands
        ), [{
            build($_builder, $_state, mlir::FlatSymbolRefAttr::get($_builder.getContext(), callee), results, operands);
        }]>
    ];

    let assemblyFormat = [{
        $callee `(` $args `)` attr-dict `:` functional-type( $args, $results )
    }];
}

// TODO(abi): This is mostly copy & paste of hl.func. Once ABI dialect is stable,
//            extract shared parts from both definitions into some helper.
class ABIFuncBase< string mnemonic, list< Trait> traits = [] >
  : Core_FuncBaseOp< ABI_Dialect, mnemonic, traits >
{}

def FuncOp : ABIFuncBase< "func", [
    Core_FuncSymbol,
    Core_ShadowingSymbolTable< [
      [Core_VarSymbol, Core_TypeSymbol, Core_EnumConstantSymbol],
      [Core_ElaboratedTypeSymbol]
    ] >,
    IsolatedFromAbove
] >
{
    let summary = [{ Function with transformed type. }];
}

#endif // VAST_DIALECT_IR_ABIOPS

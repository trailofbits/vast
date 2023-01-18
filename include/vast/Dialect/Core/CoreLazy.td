// Copyright (c) 2022-present, Trail of Bits, Inc.
#ifndef VAST_DIALECT_CORE_CORELAZY
#define VAST_DIALECT_CORE_CORELAZY

include "mlir/IR/OpBase.td"

include "mlir/IR/FunctionInterfaces.td"

include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/CastInterfaces.td"
include "vast/Interfaces/SymbolInterface.td"

def ValueRegion : Region< HasOneBlock, "value region" >;

class lazyEval< string mnemonic, list < Trait > traits = [] >
    : Core_Op< mnemonic, !listconcat(traits, []) >, Results<(outs AnyType:$result)>
{
    let summary = "lazy evaluator";
    let description = [{ VAST lazy evaluator }];
}

def Core_LazyOp : lazyEval< "lazy.op", [] > {
    let summary = "Lazily evaluate a region.";
    let description = [{
        An operation that marks that the region it contains should be evalutated only when necessary.
    }];

    let regions = (region ValueRegion:$lazy);

    let assemblyFormat = [{ $lazy attr-dict `:` type(results) }];
}

class LogicBinOp< string mnemonic, list< Trait > traits = [] >
    : Core_Op< mnemonic, traits >
    , Arguments<(ins AnyType:$lhs, AnyType:$rhs)>
    , Results<(outs AnyType:$result)>
{
    let summary = "VAST core dialect logical binary operation";
    let description = [{
        Core dialect logical binary operation. This operation takes two operands
        and returns one result, each of these is required to be of the same
        type.

        The custom assembly form of the operation is as follows:

        %result = <op> %lhs, %rhs  : type
    }];

    let assemblyFormat = [{ $lhs `,` $rhs attr-dict `:` functional-type(operands, results) }];
}

def BinLAndOp : LogicBinOp<"bin.land", []>;
def BinLOrOp  : LogicBinOp< "bin.lor", []>;

#endif //VAST_DIALECT_CORE_CORELAZY

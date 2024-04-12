// Copyright (c) 2021-present, Trail of Bits, Inc.

#ifndef VAST_DIALECT_IR_LOWLEVELOPS
#define VAST_DIALECT_IR_LOWLEVELOPS

include "mlir/IR/OpBase.td"
include "mlir/IR/BuiltinAttributes.td"

include "mlir/Interfaces/ControlFlowInterfaces.td"

include "vast/Interfaces/SymbolInterface.td"
include "vast/Interfaces/ElementTypeInterface.td"

include "vast/Dialect/Core/CoreTraits.td"

include "vast/Dialect/Core/Func.td"

// TODO(lukas): Add type constraints.
def LowLevel_StructGEPOp
  : LowLevel_Op< "gep", [] >
  , Arguments<(ins AnyType:$record, I32Attr:$idx, StrAttr:$name)>
  , Results<(outs AnyType:$element)>
{
    let summary = "VAST struct gep  operation";
    let description = [{ VAST struct gep operation }];
}

def Subscript
    : LowLevel_Op< "subscript" >
    , Arguments<(ins
        AnyType:$array,
        AnyType:$index)>
    , Results<(outs AnyType:$result)>
{
  let summary = "VAST array subscript operator";
  let description = [{ VAST array subscript operator }];

  let assemblyFormat = [{
    $array `at` ` ` `[` $index `:` type($index) `]` attr-dict
      `:` type($array) `->` type($result)
  }];
}

def ArgAlloca
    : LowLevel_Op< "arg_alloca" >
    , Arguments<(ins AnyType:$fn_arg)>
    , Results<(outs AnyType:$result)>
{
    let summary = "Alloca that holds the function argument.";

    let assemblyFormat = [{
        operands attr-dict `:` functional-type(operands, results)
    }];
}

def Alloca
    : LowLevel_Op< "alloca" >,
      Results<(outs AnyType:$result)>
{
    let summary = "Alloca for a piece of memory. For now does not support array types.";

    let assemblyFormat = [{
        attr-dict `:` type($result)
    }];
}

class PointerPointeeTypeMatch< string ptr, string val >
    : TypesMatchWith< "Inconsistent type between pointer and its value", ptr, val,
                      "(mlir::isa< ElementTypeInterface >($_self))"
                      "? mlir::cast< ElementTypeInterface >($_self).getElementType()"
                      ": mlir::Type{}" >;

def Load
    : LowLevel_Op< "load", [PointerPointeeTypeMatch<"ptr", "result"> ] >
    , Arguments<(ins ElementTypeInterface:$ptr)>
    , Results<(outs AnyType:$result)>
{
    let summary = "Load value from memory. Expects `hl.ptr` as pointer type.";

    let assemblyFormat = [{
        operands attr-dict `:` functional-type(operands, results)
    }];
}

def Store
    : LowLevel_Op< "store", [PointerPointeeTypeMatch<"ptr", "val">]>
    , Arguments<(ins AnyType:$val, ElementTypeInterface:$ptr)>
{
    let summary = "Store value into memory. Expects `hl.ptr` as pointer type.";

    let assemblyFormat = [{
        $ptr `,` $val attr-dict `:` type($ptr) `,` type($val)
    }];
}

def UninitializedVar
    : LowLevel_Op< "uninitialized_var", [VastSymbol] >
    , Results<(outs AnyType:$result)>
{
    let summary = "Declaration of variable that have not been initialized yet.";
    let description = [{ Declaration of variable that have not been initialized yet. }];

    let results = (outs AnyType:$result);

    let assemblyFormat = [{
        attr-dict `:` type($result)
    }];
}

def InitializeVar
    : LowLevel_Op< "initialize" >
    , Arguments<(ins AnyType:$var, Variadic<AnyType>:$elements)>
    , Results<(outs AnyType:$result)>
{
    let summary = "Initialize a variable.";
    let description = [{
        Initialize a variable - for now this operation is a direct lowering from hl.var
        initialization section. Later there will be need to discover how this ties
        to constructors.
    }];

    let assemblyFormat = [{
        operands attr-dict `:` functional-type(operands, results)
    }];
}

def Concat
    : LowLevel_Op< "concat" >
    , Arguments<(ins Variadic<AnyType>:$args)>
    , Results<(outs AnyType:$result)>
{
    let summary = "Concat integers together";
    let description = [{
        Concat operands together, where first argument occupies lsb.
    }];

    let assemblyFormat = [{
        operands attr-dict `:` functional-type(operands, results)
    }];
}

def Extract
    : LowLevel_Op< "extract" >
    , Arguments<(ins AnyType:$arg, TypedAttrInterface:$from, TypedAttrInterface:$to)>
    , Results<(outs AnyType:$result)>
{
    let summary = "Extracts value";
    let description = [{
        `0` is lsb, `[inc, exc)`
    }];

    let assemblyFormat = [{
        operands attr-dict `:` functional-type(operands, results)
    }];

    let builders = [
        OpBuilder<(ins "mlir::Type":$type,
                       "mlir::Value":$value, "std::size_t":$from, "std::size_t":$to),
        [{
            auto attr_type = mlir::IntegerType::get($_builder.getContext(),
                                                    64, mlir::IntegerType::Unsigned);
            $_state.addOperands(value);
            $_state.addAttribute("from", mlir::IntegerAttr::get(attr_type, from));
            $_state.addAttribute("to", mlir::IntegerAttr::get(attr_type, to));
            $_state.addTypes(type);
        }] >
    ];

    // TODO(ll): Utility API like size().

    let extraClassDeclaration = [{
        std::size_t from()
        {
            return mlir::dyn_cast< mlir::IntegerAttr >(getFrom()).getUInt();
        }

        std::size_t to()
        {
            return mlir::dyn_cast< mlir::IntegerAttr >(getTo()).getUInt();
        }

        std::size_t size()
        {
            return to() - from() + 1;
        }
    }];
}

def Br
    : LowLevel_Op< "br",
        [Terminator, DeclareOpInterfaceMethods<BranchOpInterface>]
      >
{
    let summary = "Direct branch.";
    let description = [{ Direct branch }];

    let successors = (successor AnySuccessor:$dest);
    let arguments = (ins Variadic<AnyType>:$operands);

    let builders = [
        OpBuilder< (ins "mlir::ValueRange":$operands, "mlir::SuccessorRange":$dest),
        [{
            $_state.addOperands(operands);
            $_state.addSuccessors(dest);
        }] >,

        OpBuilder< (ins "mlir::Block *":$dest),
        [{
            $_state.addSuccessors(dest);
        }] >
    ];

    let assemblyFormat = [{
        $dest (`(` $operands^ `:` type($operands) `)`)? attr-dict
    }];
}

def CondBr
    : LowLevel_Op< "cond_br", [Terminator, AttrSizedOperandSegments] >
    , Arguments<(ins AnyType:$cond, Variadic<AnyType>:$trueOperands, Variadic<AnyType>:$falseOperands )>
{
    let summary = "Conditional branch.";
    let description = [{ Direct branch }];

    let successors = (successor AnySuccessor:$trueDest, AnySuccessor:$falseDest);

    let builders = [
        OpBuilder< (ins
            "mlir::Value":$cond,
            "mlir::ValueRange":$trueOperands, "mlir::Block *":$trueDest,
            "mlir::ValueRange":$falseOperands, "mlir::Block *":$falseDest),
        [{
            build($_builder, $_state,
                  cond,
                  trueOperands, falseOperands,
                  trueDest, falseDest);
        }] >,
        OpBuilder< (ins
            "mlir::Value":$cond,
            "mlir::Block *":$trueDest,
            "mlir::Block *":$falseDest),
        [{
            build($_builder, $_state,
                  cond,
                  mlir::ValueRange(), mlir::ValueRange(),
                  trueDest, falseDest);
        }] >
    ];

    let assemblyFormat = [{
        $cond `:` type($cond) `,`
        $trueDest (`(` $trueOperands^ `:` type($trueOperands) `)`)? `,`
        $falseDest (`(` $falseOperands^ `:` type($falseOperands) `)`)?
        attr-dict
    }];
}

def ScopeRet
    : LowLevel_Op< "scope_ret", [Terminator] >
{
    let summary = "Terminator of scope.";
    let description = [{ Terminator of scopes (for example during lowering of loops). }];

    let assemblyFormat = [{attr-dict}];
}

def ScopeRecurse
    : LowLevel_Op< "scope_recurse", [Terminator] >
{
    let summary = "Jump to first block of scope.";
    let description = [{ Modelling continue. }];

    let assemblyFormat = [{attr-dict}];
}

def CondScopeRet
    : LowLevel_Op< "cond_scope_ret", [Terminator] >
{
    let summary = "Terminator of scope if condition is met, otherwise branch.";
    let description = [{ Terminate or branch. }];

    let successors = (successor AnySuccessor:$dest);
    let arguments = (ins AnyType:$cond, Variadic<AnyType>:$dest_operands);

    let builders = [

        OpBuilder< (ins "mlir::Value":$cond, "mlir::Block *":$dest),
        [{
            build($_builder, $_state, cond, mlir::ValueRange(), dest);
        }] >
    ];

    let assemblyFormat = [{
        $cond `:` type($cond) `,`
        $dest (`(` $dest_operands^ `:` type($dest_operands) `)`)? attr-dict
    }];}

def ReturnOp
    : LowLevel_Op< "return", [Terminator, ReturnLikeTrait] >
    , Arguments<(ins Variadic<AnyType>:$result)>
{
    // Allow building a ReturnOp with no return operand.
    let builders = [
        OpBuilder<(ins), [{ build($_builder, $_state, std::nullopt); }]>
    ];

    let assemblyFormat = "($result^ `:` type($result))? attr-dict";
}

def Scope
    : LowLevel_Op< "scope", [NoRegionArguments] >
{
    let summary = "Scope, holds one region.";
    let description = [{ Scope that holds one region, each block should be terminated
                         with either branch, scope return or their conditional variants. }];

    let regions = (region AnyRegion:$body);

    let extraClassDeclaration = [{
        mlir::Block *start_block()
        {
            auto size = std::distance(getBody().begin(), getBody().end());
            if (size < 2)
                return nullptr;
            return &*std::next(getBody().begin());
        }
    }];

    let assemblyFormat = [{
        $body attr-dict
    }];
}

def InlineScope
    : LowLevel_Op< "inline_scope", [NoRegionArguments] >
{
    let summary = "Scope, that forwards (cond)scope return up.";
    let description = [{ Result of inlined if. }];

    let regions = (region AnyRegion:$body);
}

def LowLevel_FuncOp : Core_FuncBaseOp< LowLevel_Dialect, "func", [] >
{
  let skipDefaultBuilders = 1;

  let builders = [OpBuilder< (ins
    "llvm::StringRef":$name,
    "core::FunctionType":$type,
    CArg< "core::GlobalLinkageKind", "core::GlobalLinkageKind::ExternalLinkage" >:$linkage,
    CArg< "llvm::ArrayRef<mlir::NamedAttribute>", "{}" >:$attrs,
    CArg< "llvm::ArrayRef<mlir::DictionaryAttr>", "{}" >:$arg_attrs,
    CArg< "llvm::ArrayRef<mlir::DictionaryAttr>", "{}" >:$res_attrs,
    CArg< "BuilderCallback", "std::nullopt" >:$body), [{
      InsertionGuard guard($_builder);
      build_region($_builder, $_state, body);

      $_state.addAttribute(
        mlir::SymbolTable::getSymbolAttrName(), $_builder.getStringAttr(name)
      );
      $_state.addAttribute(getFunctionTypeAttrName($_state.name), mlir::TypeAttr::get(type));
      $_state.addAttribute(
        "linkage", core::GlobalLinkageKindAttr::get($_builder.getContext(), linkage)
      );

      $_state.attributes.append(attrs.begin(), attrs.end());

      if (arg_attrs.empty())
        return;

      mlir::function_interface_impl::addArgAndResultAttrs(
        $_builder, $_state, arg_attrs, res_attrs,
        getArgAttrsAttrName($_state.name), getResAttrsAttrName($_state.name)
      );
    }] >,

    OpBuilder< (ins
      "llvm::ArrayRef<mlir::NamedAttribute>":$attrs,
      CArg< "llvm::ArrayRef<mlir::DictionaryAttr>", "{}" >:$arg_attrs,
      CArg< "llvm::ArrayRef<mlir::DictionaryAttr>", "{}" >:$res_attrs,
      CArg< "BuilderCallback", "std::nullopt" >:$body), [{

        InsertionGuard guard($_builder);
        build_region($_builder, $_state, body);

        $_state.attributes.append(attrs.begin(), attrs.end());

        if (arg_attrs.empty())
          return;

        mlir::function_interface_impl::addArgAndResultAttrs(
          $_builder, $_state, arg_attrs, res_attrs,
          getArgAttrsAttrName($_state.name), getResAttrsAttrName($_state.name)
      );
    }] >
  ];

  let assemblyFormat = [{
    $linkage $sym_name custom< FunctionSignatureAndBody >($function_type, attr-dict, $body)
  }];

  let hasVerifier = 1;
}

#endif // VAST_DIALECT_IR_LOWLEVELOPS

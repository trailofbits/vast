// Copyright (c) 2023, Trail of Bits, Inc.

#ifndef VAST_DIALECT_CORE_CORETYPES
#define VAST_DIALECT_CORE_CORETYPES

class Core_Type< string name, string _mnemonic, list<Trait> traits = [] >
    : TypeDef< Core_Dialect, name, traits >
{
    let mnemonic = _mnemonic;
}

def Core_TypeArrayRefParameter : ArrayRefParameter<"Type"> {
  let printer = [{
    llvm::interleaveComma($_self, $_printer, [&$_printer](mlir::Type type) {
      $_printer.printType(type);
    });
    $_printer << ")";
  }];
  let parser = [{ [&] () -> mlir::FailureOr<llvm::SmallVector<Type>> {
    llvm::SmallVector<Type> params;
    if (mlir::succeeded($_parser.parseOptionalRParen())) {
        return mlir::success();
    }

    mlir::Type type;
    if ($_parser.parseType(type)) {
        return mlir::failure();
    }
    params.push_back(type);
    while (mlir::succeeded($_parser.parseOptionalComma())) {
        if ($_parser.parseType(type)) {
            return mlir::failure();
        }
        params.push_back(type);
    }

    if ($_parser.parseRParen()) {
        return mlir::failure();
    }
    return params;
  }() }];
}

def Core_FunctionType : Core_Type< "Function", "fn" > {
  let summary = "Vast function type";
  let description = [{
    The `!core.fn` is a function type. It consists of a variadic return type,
    and list of parameter types and can optionally be variadic.

    Example:

    ```mlir
    !core.fn<!hl.bool ()>
    !core.fn<!hl.int (!hl.char, !hl.char)>
    !core.fn<!i32 (!i32, ...)>
    ```
  }];

  let parameters = (ins
    ArrayRefParameter<"Type">:$inputs,
    Core_TypeArrayRefParameter:$results,
    "bool":$varArg
  );

  let assemblyFormat = [{
    `<` `(` custom< FunctionTypeArgs >($inputs, $varArg) `->` `(` $results `>`
  }];

  let builders = [
    TypeBuilderWithInferredContext<(ins
      "llvm::ArrayRef<Type>":$inputs,
      "llvm::ArrayRef<Type>":$results,
      CArg<"bool", "false">:$isVarArg),
    [{
        auto ctx = results.front().getContext();
        return $_get(ctx, inputs, results, isVarArg);
    }]>
  ];

  let extraClassDeclaration = [{
    /// Returns whether the function is variadic.
    bool isVarArg() const { return getVarArg(); }

    /// Returns the `i`th input operand type. Asserts if out of bounds.
    mlir::Type getInput(unsigned i) const { return getInputs()[i]; }

    /// Returns the number of arguments to the function.
    unsigned getNumInputs() const { return getInputs().size(); }

    /// Returns the `i`th output type. Asserts if out of bounds.
    mlir::Type getResult(unsigned i) const { return getResults()[i]; }

    /// Returns the number of arguments to the function.
    unsigned getNumResults() const { return getResults().size(); }

    /// Returns a clone of this function type with the given argument
    /// and result types.
    FunctionType clone(mlir::TypeRange inputs, mlir::TypeRange results) const;
  }];
}

#endif // VAST_DIALECT_CORE_CORETYPES

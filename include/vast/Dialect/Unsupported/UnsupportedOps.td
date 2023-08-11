// Copyright (c) 2023-present, Trail of Bits, Inc.

#ifndef VAST_DIALECT_IR_UNSUPPORTED_OPS
#define VAST_DIALECT_IR_UNSUPPORTED_OPS

def UnsupportedDecl
    : Unsupported_Op< "decl", [NoTerminator] >
    , Arguments<(ins StrAttr:$name)>
{
    let summary = "VAST unsupported decl";
    let description = [{ VAST unsupported decl }];

    let regions = (region AnyRegion:$body);

    let skipDefaultBuilders = 1;
    let builders = [
        OpBuilder<(ins
            "llvm::StringRef":$name,
             CArg<"BuilderCallback", "std::nullopt">:$body
        )>
    ];

    let assemblyFormat = [{ $name attr-dict `:` $body }];
}

def UnsupportedExpr
    : Unsupported_Op< "expr" >
    , Arguments<(ins StrAttr:$name)>
    , Results<(outs Optional< AnyType >:$result)>
{
    let summary = "VAST unsupported expression";
    let description = [{ VAST unsupported expression }];

    let regions = (region AnyRegion:$substmt);

    let skipDefaultBuilders = 1;
    let builders = [ OpBuilder<(ins "::mlir::StringRef":$name, "::mlir::Type":$rty, "std::unique_ptr<::mlir::Region> &&":$region)> ];

    let assemblyFormat = [{
        attr-dict `:` type($result) $substmt
    }];
}

def UnsupportedOp
    : Unsupported_Op<"op">
    , Arguments<(ins StrAttr:$name, Variadic<AnyType>:$elements)>
    , Results<(outs Optional< AnyType >:$result)>
{
    let summary = "VAST unsupported operation";
    let description = [{ VAST unsupported operation }];
}

#endif // VAST_DIALECT_IR_UNSUPPORTED_OPS

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
             CArg< "maybe_builder_callback_ref", "std::nullopt" >:$body
        )>
    ];

    let assemblyFormat = [{ $name attr-dict `:` $body }];
}

def UnsupportedStmt
    : Unsupported_Op< "stmt", [NoTerminator] >
    , Arguments<(ins StrAttr:$name)>
    , Results<(outs Optional< AnyType >:$result)>
{
    let summary = "VAST unsupported statement";
    let description = [{ VAST unsupported statement }];

    let regions = (region VariadicRegion< AnyRegion >:$children);

    let skipDefaultBuilders = 1;
    let builders = [
        OpBuilder<(ins
            "llvm::StringRef":$name,
            "Type":$rty,
            "const std::vector< builder_callback > &":$builders
        )>
    ];

    let assemblyFormat = [{ $name attr-dict `:` type($result) $children }];
}

#endif // VAST_DIALECT_IR_UNSUPPORTED_OPS

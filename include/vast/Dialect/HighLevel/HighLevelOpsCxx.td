// Copyright (c) 2023-present, Trail of Bits, Inc.

#ifndef VAST_DIALECT_HIGHLEVEL_IR_HIGHLEVELOPS_CXX
#define VAST_DIALECT_HIGHLEVEL_IR_HIGHLEVELOPS_CXX

class CxxRecordLikeDeclOp< string mnemonic, list< Trait > traits = [] >
    : HighLevel_Op< mnemonic, !listconcat(traits, [NoTerminator, VastSymbol]) >
    , Arguments<(ins StrAttr:$name)>
{
  // TODO(hl:cxx): Add region constraints.
  let regions = (region
      AnyRegion:$bases,
      AnyRegion:$fields);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins
      "llvm::StringRef":$name,
       CArg<"BuilderCallback", "std::nullopt">:$bases,
       CArg<"BuilderCallback", "std::nullopt">:$fields
    )>
  ];

  let assemblyFormat = [{ $name attr-dict `:` `bases` $bases $fields }];
}

def ClassDeclOp : CxxRecordLikeDeclOp< "class" > {
  let summary = "VAST C++ class declaration";
  let description = [{ VAST C++ class declaration }];
}

def CxxStructDeclOp : CxxRecordLikeDeclOp< "cxxstruct" > {
  let summary = "VAST C++ struct declaration";
  let description = [{ VAST C++ struct declaration }];
}

def PublicAccess :
  I32EnumAttrCase<"as_public", 0, "public">;
def ProtectedAccess :
  I32EnumAttrCase<"as_protected", 1, "protected">;
def PrivateAccess :
  I32EnumAttrCase<"as_private", 2, "private">;
def NoneAccess :
  I32EnumAttrCase<"as_none", 3, "none">;

def AccessSpecifier : I32EnumAttr<
    "AccessSpecifier", "Access specifier",
    [
        PublicAccess, ProtectedAccess,
        PrivateAccess, NoneAccess
    ]>
{
  let cppNamespace = "::vast::hl";
}

def AccessSpecifierOp
    : HighLevel_Op< "access", [] >
    , Arguments<(ins AccessSpecifier:$spec)>
{
  let summary = "VAST C++ access specifier declaration";
  let description = [{ VAST C++ access specifier declaration }];

  let assemblyFormat = [{ attr-dict $spec }];
}

def CxxBaseSpecifierOp
  : HighLevel_Op< "base", [VastSymbol] >
  , Arguments<(ins TypeAttr:$type, AccessSpecifier:$access, UnitAttr:$is_virtual)>
{
  let summary = "VAST base class specifier";
  let description = [{ VAST base class specifier }];

  let assemblyFormat = [{ $type attr-dict $access (`virtual` $is_virtual^)? }];
}

def ThisOp
  : HighLevel_Op< "this", [] >
  , Results<(outs AnyType:$result)>
{
  let summary = "VAST this operator";
  let description = [{ VAST this operator }];

  let assemblyFormat = [{
     attr-dict `:` type($result)
  }];
}

def NoRef :
  I32EnumAttrCase<"ref_none", 0, "ref_none">;
def LValueRef :
  I32EnumAttrCase<"ref_lvalue", 1, "ref_lvalue">;
def RValueRef :
  I32EnumAttrCase<"ref_rvalue", 2, "ref_rvalue">;

def RefQualifier : I32EnumAttr<
    "RefQualifier", "ref qualifier",
    [
        NoRef, LValueRef, RValueRef
    ]>
{
  let cppNamespace = "::vast::hl";
}

def MethodOp
    : FuncLikeOp< "method", (ins
        UnitAttr:$is_virtual,
        UnitAttr:$is_const,
        UnitAttr:$is_volatile,
        RefQualifier:$ref), (ins
        CArg< "bool", "false" >:$is_virtual,
        CArg< "bool", "false" >:$is_const,
        CArg< "bool", "false" >:$is_volatile,
        CArg< "RefQualifier", "RefQualifier::ref_none" >:$ref), [{
          if (is_virtual) {
            $_state.addAttribute(
              "is_virtual", mlir::UnitAttr::get($_builder.getContext())
            );
          }
          if (is_const) {
            $_state.addAttribute(
              "is_const", mlir::UnitAttr::get($_builder.getContext())
            );
          }
          if (is_volatile) {
            $_state.addAttribute(
              "is_volatile", mlir::UnitAttr::get($_builder.getContext())
            );
          }
          $_state.addAttribute(
            "ref", RefQualifierAttr::get($_builder.getContext(), ref)
          );
        }] >
{
  let summary = "VAST high-level method definintion or declaration";

  let assemblyFormat = [{
    $linkage (`virtual` $is_virtual^)? $ref (`const` $is_const^)? (`volatile` $is_volatile^)? $sym_name custom< FunctionSignatureAndBody >($function_type, attr-dict, $body)
  }];
}

def DtorOp
    : FuncLikeOp< "dtor",
        (ins UnitAttr:$is_virtual),
        (ins CArg< "bool", "false" >:$is_virtual),
        [{
          if (is_virtual) {
            $_state.addAttribute(
              "is_virtual", mlir::UnitAttr::get($_builder.getContext())
            );
          }
        }] >
{
  let summary = "VAST high-level destructor definintion or declaration";

  let assemblyFormat = [{
    $linkage (`virtual` $is_virtual^)? $sym_name custom< FunctionSignatureAndBody >($function_type, attr-dict, $body)
  }];
}

#endif // VAST_DIALECT_HIGHLEVEL_IR_HIGHLEVELOPS_CXX
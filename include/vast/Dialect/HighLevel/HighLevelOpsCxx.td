// Copyright (c) 2023-present, Trail of Bits, Inc.

#ifndef VAST_DIALECT_HIGHLEVEL_IR_HIGHLEVELOPS_CXX
#define VAST_DIALECT_HIGHLEVEL_IR_HIGHLEVELOPS_CXX

class HighLevel_CxxRecordLikeDeclOp< string mnemonic, list< Trait > traits = [] >
    : HighLevel_Op< mnemonic, !listconcat(traits, [NoTerminator, Core_TypeSymbol]) >
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
       CArg< "maybe_builder_callback_ref", "std::nullopt" >:$bases,
       CArg< "maybe_builder_callback_ref", "std::nullopt" >:$fields
    )>
  ];

  let assemblyFormat = [{ $name attr-dict `:` `bases` $bases $fields }];
}

def HighLevel_ClassDeclOp
  : HighLevel_CxxRecordLikeDeclOp< "class" >
{
  let summary = "VAST C++ class declaration";
  let description = [{ VAST C++ class declaration }];
}

def HighLevel_CxxStructDeclOp
  : HighLevel_CxxRecordLikeDeclOp< "cxxstruct" >
{
  let summary = "VAST C++ struct declaration";
  let description = [{ VAST C++ struct declaration }];
}

def HighLevel_PublicAccess :
  I32EnumAttrCase<"as_public", 0, "public">;
def HighLevel_ProtectedAccess :
  I32EnumAttrCase<"as_protected", 1, "protected">;
def HighLevel_PrivateAccess :
  I32EnumAttrCase<"as_private", 2, "private">;
def HighLevel_NoneAccess :
  I32EnumAttrCase<"as_none", 3, "none">;

def HighLevel_AccessSpecifier : I32EnumAttr<
  "AccessSpecifier", "Access specifier", [
      HighLevel_PublicAccess,
      HighLevel_ProtectedAccess,
      HighLevel_PrivateAccess,
      HighLevel_NoneAccess
  ]>
{
  let cppNamespace = "::vast::hl";
}

def HighLevel_AccessSpecifierOp
    : HighLevel_Op< "access", [] >
    , Arguments<(ins HighLevel_AccessSpecifier:$spec)>
{
  let summary = "VAST C++ access specifier declaration";
  let description = [{ VAST C++ access specifier declaration }];

  let assemblyFormat = [{ attr-dict $spec }];
}

def HighLevel_CxxBaseSpecifierOp
  : HighLevel_Op< "base" >
  , Arguments<(ins
    TypeAttr:$type,
    HighLevel_AccessSpecifier:$access,
    UnitAttr:$is_virtual
  ) >
{
  let summary = "VAST base class specifier";
  let description = [{ VAST base class specifier }];

  let assemblyFormat = [{ $type attr-dict $access (`virtual` $is_virtual^)? }];
}

def HighLevel_ThisOp
  : HighLevel_Op< "this", [] >
  , Results<(outs AnyType:$result)>
{
  let summary = "VAST this operator";
  let description = [{ VAST this operator }];

  let assemblyFormat = [{
     attr-dict `:` type($result)
  }];
}

#endif // VAST_DIALECT_HIGHLEVEL_IR_HIGHLEVELOPS_CXX
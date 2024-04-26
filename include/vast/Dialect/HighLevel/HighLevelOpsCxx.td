// Copyright (c) 2023-present, Trail of Bits, Inc.

#ifndef VAST_DIALECT_HIGHLEVEL_IR_HIGHLEVELOPS_CXX
#define VAST_DIALECT_HIGHLEVEL_IR_HIGHLEVELOPS_CXX

class CxxRecordLikeDeclOp< string mnemonic, list< Trait > traits = [] >
    : HighLevel_Op< mnemonic, !listconcat(traits, [NoTerminator, TypeSymbol]) >
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
  : HighLevel_Op< "base" >
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

#endif // VAST_DIALECT_HIGHLEVEL_IR_HIGHLEVELOPS_CXX
// Copyright (c) 2021-present, Trail of Bits, Inc.

#ifndef VAST_DIALECT_HIGHLEVEL_IR_HIGHLEVELTYPES
#define VAST_DIALECT_HIGHLEVEL_IR_HIGHLEVELTYPES

include "mlir/Interfaces/DataLayoutInterfaces.td"
include "mlir/IR/SubElementInterfaces.td"
include "mlir/IR/BuiltinTypeInterfaces.td"

// NOTE(lukas): I have no idea how to do this both in a sane way and in tablegen
class DefaultDLTypeInterface {
  code default_dl_interface_impl = [{
    using dl_t = mlir::DataLayout;
    using dl_entries_ref = mlir::DataLayoutEntryListRef;

    unsigned getTypeSizeInBits(const mlir::DataLayout &dl,
                               mlir::DataLayoutEntryListRef entries) const
    {
        using self_t = std::remove_cvref_t< decltype(*this) >;
        return DefaultDL< self_t >::getTypeSizeInBits(dl, entries);
    }
    unsigned getABIAlignment(const mlir::DataLayout &dl,
                             mlir::DataLayoutEntryListRef entries) const
    {
        using self_t = std::remove_cvref_t< decltype(*this) >;
        return DefaultDL< self_t >::getABIAlignment(dl, entries);
    }
    unsigned getPreferredAlignment(const mlir::DataLayout &dl,
                                   mlir::DataLayoutEntryListRef entries) const
    {
        using self_t = std::remove_cvref_t< decltype(*this) >;
        return DefaultDL< self_t >::getPreferredAlignment(dl, entries);
    }
  }];
}

def WithDL : DefaultDLTypeInterface;

class HighLevelType< string name, list<Trait> traits = []>
  : TypeDef< HighLevel_Dialect, name ,
             !listconcat(traits, [DataLayoutTypeInterface]) >
{
    let extraClassDeclaration = WithDL.default_dl_interface_impl;
}

def VoidType : HighLevelType< "Void" > {
  let mnemonic = "void";
}

class CVQualified< string name, string mnem, list<Trait> traits = []>
  : HighLevelType< name, traits >
{
  let mnemonic = mnem;

  let parameters = (ins
    "bool":$isConst,
    "bool":$isVolatile
  );

  let extraClassDeclaration = WithDL.default_dl_interface_impl #
  [{
    bool isConst() const { return this->getIsConst(); }
    bool isVolatile() const { return this->getIsVolatile(); }
  }];

  let builders = [
    TypeBuilder<(ins
      CArg<"bool", "false">:$isConst,
      CArg<"bool", "false">:$isVolatile
    ), [{ return $_get($_ctxt, isConst, isVolatile); }]>
  ];

  let skipDefaultBuilders = 1;

  let parser  = [{
    return parse_cv_type<}] # cppClassName # [{>($_ctxt, parser);
  }];

  let printer = [{ return print_cv_type(*this, printer); }];
}

def BoolType : CVQualified< "Bool", "bool" >;

def BoolLikeType : TypeConstraint<
  Or< [BoolType.predicate, AnyI1.predicate] >,
  "bool like type"
>;

class IntegerType< string name, string mnem, list<Trait> traits = []>
  : HighLevelType< name, traits >
{
  let mnemonic = mnem;

  let parameters = (ins
    "bool":$isUnsigned,
    "bool":$isConst,
    "bool":$isVolatile
  );

  let extraClassDeclaration = WithDL.default_dl_interface_impl #
  [{
    bool isUnsigned() const { return this->getIsUnsigned(); }
    bool isSigned() const { return !this->isUnsigned(); }
    bool isConst() const { return this->getIsConst(); }
    bool isVolatile() const { return this->getIsVolatile(); }
  }];

  let builders = [
    TypeBuilder<(ins
      CArg<"bool", "false">:$isUnsigned,
      CArg<"bool", "false">:$isConst,
      CArg<"bool", "false">:$isVolatile
    ), [{ return $_get($_ctxt, isUnsigned, isConst, isVolatile); }]>
  ];

  let skipDefaultBuilders = 1;

  let parser  = [{
    return parse_integer_type<}] # cppClassName # [{>($_ctxt, parser);
  }];

  let printer = [{ return print_integer_type(*this, printer); }];
}

def CharType     : IntegerType< "Char", "char" >;
def ShortType    : IntegerType< "Short", "short" >;
def IntType      : IntegerType< "Int", "int" >;
def LongType     : IntegerType< "Long", "long" >;
def LongLongType : IntegerType< "LongLong", "longlong" >;
def Int128Type   : IntegerType< "Int128", "int128" >;

def HLIntegerType : AnyTypeOf<[
  CharType, ShortType, IntType, LongType, LongLongType, Int128Type,
]>;

def IntegerLikeType : TypeConstraint<
  Or< [HLIntegerType.predicate, AnyInteger.predicate] >,
  "integer like type"
>;

def IntOrBoolType : TypeConstraint<
  Or< [IntegerLikeType.predicate, BoolLikeType.predicate] >,
  "bool or integer like type"
>;

class FloatingType< string name, string mnem, list<Trait> traits = []>
  : CVQualified< name, mnem, traits >;

def HalfType       : FloatingType< "Half", "half" >;
def BFloat16Type   : FloatingType< "BFloat16", "bfloat16" >;
def FloatType      : FloatingType< "Float", "float" >;
def DoubleType     : FloatingType< "Double", "double" >;
def LongDoubleType : FloatingType< "LongDouble", "longdouble" >;
def Float128Type   : FloatingType< "Float128", "float128" >;

def HLFloatType : AnyTypeOf<[
  HalfType, BFloat16Type, FloatType, DoubleType, LongDoubleType, Float128Type
]>;

def FloatLikeType : TypeConstraint<
  Or< [HLFloatType.predicate, AnyFloat.predicate] >,
  "float like type"
>;

def PointerType : HighLevelType< "Pointer",
                                 [DeclareTypeInterfaceMethods< SubElementTypeInterface >,
                                  MemRefElementTypeInterface]>
{
  let mnemonic = "ptr";

  let parameters = (ins
    "Type":$elementType,
    "bool":$isConst,
    "bool":$isVolatile
  );

  let extraClassDeclaration = WithDL.default_dl_interface_impl #
  [{
    bool isConst() const { return this->getIsConst(); }
    bool isVolatile() const { return this->getIsVolatile(); }
  }];

  let builders = [
    TypeBuilder<(ins
      "Type":$elementType,
      CArg<"bool", "false">:$isConst,
      CArg<"bool", "false">:$isVolatile
    ), [{ return $_get($_ctxt, elementType, isConst, isVolatile); }]>
  ];

  let skipDefaultBuilders = 1;

  let parser  = [{ return parse_pointer_type($_ctxt, parser); }];
  let printer = [{ return print_pointer_type(*this, printer); }];
}

def ConstantArrayType : HighLevelType< "ConstantArray",
                        [DeclareTypeInterfaceMethods< SubElementTypeInterface >,
                         MemRefElementTypeInterface]> {
  let mnemonic = "const.array";

  let parameters = (ins
    "Type":$elementType,
    "llvm::APInt":$size,
    "bool":$isConst,
    "bool":$isVolatile
  );

  let extraClassDeclaration = WithDL.default_dl_interface_impl #
  [{
    using dimensions_t = std::vector< uint32_t >;
    std::tuple< dimensions_t, mlir::Type > dim_and_type();

    dimensions_t dimensions() { return std::get< 0 >(dim_and_type()); }

    bool isConst() const { return this->getIsConst(); }
    bool isVolatile() const { return this->getIsVolatile(); }
    uint32_t getNumElems() const { return this->getSize().getLimitedValue(); }
  }];
}

def NamedType : HighLevelType< "Named" > {
  let mnemonic = "named_type";

  let parameters = (ins
    "mlir::SymbolRefAttr":$name
  );
}

def SubscriptableType : TypeConstraint<
  Or< [ConstantArrayType.predicate, PointerType.predicate] >,
  "array like type"
>;


def PointerLikeType : TypeConstraint<
  PointerType.predicate,
  "pointer like type"
>;

#endif // VAST_DIALECT_HIGHLEVEL_IR_HIGHLEVELTYPES

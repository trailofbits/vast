// Copyright (c) 2021-present, Trail of Bits, Inc.

#ifndef VAST_DIALECT_HIGHLEVEL_IR_HIGHLEVELTYPES
#define VAST_DIALECT_HIGHLEVEL_IR_HIGHLEVELTYPES

include "mlir/Interfaces/DataLayoutInterfaces.td"
include "mlir/IR/SubElementInterfaces.td"
include "mlir/IR/BuiltinTypeInterfaces.td"

include "vast/Dialect/HighLevel/HighLevelAttributes.td"
include "vast/Interfaces/TypeQualifiersInterfaces.td"

// NOTE(lukas): I have no idea how to do this both in a sane way and in tablegen
class WithDefaultDLTypeInterface {
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

//
// Generic high-level type
//
class HighLevelType< string name, list<Trait> traits = []>
  : TypeDef< HighLevel_Dialect, name,
    !listconcat(traits, [DataLayoutTypeInterface])
  >
  , WithDefaultDLTypeInterface
{
  let extraClassDeclaration = default_dl_interface_impl;
}

//
// LValue high-level type and its utilities
//
def LValue : HighLevelType< "LValue",
  [DeclareTypeInterfaceMethods< SubElementTypeInterface >]
> {
  let mnemonic = "lvalue";

  let parameters = (ins
    "Type":$elementType
  );

  let assemblyFormat = "`<` $elementType `>`";
}

class LValueOf<TypeConstraint value> : Type< And< [
    LValue.predicate,
    SubstLeaves< "$_self", "$_self.cast< LValueType >().getElementType()", value.predicate >
  ] >,
  "lvalue to " # value.summary
>;

class LValueOrType<TypeConstraint value> : Type<
  Or< [
    And< [
      LValue.predicate,
      SubstLeaves< "$_self", "$_self.cast< LValueType >().getElementType()", value.predicate >
    ] >,
    value.predicate
  ] >,
  "lvalue to " # value.summary
>;

class TypesOrLValueTypesMatch<string summary, string lhs, string rhs> : PredOpTrait< summary,
  Or<[
    TypesMatchWith< "underlying lvalue type does not match",
      lhs, rhs, "LValueType::get( $_ctxt, $_self )"
    >.predicate,
    AllTypesMatch< ["src", "dst"] >.predicate
  ]>
>;

//
// Type qualifiers
//
class Qualifier< string name, TypeInterface type_interface, dag qualifier > {
  TypeInterface interface = type_interface;

  dag param = qualifier;

  code qualifier_methods_definitions = [{
    bool $cppClass::has_}] # name # [{() const { return getImpl()->is_}] # name # [{; }
  }];
}

class QualfierList< list< Qualifier > quals > {
  // qualifiers type interfaces
  list<TypeInterface> interfaces = !foreach(qual, quals, DeclareTypeInterfaceMethods<qual.interface>);

  // qualifiers type parameters
  list<dag> param_list = !foreach(qual, quals, qual.param);
  dag params = !foldl((ins), param_list, result, param, !con(result, param));

  // qualifiers accessor methods
  list<code> methods_list = !foreach(qual, quals, qual.qualifier_methods_definitions);
  code qualifier_methods_definitions = !foldl("",  methods_list, result, methods, !strconcat(result, methods));
}

class QualifiedType< string name, string mnem, dag params = (ins), QualfierList quals, list<Trait> traits = [] >
  : HighLevelType< name, !listconcat(traits, quals.interfaces) >
{
  let mnemonic = mnem;

  let parameters = !con(params, quals.params);

  // TODO remove
  let extraClassDeclaration = default_dl_interface_impl;

  let extraClassDefinition = quals.qualifier_methods_definitions;

  let assemblyFormat = "(`<` params^ `>`)?";
}

//
// optional type qualifier parameter
//
class QualifierParam< string name, string desc > : TypeParameter<"bool", desc> {
  let printer = [{ $_printer << ($_self ? "}] # name # [{" : ""); }];
  let parser = [{ [&] () -> mlir::FailureOr<bool> {
    return !$_parser.parseKeyword( "}] # name # [{" ); } ()
  }];
  let isOptional = 1;
}

//
// high-level language qualifiers
//
def ConstParam    : QualifierParam<    "const", "const qualifier" >;
def VolatileParam : QualifierParam< "volatile", "volatile qualifier" >;
def RestrictParam : QualifierParam< "restrict", "restrict qualifier" >;

def UnsignedParam : QualifierParam< "unsigned", "unsigned qualifier" >;

def ConstQualifier    : Qualifier< "const"   , ConstQualifierInterface,    (ins ConstParam:$is_const)> {}
def VolatileQualifier : Qualifier< "volatile", VolatileQualifierInterface, (ins VolatileParam:$is_volatile)> {}
def RestrictQualifier : Qualifier< "restrict", RestrictQualifierInterface, (ins RestrictParam:$is_restrict)> {}

class CVQualifiedType< string name, string mnem, dag params = (ins), list<Trait> traits = [] >
  : QualifiedType< name, mnem, params
  , QualfierList< [ConstQualifier, VolatileQualifier] >
  , traits >
{}

class CVRQualifiedType< string name, string mnem, dag params = (ins), list<Trait> traits = [] >
  : QualifiedType< name, mnem, params
  , QualfierList< [ConstQualifier, VolatileQualifier, RestrictQualifier] >
  , traits >
{}

//
// Basic types
//
def VoidType : CVQualifiedType< "Void", "void" > {}

def BoolType : CVQualifiedType< "Bool", "bool" > {}

def BoolLikeType : TypeConstraint<
  Or< [BoolType.predicate, AnyI1.predicate] >,
  "bool like type"
>;

//
// Integer types & traits
//
class IntegerType< string name, string mnemonic, list<Trait> traits = []>
  : CVQualifiedType< name, mnemonic, (ins UnsignedParam:$is_unsigned), traits >
{}

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

//
// Floating point types & traits
//
class FloatingType< string name, string mnem, list<Trait> traits = []>
  : CVQualifiedType< name, mnem, (ins), traits >;

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

//
// Memory related types & traits
//
def PointerType : CVRQualifiedType<
  "Pointer", "ptr",
  (ins "Type":$elementType),
  [ DeclareTypeInterfaceMethods< SubElementTypeInterface >, MemRefElementTypeInterface]
>
{
  // TODO can we somehow use optional `params` group and not list all qualifiers?
  let assemblyFormat = "`<` $elementType (`,` $is_const^ )? (`,` $is_volatile^ )? (`,` $is_restrict^ )? `>`";
}

def PointerLikeType : TypeConstraint< PointerType.predicate, "pointer like type" >;


//
// Array types
//
class WithDimensionsClass
{
  code default_dimensions_impl = [{
    using dimensions_t = std::vector< SizeParam >;

    std::tuple< dimensions_t, mlir::Type > dim_and_type();

    dimensions_t dimensions() { return std::get< 0 >(dim_and_type()); }
  }];
}

def SizeParam : TypeParameter<"SizeParam", "size parameter for arrays"> {}

def ArrayType : CVRQualifiedType<
    "Array", "array",
    (ins "Type":$elementType),
    [ DeclareTypeInterfaceMethods< SubElementTypeInterface >, MemRefElementTypeInterface]
  >
  , WithDimensionsClass
{
  let extraClassDeclaration = default_dl_interface_impl # default_dimensions_impl;

  // TODO can we somehow use optional `params` group and not list all qualifiers?
  let assemblyFormat = "`<` $elementType (`,` $is_const^ )? (`,` $is_volatile^ )? (`,` $is_restrict^ )? `>`";
}

def SubscriptableType : TypeConstraint<
  Or< [ArrayType.predicate, PointerType.predicate] >, "array like type"
>;

//
// Tagged types
//
def RecordType : CVQualifiedType< "Record", "record", (ins StringRefParameter<>:$name), [MemRefElementTypeInterface] > {
  // TODO can we somehow use optional `params` group and not list all qualifiers?
  let assemblyFormat = "`<` $name (`,` $is_const^ )? (`,` $is_volatile^ )? `>`";
}

def EnumType : CVQualifiedType< "Enum", "enum", (ins StringRefParameter<>:$name), [MemRefElementTypeInterface] > {
  // TODO can we somehow use optional `params` group and not list all qualifiers?
  let assemblyFormat = "`<` $name (`,` $is_const^ )? (`,` $is_volatile^ )? `>`";
}

def TypedefType : CVRQualifiedType< "Typedef", "typedef", (ins StringRefParameter<>:$name), [MemRefElementTypeInterface] > {
  // TODO can we somehow use optional `params` group and not list all qualifiers?
  let assemblyFormat = "`<` $name (`,` $is_const^ )? (`,` $is_volatile^ )? (`,` $is_restrict^ )? `>`";
}

def ElaboratedType : CVRQualifiedType<
    "Elaborated", "elaborated",
    (ins "Type":$elementType),
    [ DeclareTypeInterfaceMethods< SubElementTypeInterface >]
  >
{
  // TODO can we somehow use optional `params` group and not list all qualifiers?
  let assemblyFormat = "`<` $elementType (`,` $is_const^ )? (`,` $is_volatile^ )? (`,` $is_restrict^ )? `>`";
}

def LabelType : HighLevelType< "Label" > {
  let mnemonic = "label";
}

def ParenType : HighLevelType< "Paren", [MemRefElementTypeInterface] > {
  let mnemonic = "paren";
  let parameters = (ins "Type":$elementType);
  let assemblyFormat = "`<` $elementType `>`";
}

#endif // VAST_DIALECT_HIGHLEVEL_IR_HIGHLEVELTYPES

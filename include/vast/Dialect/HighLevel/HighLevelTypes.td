// Copyright (c) 2021-present, Trail of Bits, Inc.

#ifndef VAST_DIALECT_HIGHLEVEL_IR_HIGHLEVELTYPES
#define VAST_DIALECT_HIGHLEVEL_IR_HIGHLEVELTYPES

include "mlir/Interfaces/DataLayoutInterfaces.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
include "mlir/IR/AttrTypeBase.td"

include "vast/Dialect/HighLevel/HighLevelAttributes.td"
include "vast/Interfaces/TypeQualifiersInterfaces.td"
include "vast/Interfaces/AliasTypeInterface.td"
include "vast/Interfaces/DefaultDataLayoutTypeInterface.td"
include "vast/Interfaces/ElementTypeInterface.td"

include "vast/Dialect/Core/Utils.td"

//
// Generic high-level type
//
class HighLevelType< string name, list<Trait> traits = []>
  : TypeDef< HighLevel_Dialect, name,
    !listconcat(traits, [
      DataLayoutTypeInterface,
      DefaultDataLayoutTypeInterface
    ])
  >
{}

//
// Type qualifiers
//

class QualifierParam< string name, string desc > : OptionalParameter<"bool", desc> {
  let printer = [{ $_printer << ($_self ? "}] # name # [{" : ""); }];
  let parser = [{ [&] () -> mlir::FailureOr<bool> {
    return succeeded($_parser.parseOptionalKeyword( "}] # name # [{" )); } ()
  }];
}

//
// high-level language qualifiers
//
def ConstParam    : QualifierParam<    "const", "const qualifier" >;
def VolatileParam : QualifierParam< "volatile", "volatile qualifier" >;
def RestrictParam : QualifierParam< "restrict", "restrict qualifier" >;

def UnsignedParam : QualifierParam< "unsigned", "unsigned qualifier" >;

class Qualifier< dag qualifier, list<Trait> traits > {
  dag param = qualifier;

  list<Trait> interface = traits;
}

def ConstQualifier    : Qualifier< (ins ConstParam:$is_const), [ConstQualifierInterface] > {}
def VolatileQualifier : Qualifier< (ins VolatileParam:$is_volatile), [VolatileQualifierInterface] > {}
def RestrictQualifier : Qualifier< (ins RestrictParam:$is_restrict), [RestrictQualifierInterface] > {}

def UnsignedQualifier : Qualifier< (ins UnsignedParam:$is_unsigned), [UnsignedQualifierInterface] > {}

class FlattenList< list<list<Trait>> traits > {
  list<Trait> result = !foldl([]<Trait>, traits, res, trait, !listconcat(res, trait));
}

class FlattenInsList< list<dag> dags > {
  dag result = !foldl((ins), dags, res, elem, !con(res, elem));
}

class QualfiersParams< string name, list<Qualifier> qualifiers>
  : HighLevel_Attr< name, "quals", FlattenList< !foreach(qual, qualifiers, qual.interface) >.result >
{
  let parameters = FlattenInsList< !foreach(qual, qualifiers, qual.param) >.result;

  // See commit `3dbac2c007c11` in the upstream.
  string attrName = name;

  let assemblyFormat = "params ` `";
}

def CVQualifiers  : QualfiersParams< "CVQualifiers",  [ConstQualifier, VolatileQualifier] > {}
def UCVQualifiers : QualfiersParams< "UCVQualifiers", [UnsignedQualifier, ConstQualifier, VolatileQualifier] > {}
def CVRQualifiers : QualfiersParams< "CVRQualifiers", [ConstQualifier, VolatileQualifier, RestrictQualifier] > {}

class QualifiedType< string name, string mnem, dag params = (ins), list<Trait> traits = [] >
  : HighLevelType< name, traits >
{
  let mnemonic = mnem;

  let parameters = params;

  let assemblyFormat = "(`<` $quals^ `>`)?";
}

class CVQualifiedType< string name, string mnem, dag params = (ins), list<Trait> traits = [] >
  : QualifiedType< name, mnem, !con(params, (ins OptionalParameter< "CVQualifiersAttr" >:$quals)), traits >
{}

class UCVQualifiedType< string name, string mnem, dag params = (ins), list<Trait> traits = [] >
  : QualifiedType< name, mnem, !con(params, (ins OptionalParameter< "UCVQualifiersAttr" >:$quals)), traits >
{}

class CVRQualifiedType< string name, string mnem, dag params = (ins), list<Trait> traits = [] >
  : QualifiedType< name, mnem, !con(params, (ins OptionalParameter< "CVRQualifiersAttr" >:$quals)), traits >
{}

//
// Tagged types and utils
//
def RecordType : CVQualifiedType<
  "Record", "record", (ins StringRefParameter<>:$name), [MemRefElementTypeInterface]
> {

  let builders = [
    TypeBuilder<(ins "llvm::StringRef":$name), [{
      return $_get($_ctxt, name, CVQualifiersAttr());
    }]>
  ];

  let assemblyFormat = "`<` $name (`,` $quals^ )? `>`";
}

def EnumType : CVQualifiedType<
  "Enum", "enum", (ins StringRefParameter<>:$name), [MemRefElementTypeInterface]
> {
  let builders = [
    TypeBuilder<(ins "llvm::StringRef":$name), [{
      return $_get($_ctxt, name, CVQualifiersAttr());
    }]>
  ];

  let assemblyFormat = "`<` $name (`,` $quals^ )? `>`";
}

def TypedefType : CVRQualifiedType<
  "Typedef", "typedef", (ins StringRefParameter<>:$name), [MemRefElementTypeInterface, TypedefTrait]
> {
  let builders = [
    TypeBuilder<(ins "llvm::StringRef":$name), [{
      return $_get($_ctxt, name, CVRQualifiersAttr());
    }]>
  ];

  let assemblyFormat = "`<` $name (`,` $quals^ )? `>`";
}

def ElaboratedType : CVRQualifiedType<
  "Elaborated", "elaborated", (ins "Type":$elementType), [ElementTypeInterface, MemRefElementTypeInterface]
> {
  let builders = [
    TypeBuilder<(ins "Type":$element), [{
      return $_get($_ctxt, element, CVRQualifiersAttr());
    }]>
  ];

  let assemblyFormat = "`<` $elementType (`,` $quals^ )? `>`";
}

def LabelType : HighLevelType< "Label" > {
  let mnemonic = "label";
}

def ParenType : HighLevelType< "Paren", [ElementTypeInterface, MemRefElementTypeInterface] > {
  let mnemonic = "paren";
  let parameters = (ins "Type":$elementType);
  let assemblyFormat = "`<` $elementType `>`";
}

class ElaboratedType< TypeConstraint value > : Type<
  And< [
    ElaboratedType.predicate,
    SubstLeaves< "$_self", "$_self.cast< ElaboratedType >().getElementType()", value.predicate >
  ] >
>;

class TypeOrElaboratedType< TypeConstraint value > : Type<
  Or< [
    value.predicate,
    ElaboratedType< value >.predicate
  ]>
>;

class TypeWithSubType<string name, string mnem> : HighLevelType< name,
  [ElementTypeInterface]
> {
  let mnemonic = mnem;

  let parameters = (ins
    "Type":$elementType
  );

  let assemblyFormat = "`<` $elementType `>`";
}

//
// LValue high-level type and its utilities
//
def LValue : TypeWithSubType< "LValue", "lvalue" > { }

class LValueOf<TypeConstraint value> : Type< And< [
    LValue.predicate,
    SubstLeaves< "$_self", "$_self.cast< LValueType >().getElementType()", value.predicate >
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
// RValue high-level type
//
def RValue : TypeWithSubType< "RValue", "rvalue" > { }

//
// Basic types
//
def VoidType : CVQualifiedType< "Void", "void", (ins), [VoidTrait] > {
  let builders = [ TypeBuilder<(ins), [{ return $_get($_ctxt); }]> ];
}

def BoolType : CVQualifiedType< "Bool", "bool", (ins) > {
  let builders = [ TypeBuilder<(ins), [{ return $_get($_ctxt); }]> ];
}

//
// Integer types & traits
//
class IntegerType< string name, string mnemonic, list<Trait> traits = []>
  : UCVQualifiedType< name, mnemonic, (ins),
    !listconcat(traits, [IntegralTypeTrait])
  >
{
  let builders = [ TypeBuilder<(ins), [{ return $_get($_ctxt); }]> ];
}

def CharType     : IntegerType< "Char", "char", [CharTypeTrait] >;
def ShortType    : IntegerType< "Short", "short", [ShortTypeTrait] >;
def IntType      : IntegerType< "Int", "int", [IntegerTypeTrait] >;
def LongType     : IntegerType< "Long", "long", [LongTypeTrait] >;
def LongLongType : IntegerType< "LongLong", "longlong", [LongLongTypeTrait] >;
def Int128Type   : IntegerType< "Int128", "int128", [Int128TypeTrait] >;

//
// Floating point types & traits
//
class FloatingType< string name, string mnem, list<Trait> traits = []>
  : CVQualifiedType< name, mnem, (ins),
    !listconcat(traits, [FloatingTypeTrait])
  >
{
  let builders = [ TypeBuilder<(ins), [{ return $_get($_ctxt); }]> ];
}

def HalfType       : FloatingType< "Half", "half" >;
def BFloat16Type   : FloatingType< "BFloat16", "bfloat16" >;
def FloatType      : FloatingType< "Float", "float" >;
def DoubleType     : FloatingType< "Double", "double" >;
def LongDoubleType : FloatingType< "LongDouble", "longdouble" >;
def Float128Type   : FloatingType< "Float128", "float128" >;

def ComplexType : CVRQualifiedType<
  "Complex", "complex"
  , (ins "Type":$elementType) > {
  let builders = [
    TypeBuilder<(ins "Type":$elementType), [{
      return $_get($_ctxt, elementType, CVRQualifiersAttr());
    }]>
  ];
  let assemblyFormat = "`<` $elementType (`,` $quals^)? `>`";
}

//
// Memory related types & traits
//
def PointerType : CVRQualifiedType< "Pointer", "ptr",
    (ins "Type":$elementType),
    [MemRefElementTypeInterface, PointerTypeTrait, ElementTypeInterface]
  >
{
  let builders = [
    TypeBuilder<(ins "Type":$element), [{
      return $_get($_ctxt, element, CVRQualifiersAttr());
    }]>,
    TypeBuilderWithInferredContext<(ins "Type":$element)>
  ];

  let assemblyFormat = "`<` $elementType (`,` $quals^ )? `>`";
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

def SizeParam : TypeParameter<"SizeParam", "size parameter for arrays"> {
  let printer = [{
    if ($_self.has_value())
      $_printer << $_self.value();
    else
      $_printer << "?";
  }];

  let parser = [{ [&] () -> mlir::FailureOr<SizeParam> {
    if (succeeded($_parser.parseOptionalQuestion())) {
      return unknown_size;
    }

    mlir::APInt value;
    if (succeeded($_parser.parseInteger(value))) {
      return SizeParam(value.getLimitedValue());
    }

    return mlir::failure();
  }() }];
}

class ArrayLikeType< string name, string mnem, list<Trait> traits = [] >
    : CVRQualifiedType< name, mnem,
    (ins SizeParam:$size, "Type":$elementType),
    !listconcat(traits, [MemRefElementTypeInterface])
  >
  , WithDimensionsClass
{
  let extraClassDeclaration = default_dimensions_impl;

  let builders = [
    TypeBuilder<(ins "SizeParam":$size, "Type":$element), [{
      return $_get($_ctxt, size, element, CVRQualifiersAttr());
    }]>
  ];

  let assemblyFormat = "`<` $size `,` $elementType (`,` $quals^ )? `>`";
}

def ArrayType  : ArrayLikeType< "Array", "array" >;
def VectorType : ArrayLikeType< "Vector", "vector" >;

def DecayedType : HighLevelType< "Decayed",
  [MemRefElementTypeInterface, ElementTypeInterface, PointerTypeTrait]
> {
  let mnemonic = "decayed";
  let parameters = (ins "Type":$elementType);
  let assemblyFormat = "`<` $elementType `>`";
}

def SubscriptableType : TypeConstraint<
  Or< [ArrayType.predicate, PointerType.predicate, DecayedType.predicate] >, "subscriptable type"
>;

def AttributedType : HighLevelType< "Attributed" > {
  let mnemonic = "attributed";
  // TODO: add attribute field
  let parameters = (ins "Type":$elementType);
  let assemblyFormat = "`<` $elementType `>`";
}

def AdjustedType : HighLevelType< "Adjusted" > {
  let mnemonic = "adjusted";
  let parameters = (ins "Type":$original, "Type":$adjusted);
  let assemblyFormat = "`<` $original `,` $adjusted `>`";
}

def ReferenceType : TypeWithSubType< "Reference", "reference" > { }

def TypeOfExprType : CVRQualifiedType<
  "TypeOfExpr", "typeof.expr"
  , (ins StringRefParameter<>:$name)
  , [MemRefElementTypeInterface, TypeOfTrait]
> {
  let builders = [
    TypeBuilder<(ins "llvm::StringRef":$name), [{
      return $_get($_ctxt, name, CVRQualifiersAttr());
    }]>
  ];

  let assemblyFormat = "`<` $name (`,` $quals^ )? `>`";
}

def TypeOfType : CVRQualifiedType<
  "TypeOfType", "typeof.type"
  , (ins "Type":$unmodifiedType)
  , [MemRefElementTypeInterface, TypeOfTrait]
> {
  let builders = [
    TypeBuilder<(ins "Type":$unmodifiedType), [{
      return $_get($_ctxt, unmodifiedType, CVRQualifiersAttr());
    }]>
  ];

  let assemblyFormat = "`<` $unmodifiedType (`,` $quals^ )? `>`";
}

def AutoType : CVRQualifiedType<
  "Auto", "auto",
  (ins OptionalParameter<"Type">:$deducedType), [MemRefElementTypeInterface]
> {
  let builders = [
    TypeBuilder<(ins "Type":$deducedType), [{
      return $_get($_ctxt, deducedType, CVRQualifiersAttr());
    }]>,
    TypeBuilder<(ins), [{
      return $_get($_ctxt, ::mlir::Type(), CVRQualifiersAttr());
    }]>,
    TypeBuilder<(ins "CVRQualifiersAttr":$quals), [{
      return $_get($_ctxt, ::mlir::Type(), std::move(quals));
    }]>
  ];
  let assemblyFormat = "`<` ( $deducedType^ )? (`,` $quals^ )? `>`";
}

def Atomic : CVRQualifiedType<
  "Atomic", "atomic"
  , (ins "Type":$elementType)
  , [MemRefElementTypeInterface]
> {
  let builders = [
    TypeBuilder<(ins "Type":$elementType), [{
      return $_get($_ctxt, elementType, CVRQualifiersAttr());
    }]>
  ];

  let assemblyFormat = "`<` $elementType (`,` $quals^ )? `>`";
}

// Utils with multiple dependencies

def UnresolvedType : Type<
  Or< [
    TypeOrElaboratedType< TypedefType >.predicate,
    TypeOrElaboratedType< TypeOfType >.predicate,
    TypeOrElaboratedType< TypeOfExprType >.predicate,
    TypeOrElaboratedType< AutoType >.predicate
  ] >
>;

class ContainsUnresolved< list< string > tl > : PredOpTrait< "contains unresolved type",
    Or< !foreach(type, tl, SubstLeaves< "$_self", "$" # type #".getType()", UnresolvedType.predicate >) >
>;

class TypesMatchOrUnresolved< list< string > tl > : PredOpTrait<
    "all types match or there is a typedef",
    Or< [AllTypesMatch< tl >.predicate, ContainsUnresolved< tl >.predicate] >
>;

def BoolLikeType : TypeConstraint<
  Or< [BoolType.predicate, AnyI1.predicate, UnresolvedType.predicate] >,
  "bool like type"
>;

class LValueOrType<TypeConstraint value> : Type<
  Or< [
    And< [
      LValue.predicate,
      SubstLeaves< "$_self", "$_self.cast< LValueType >().getElementType()",
                   Or< [ value.predicate, UnresolvedType.predicate ] > >
    ] >,
    value.predicate,
    UnresolvedType.predicate
  ] >,
  "lvalue to " # value.summary
>;

def HLIntegerType : AnyTypeOf<[
  CharType, ShortType, IntType, LongType, LongLongType, Int128Type, UnresolvedType
]>;

def IntegerLikeType : TypeConstraint<
  Or< [HLIntegerType.predicate, AnyInteger.predicate,
       UnresolvedType.predicate,
       ElaboratedType<EnumType>.predicate] >,
  "integer like type"
>;

def IntOrBoolType : TypeConstraint<
  Or< [IntegerLikeType.predicate, BoolLikeType.predicate] >,
  "bool or integer like type"
>;

def HLFloatType : AnyTypeOf<[
  HalfType, BFloat16Type, FloatType, DoubleType, LongDoubleType, Float128Type, UnresolvedType
]>;

def FloatLikeType : TypeConstraint<
  Or< [HLFloatType.predicate, AnyFloat.predicate] >,
  "float like type"
>;

#endif // VAST_DIALECT_HIGHLEVEL_IR_HIGHLEVELTYPES

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

include "vast/Dialect/Core/CoreAttrTypeBase.td"
include "vast/Dialect/Core/Utils.td"

//
// Generic high-level type
//
class HighLevel_Type< string name, list<Trait> traits = []>
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

class HighLevel_QualifierParam< string name, string desc >
  : OptionalParameter< "bool", desc >
{
  let printer = [{ $_printer << ($_self ? "}] # name # [{" : ""); }];
  let parser = [{ [&] () -> mlir::FailureOr<bool> {
    return succeeded($_parser.parseOptionalKeyword( "}] # name # [{" )); } ()
  }];
}

//
// high-level language qualifiers
//
def HighLevel_ConstParam    : HighLevel_QualifierParam<    "const", "const qualifier" >;
def HighLevel_VolatileParam : HighLevel_QualifierParam< "volatile", "volatile qualifier" >;
def HighLevel_RestrictParam : HighLevel_QualifierParam< "restrict", "restrict qualifier" >;

def HighLevel_UnsignedParam : HighLevel_QualifierParam< "unsigned", "unsigned qualifier" >;

class HighLevel_Qualifier< dag qualifier, list<Trait> traits > {
  dag param = qualifier;

  list<Trait> interface = traits;
}

def HighLevel_ConstQualifier    : HighLevel_Qualifier<
  (ins HighLevel_ConstParam:$is_const), [ConstQualifierInterface]
> {}

def HighLevel_VolatileQualifier : HighLevel_Qualifier<
  (ins HighLevel_VolatileParam:$is_volatile), [VolatileQualifierInterface]
> {}

def HighLevel_RestrictQualifier : HighLevel_Qualifier<
  (ins HighLevel_RestrictParam:$is_restrict), [RestrictQualifierInterface]
> {}

def HighLevel_UnsignedQualifier : HighLevel_Qualifier<
  (ins HighLevel_UnsignedParam:$is_unsigned), [UnsignedQualifierInterface]
> {}

// TODO move to tablegen utilities
class VAST_FlattenList< list<list<Trait>> traits > {
  list<Trait> result = !foldl([]<Trait>, traits, res, trait, !listconcat(res, trait));
}

class VAST_FlattenInsList< list<dag> dags > {
  dag result = !foldl((ins), dags, res, elem, !con(res, elem));
}

class HighLevel_QualfiersParams< string name, list< HighLevel_Qualifier > qualifiers>
  : HighLevel_Attr< name, "quals", VAST_FlattenList< !foreach(qual, qualifiers, qual.interface) >.result >
{
  let parameters = VAST_FlattenInsList< !foreach(qual, qualifiers, qual.param) >.result;

  // See commit `3dbac2c007c11` in the upstream.
  string attrName = name;

  let assemblyFormat = "params ` `";
}

def HighLevel_CVQualifiers  : HighLevel_QualfiersParams< "CVQualifiers",
  [HighLevel_ConstQualifier, HighLevel_VolatileQualifier]
> {}

def HighLevel_UCVQualifiers : HighLevel_QualfiersParams< "UCVQualifiers",
  [HighLevel_UnsignedQualifier, HighLevel_ConstQualifier, HighLevel_VolatileQualifier]
> {}

def HighLevel_CVRQualifiers : HighLevel_QualfiersParams< "CVRQualifiers",
  [HighLevel_ConstQualifier, HighLevel_VolatileQualifier, HighLevel_RestrictQualifier]
> {}

class HighLevel_QualifiedType< string name, string mnem, dag params = (ins), list<Trait> traits = [] >
  : HighLevel_Type< name, traits >
{
  let mnemonic = mnem;

  let parameters = params;

  let assemblyFormat = "(`<` $quals^ `>`)?";
}

class HighLevel_CVQualifiedType< string name, string mnem, dag params = (ins), list<Trait> traits = [] >
  : HighLevel_QualifiedType< name, mnem, !con(params, (ins OptionalParameter< "CVQualifiersAttr" >:$quals)), traits >
{}

class HighLevel_UCVQualifiedType< string name, string mnem, dag params = (ins), list<Trait> traits = [] >
  : HighLevel_QualifiedType< name, mnem, !con(params, (ins OptionalParameter< "UCVQualifiersAttr" >:$quals)), traits >
{}

class HighLevel_CVRQualifiedType< string name, string mnem, dag params = (ins), list<Trait> traits = [] >
  : HighLevel_QualifiedType< name, mnem, !con(params, (ins OptionalParameter< "CVRQualifiersAttr" >:$quals)), traits >
{}

//
// Tagged types and utils
//
def HighLevel_RecordType : HighLevel_CVQualifiedType<
  "Record", "record", (ins Core_SymbolRefParameter<>:$name), [MemRefElementTypeInterface]
> {

  let builders = [
    TypeBuilder<(ins "llvm::StringRef":$name), [{
      return $_get($_ctxt, name, CVQualifiersAttr());
    }]>
  ];

  let assemblyFormat = "`<` $name (`,` $quals^ )? `>`";
}

def HighLevel_EnumType : HighLevel_CVQualifiedType<
  "Enum", "enum", (ins Core_SymbolRefParameter<>:$name), [MemRefElementTypeInterface]
> {
  let builders = [
    TypeBuilder<(ins "llvm::StringRef":$name), [{
      return $_get($_ctxt, name, CVQualifiersAttr());
    }]>
  ];

  let assemblyFormat = "`<` $name (`,` $quals^ )? `>`";
}

def HighLevel_TypedefType : HighLevel_CVRQualifiedType<
  "Typedef", "typedef", (ins Core_SymbolRefParameter<>:$name),
  [MemRefElementTypeInterface, Core_TypedefTrait]
> {
  let builders = [
    TypeBuilder<(ins "llvm::StringRef":$name), [{
      return $_get($_ctxt, name, CVRQualifiersAttr());
    }]>
  ];

  let assemblyFormat = "`<` $name (`,` $quals^ )? `>`";
}

def HighLevel_ElaboratedType : HighLevel_CVRQualifiedType<
  "Elaborated", "elaborated", (ins "Type":$elementType), [ElementTypeInterface, MemRefElementTypeInterface]
> {
  let builders = [
    TypeBuilder<(ins "Type":$element), [{
      return $_get($_ctxt, element, CVRQualifiersAttr());
    }]>
  ];

  let assemblyFormat = "`<` $elementType (`,` $quals^ )? `>`";
}

def HighLevel_LabelType : HighLevel_Type< "Label" > {
  let mnemonic = "label";
}

def HighLevel_ParenType : HighLevel_Type< "Paren", [ElementTypeInterface, MemRefElementTypeInterface] > {
  let mnemonic = "paren";
  let parameters = (ins "Type":$elementType);
  let assemblyFormat = "`<` $elementType `>`";
}

class HighLevel_ElaboratedType< TypeConstraint value > : Type<
  And< [
    HighLevel_ElaboratedType.predicate,
    SubstLeaves< "$_self", "mlir::cast< ElaboratedType >($_self).getElementType()", value.predicate >
  ] >
>;

class HighLevel_TypeOrElaboratedType< TypeConstraint value > : Type<
  Or< [
    value.predicate,
    HighLevel_ElaboratedType< value >.predicate
  ]>
>;

class HighLevel_TypeWithSubType<string name, string mnem>
  : HighLevel_Type< name, [ElementTypeInterface] >
{
  let mnemonic = mnem;

  let parameters = (ins
    "Type":$elementType
  );

  let assemblyFormat = "`<` $elementType `>`";
}

//
// LValue high-level type and its utilities
//
def HighLevel_LValue : HighLevel_TypeWithSubType< "LValue", "lvalue" > { }

class HighLevel_LValueOf<TypeConstraint value> : Type< And< [
    HighLevel_LValue.predicate,
    SubstLeaves< "$_self", "mlir::cast< LValueType >($_self).getElementType()", value.predicate >
  ] >,
  "lvalue to " # value.summary
>;

class HighLevel_TypesOrLValueTypesMatch<string summary, string lhs, string rhs> : PredOpTrait< summary,
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
def HighLevel_RValue : HighLevel_TypeWithSubType< "RValue", "rvalue" > { }

//
// Basic types
//
def HighLevel_VoidType
  : HighLevel_CVQualifiedType< "Void", "void", (ins), [Core_VoidTrait] >
{
  let builders = [ TypeBuilder<(ins), [{ return $_get($_ctxt); }]> ];
}

def HighLevel_BoolType
  : HighLevel_CVQualifiedType< "Bool", "bool", (ins), [Core_BoolTypeTrait] >
{
  let builders = [ TypeBuilder<(ins), [{ return $_get($_ctxt); }]> ];
}

//
// Integer types & traits
//
class HighLevel_IntegerType< string name, string mnemonic, list<Trait> traits = []>
  : HighLevel_UCVQualifiedType< name, mnemonic, (ins),
    !listconcat(traits, [Core_IntegralTypeTrait])
  >
{
  let builders = [ TypeBuilder<(ins), [{ return $_get($_ctxt); }]> ];
}

def HighLevel_CharType     : HighLevel_IntegerType< "Char", "char", [Core_CharTypeTrait] >;
def HighLevel_ShortType    : HighLevel_IntegerType< "Short", "short", [Core_ShortTypeTrait] >;
def HighLevel_IntType      : HighLevel_IntegerType< "Int", "int", [Core_IntegerTypeTrait] >;
def HighLevel_LongType     : HighLevel_IntegerType< "Long", "long", [Core_LongTypeTrait] >;
def HighLevel_LongLongType : HighLevel_IntegerType< "LongLong", "longlong", [Core_LongLongTypeTrait] >;
def HighLevel_Int128Type   : HighLevel_IntegerType< "Int128", "int128", [Core_Int128TypeTrait] >;

//
// Floating point types & traits
//
class HighLevel_FloatingType< string name, string mnem, list<Trait> traits = []>
  : HighLevel_CVQualifiedType< name, mnem, (ins),
    !listconcat(traits, [Core_FloatingTypeTrait])
  >
{
  let builders = [ TypeBuilder<(ins), [{ return $_get($_ctxt); }]> ];
}

def HighLevel_HalfType       : HighLevel_FloatingType< "Half", "half" >;
def HighLevel_BFloat16Type   : HighLevel_FloatingType< "BFloat16", "bfloat16" >;
def HighLevel_FloatType      : HighLevel_FloatingType< "Float", "float" >;
def HighLevel_DoubleType     : HighLevel_FloatingType< "Double", "double" >;
def HighLevel_LongDoubleType : HighLevel_FloatingType< "LongDouble", "longdouble" >;
def HighLevel_Float128Type   : HighLevel_FloatingType< "Float128", "float128" >;

def HighLevel_ComplexType : HighLevel_CVRQualifiedType<
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
def HighLevel_PointerType : HighLevel_CVRQualifiedType< "Pointer", "ptr",
    (ins "Type":$elementType),
    [MemRefElementTypeInterface, Core_PointerTypeTrait, ElementTypeInterface]
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

def HighLevel_PointerLikeType : TypeConstraint< HighLevel_PointerType.predicate, "pointer like type" >;

//
// Array types
//
class HighLevel_WithDimensionsClass
{
  code default_dimensions_impl = [{
    using dimensions_t = std::vector< SizeParam >;

    std::tuple< dimensions_t, mlir::Type > dim_and_type();

    dimensions_t dimensions() { return std::get< 0 >(dim_and_type()); }
  }];
}

def HighLevel_SizeParam
  : TypeParameter< "SizeParam", "size parameter for arrays" >
{
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

class HighLevel_ArrayLikeType< string name, string mnem, list<Trait> traits = [] >
    : HighLevel_CVRQualifiedType< name, mnem,
    (ins HighLevel_SizeParam:$size, "Type":$elementType),
    !listconcat(traits, [MemRefElementTypeInterface])
  >
  , HighLevel_WithDimensionsClass
{
  let extraClassDeclaration = default_dimensions_impl;

  let builders = [
    TypeBuilder<(ins "SizeParam":$size, "Type":$element), [{
      return $_get($_ctxt, size, element, CVRQualifiersAttr());
    }]>
  ];

  let assemblyFormat = "`<` $size `,` $elementType (`,` $quals^ )? `>`";
}

def HighLevel_ArrayType  : HighLevel_ArrayLikeType< "Array", "array" >;
def HighLevel_VectorType : HighLevel_ArrayLikeType< "Vector", "vector" >;

def HighLevel_DecayedType : HighLevel_Type< "Decayed",
  [MemRefElementTypeInterface, ElementTypeInterface, Core_PointerTypeTrait]
> {
  let mnemonic = "decayed";
  let parameters = (ins "Type":$elementType);
  let assemblyFormat = "`<` $elementType `>`";
}

// FIXME this should be trait of the type
def HighLevel_SubscriptableType : TypeConstraint<
  Or< [
    HighLevel_ArrayType.predicate,
    HighLevel_PointerType.predicate,
    HighLevel_DecayedType.predicate
  ] >, "subscriptable type"
>;

def HighLevel_AttributedType : HighLevel_Type< "Attributed" > {
  let mnemonic = "attributed";
  // TODO: add attribute field
  let parameters = (ins "Type":$elementType);
  let assemblyFormat = "`<` $elementType `>`";
}

def HighLevel_AdjustedType : HighLevel_Type< "Adjusted" > {
  let mnemonic = "adjusted";
  let parameters = (ins "Type":$original, "Type":$adjusted);
  let assemblyFormat = "`<` $original `,` $adjusted `>`";
}

def HighLevel_ReferenceType : HighLevel_TypeWithSubType< "Reference", "reference" > { }

def HighLevel_TypeOfExprType : HighLevel_CVRQualifiedType<
  "TypeOfExpr", "typeof.expr"
  , (ins StringRefParameter<>:$name)
  , [MemRefElementTypeInterface, Core_TypeOfTrait]
> {
  let builders = [
    TypeBuilder<(ins "llvm::StringRef":$name), [{
      return $_get($_ctxt, name, CVRQualifiersAttr());
    }]>
  ];

  let assemblyFormat = "`<` $name (`,` $quals^ )? `>`";
}

def HighLevel_TypeOfType : HighLevel_CVRQualifiedType<
  "TypeOfType", "typeof.type"
  , (ins "Type":$unmodifiedType)
  , [MemRefElementTypeInterface, Core_TypeOfTrait]
> {
  let builders = [
    TypeBuilder<(ins "Type":$unmodifiedType), [{
      return $_get($_ctxt, unmodifiedType, CVRQualifiersAttr());
    }]>
  ];

  let assemblyFormat = "`<` $unmodifiedType (`,` $quals^ )? `>`";
}

def HighLevel_AutoType : HighLevel_CVRQualifiedType<
  "Auto", "auto"
  , (ins OptionalParameter<"Type">:$deducedType)
  , [MemRefElementTypeInterface, Core_AutoTrait]
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

def HighLevel_Atomic : HighLevel_CVRQualifiedType<
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

// FIXME this should be trait of the type
def HighLevel_UnresolvedType : Type<
  Or< [
    HighLevel_TypeOrElaboratedType< HighLevel_TypedefType >.predicate,
    HighLevel_TypeOrElaboratedType< HighLevel_TypeOfType >.predicate,
    HighLevel_TypeOrElaboratedType< HighLevel_TypeOfExprType >.predicate,
    HighLevel_TypeOrElaboratedType< HighLevel_AutoType >.predicate
  ] >
>;

class HighLevel_ContainsUnresolved< list< string > tl > : PredOpTrait< "contains unresolved type",
    Or< !foreach(type, tl, SubstLeaves< "$_self", "$" # type #".getType()", HighLevel_UnresolvedType.predicate >) >
>;

class HighLevel_TypesMatchOrUnresolved< list< string > tl > : PredOpTrait<
    "all types match or there is a typedef",
    Or< [AllTypesMatch< tl >.predicate, HighLevel_ContainsUnresolved< tl >.predicate] >
>;

def HighLevel_BoolLikeType : TypeConstraint<
  Or< [HighLevel_BoolType.predicate, AnyI1.predicate, HighLevel_UnresolvedType.predicate] >,
  "bool like type"
>;

class HighLevel_LValueOrType<TypeConstraint value> : Type<
  Or< [
    And< [
      HighLevel_LValue.predicate,
      SubstLeaves< "$_self", "mlir::cast< LValueType >($_self).getElementType()",
        Or< [ value.predicate, HighLevel_UnresolvedType.predicate ] >
      >
    ] >,
    value.predicate,
    HighLevel_UnresolvedType.predicate
  ] >,
  "lvalue to " # value.summary
>;

// FIXME this should be trait of the type
def HighLevel_AnyIntegerType : AnyTypeOf<[
  HighLevel_CharType,
  HighLevel_ShortType,
  HighLevel_IntType,
  HighLevel_LongType,
  HighLevel_LongLongType,
  HighLevel_Int128Type,
  HighLevel_UnresolvedType
]>;

def HighLevel_IntegerLikeType : TypeConstraint<
  Or< [
    HighLevel_AnyIntegerType.predicate,
    AnyInteger.predicate,
    HighLevel_UnresolvedType.predicate,
    HighLevel_ElaboratedType<HighLevel_EnumType>.predicate
  ] >,
  "integer like type"
>;

def HighLevel_IntOrBoolType : TypeConstraint<
  Or< [
    HighLevel_IntegerLikeType.predicate,
    HighLevel_BoolLikeType.predicate
  ] >,
  "bool or integer like type"
>;

// FIXME this should be trait of the type
def HighLevel_AnyFloatType : AnyTypeOf<[
  HighLevel_HalfType,
  HighLevel_BFloat16Type,
  HighLevel_FloatType,
  HighLevel_DoubleType,
  HighLevel_LongDoubleType,
  HighLevel_Float128Type,
  HighLevel_UnresolvedType
]>;

def HighLevel_FloatLikeType : TypeConstraint<
  Or< [
    HighLevel_AnyFloatType.predicate,
    AnyFloat.predicate
  ] >,
  "float like type"
>;

#endif // VAST_DIALECT_HIGHLEVEL_IR_HIGHLEVELTYPES

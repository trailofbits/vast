// Copyright (c) 2021-present, Trail of Bits, Inc.

#ifndef VAST_DIALECT_HIGHLEVEL_IR_HIGHLEVELTYPES
#define VAST_DIALECT_HIGHLEVEL_IR_HIGHLEVELTYPES

include "mlir/Interfaces/DataLayoutInterfaces.td"
include "mlir/IR/SubElementInterfaces.td"
include "mlir/IR/BuiltinTypeInterfaces.td"

include "vast/Dialect/HighLevel/HighLevelAttributes.td"
include "vast/Interfaces/TypeQualifiersInterfaces.td"

include "vast/Dialect/Core/Utils.td"

// NOTE(lukas): I have no idea how to do this both in a sane way and in tablegen
class WithDefaultDLTypeInterface {
  code default_dl_interface_impl = [{
    using dl_t = mlir::DataLayout;
    using dl_entries_ref = mlir::DataLayoutEntryListRef;

    unsigned getTypeSizeInBits(const dl_t &dl, dl_entries_ref entries) const
    {
        VAST_CHECK(entries.size() != 0,
            "Query for getTypeSizeInBits for {0} failed: Must have at least one entry!",
            format_type(*this)
        );

        std::optional<uint32_t> out;
        auto handle_entry = [&](auto &dl_entry) {
            if (!out) out = dl_entry.bw;
            VAST_CHECK(*out == dl_entry.bw,
                       "Inconsistent entries {0}, {1}, entry count: {2}",
                       *out, dl_entry.bw, entries.size());
        };
        apply_on_valid_entries(entries, handle_entry);
        VAST_CHECK(out.has_value(), "getTypeSizeBits entries did not yield result.");
        return *out;
    }

    unsigned getABIAlignment(const dl_t &dl, dl_entries_ref entries) const
    {
        VAST_UNIMPLEMENTED;
    }

    unsigned getPreferredAlignment(const dl_t &dl, dl_entries_ref entries) const
    {
        VAST_UNIMPLEMENTED;
    }

    void apply_on_valid_entries(dl_entries_ref entries, auto &f) const
    {
        for (const auto &entry : entries)
        {
            auto dl_entry = dl::DLEntry::unwrap(entry);
            if (is_valid_entry_type(dl_entry.type))
                f(dl_entry);
        }
    }

    bool is_valid_entry_type(mlir::Type t) const
    {
        return t == *this;
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

  let assemblyFormat = "params ` `";
}

def CVQualifiers  : QualfiersParams< "CVQualifiers",  [ConstQualifier, VolatileQualifier] > {}
def UCVQualifiers : QualfiersParams< "UCVQualifiers", [UnsignedQualifier, ConstQualifier, VolatileQualifier] > {}
def CVRQualifiers : QualfiersParams< "CVRQualifiers", [ConstQualifier, VolatileQualifier, RestrictQualifier] > {}

class QualifiedType< string name, string mnem, dag params = (ins), list<Trait> traits = [] >
  : HighLevelType< name, !listconcat(traits, []) >
{
  let mnemonic = mnem;

  let parameters = params;

  // TODO remove
  let extraClassDeclaration = default_dl_interface_impl;

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
  "Elaborated", "elaborated", (ins "Type":$elementType), [DeclareTypeInterfaceMethods< SubElementTypeInterface >]
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

def ParenType : HighLevelType< "Paren", [MemRefElementTypeInterface] > {
  let mnemonic = "paren";
  let parameters = (ins "Type":$elementType);
  let assemblyFormat = "`<` $elementType `>`";
}

class IsMaybeWrappedTypedef< string arg > : PredOpTrait<
    "type is a typedef, maybe wrapped in the elaborated type",
    CPred< "::vast::hl::strip_elaborated($" # arg # ").hasTrait< mlir::TypeTrait::TypedefTrait >()" >
>;

class ContainsTypedef< list< string > tl > : PredOpTrait< "contains typedef type",
    Or< !foreach(type, tl, IsMaybeWrappedTypedef< type >.predicate) >
>;

class TypesMatchOrTypedef< list< string > tl > : PredOpTrait<
    "all types match or there is a typedef",
    Or< [AllTypesMatch< tl >.predicate, ContainsTypedef< tl >.predicate ] >
>;

class TypeWithSubType<string name, string mnem> : HighLevelType< name,
  [DeclareTypeInterfaceMethods< SubElementTypeInterface >]
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

class LValueOrType<TypeConstraint value> : Type<
  Or< [
    And< [
      LValue.predicate,
      SubstLeaves< "$_self", "$_self.cast< LValueType >().getElementType()", value.predicate >
    ] >,
    value.predicate,
    TypedefType.predicate
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
def VoidType : CVQualifiedType< "Void", "void" > {
  let builders = [ TypeBuilder<(ins), [{ return $_get($_ctxt); }]> ];
}

def BoolType : CVQualifiedType< "Bool", "bool" > {
  let builders = [ TypeBuilder<(ins), [{ return $_get($_ctxt); }]> ];
}

def BoolLikeType : TypeConstraint<
  Or< [BoolType.predicate, AnyI1.predicate, TypedefType.predicate] >,
  "bool like type"
>;

//
// Integer types & traits
//
class IntegerType< string name, string mnemonic, list<Trait> traits = []>
  : UCVQualifiedType< name, mnemonic, (ins), !listconcat(traits, [IntegralTypeTrait]) >
{
  let builders = [ TypeBuilder<(ins), [{ return $_get($_ctxt); }]> ];
}

def CharType     : IntegerType< "Char", "char", [CharTypeTrait] >;
def ShortType    : IntegerType< "Short", "short", [ShortTypeTrait] >;
def IntType      : IntegerType< "Int", "int", [IntegerTypeTrait] >;
def LongType     : IntegerType< "Long", "long", [LongTypeTrait] >;
def LongLongType : IntegerType< "LongLong", "longlong", [LongLongTypeTrait] >;
def Int128Type   : IntegerType< "Int128", "int128", [Int128TypeTrait] >;

def HLIntegerType : AnyTypeOf<[
  CharType, ShortType, IntType, LongType, LongLongType, Int128Type, TypedefType
]>;

def IntegerLikeType : TypeConstraint<
  Or< [HLIntegerType.predicate, AnyInteger.predicate,
       IsMaybeWrappedTypedef< "_self" >.predicate] >,
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
  : CVQualifiedType< name, mnem, (ins), !listconcat(traits, [FloatingTypeTrait]) >
{
  let builders = [ TypeBuilder<(ins), [{ return $_get($_ctxt); }]> ];
}

def HalfType       : FloatingType< "Half", "half" >;
def BFloat16Type   : FloatingType< "BFloat16", "bfloat16" >;
def FloatType      : FloatingType< "Float", "float" >;
def DoubleType     : FloatingType< "Double", "double" >;
def LongDoubleType : FloatingType< "LongDouble", "longdouble" >;
def Float128Type   : FloatingType< "Float128", "float128" >;

def HLFloatType : AnyTypeOf<[
  HalfType, BFloat16Type, FloatType, DoubleType, LongDoubleType, Float128Type, TypedefType
]>;

def FloatLikeType : TypeConstraint<
  Or< [HLFloatType.predicate, AnyFloat.predicate] >,
  "float like type"
>;

//
// Memory related types & traits
//
def PointerType : CVRQualifiedType< "Pointer", "ptr",
    (ins "Type":$elementType),
    [DeclareTypeInterfaceMethods< SubElementTypeInterface >,
     MemRefElementTypeInterface,
     PointerTypeTrait
    ]
  >
{
  let builders = [
    TypeBuilder<(ins "Type":$element), [{
      return $_get($_ctxt, element, CVRQualifiersAttr());
    }]>
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

def ArrayType : CVRQualifiedType< "Array", "array",
    (ins SizeParam:$size, "Type":$elementType),
    [DeclareTypeInterfaceMethods< SubElementTypeInterface >, MemRefElementTypeInterface]
  >
  , WithDimensionsClass
{
  let extraClassDeclaration = default_dl_interface_impl # default_dimensions_impl;

  let builders = [
    TypeBuilder<(ins "SizeParam":$size, "Type":$element), [{
      return $_get($_ctxt, size, element, CVRQualifiersAttr());
    }]>
  ];

  let assemblyFormat = "`<` $size `,` $elementType (`,` $quals^ )? `>`";
}

def DecayedType : HighLevelType< "Decayed",
                                 [SubElementTypeInterface, MemRefElementTypeInterface] >
{
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

def TypeOfType : HighLevelType< "TypeOf" > {
  let mnemonic = "typeof.type";
  let parameters = (ins "Type":$desugaredType);
  let assemblyFormat = "`<` $desugaredType `>`";
}

#endif // VAST_DIALECT_HIGHLEVEL_IR_HIGHLEVELTYPES

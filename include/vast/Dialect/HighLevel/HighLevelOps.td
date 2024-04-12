// Copyright (c) 2021-present, Trail of Bits, Inc.

#ifndef VAST_DIALECT_HIGHLEVEL_IR_HIGHLEVELOPS
#define VAST_DIALECT_HIGHLEVEL_IR_HIGHLEVELOPS

include "mlir/IR/OpBase.td"
include "mlir/IR/BuiltinAttributes.td"

include "vast/Interfaces/AggregateTypeDefinitionInterface.td"
include "mlir/Interfaces/CastInterfaces.td"
include "vast/Interfaces/SymbolInterface.td"

include "vast/Dialect/HighLevel/HighLevelCF.td"

include "vast/Dialect/Core/Utils.td"
include "vast/Dialect/Core/CoreTraits.td"

include "vast/Dialect/Core/Func.td"


def TranslationUnitOp
  : HighLevel_Op< "translation_unit", [NoTerminator, SymbolTable, IsolatedFromAbove] >
{
  let summary = "VAST translation unit";
  let description = [{ VAST tranaslation unit }];

  let regions = (region AnyRegion:$body);

  let assemblyFormat = [{ $body attr-dict }];
}

def HighLevel_FuncOp
  : Core_FuncBaseOp< HighLevel_Dialect, "func" >
{
  let skipDefaultBuilders = 1;

  let builders = [OpBuilder< (ins
    "llvm::StringRef":$name,
    "core::FunctionType":$type,
    CArg< "core::GlobalLinkageKind", "core::GlobalLinkageKind::ExternalLinkage" >:$linkage,
    CArg< "llvm::ArrayRef<mlir::NamedAttribute>", "{}" >:$attrs,
    CArg< "llvm::ArrayRef<mlir::DictionaryAttr>", "{}" >:$arg_attrs,
    CArg< "llvm::ArrayRef<mlir::DictionaryAttr>", "{}" >:$res_attrs,
    CArg< "BuilderCallback", "std::nullopt" >:$body), [{
      InsertionGuard guard($_builder);
      build_region($_builder, $_state, body);

      $_state.addAttribute(
        mlir::SymbolTable::getSymbolAttrName(), $_builder.getStringAttr(name)
      );
      $_state.addAttribute(getFunctionTypeAttrName($_state.name), mlir::TypeAttr::get(type));
      $_state.addAttribute(
        "linkage", core::GlobalLinkageKindAttr::get($_builder.getContext(), linkage)
      );
      $_state.attributes.append(attrs.begin(), attrs.end());

      mlir::function_interface_impl::addArgAndResultAttrs(
        $_builder, $_state, arg_attrs, res_attrs,
        getArgAttrsAttrName($_state.name), getResAttrsAttrName($_state.name)
      );
    }] >
  ];

  let hasVerifier = 1;
}

def TypeDeclOp
  : HighLevel_Op< "type", [VastSymbol] >
  , Arguments<(ins StrAttr:$name)>
{
  let summary = "VAST type declaration";
  let description = [{ VAST type declaration }];

  let extraClassDeclaration = [{
    mlir_type getDefinedType();
  }];

  let assemblyFormat = [{ $name attr-dict }];
}

def TypeDefOp
  : HighLevel_Op< "typedef", [VastSymbol] >
  , Arguments<(ins StrAttr:$name, TypeAttr:$type)>
{
  let summary = "VAST typedef operation";
  let description = [{
    Typedef operation servers to declare named types.
    It creates a new type symbol in the current scope to
    be referenced as NamedType later.
  }];

  let extraClassDeclaration = [{
    mlir_type getDefinedType();
  }];

  let assemblyFormat = [{ $name attr-dict `:` $type }];
}

def TypeOfTypeOp
  : HighLevel_Op< "typeof.type", [VastSymbol] >
  , Arguments<(ins TypeAttr:$type)>
{
  let summary = "VAST typeof(type) operation";

  let assemblyFormat = [{ attr-dict `:` $type }];
}

def TypeOfExprOp
  : HighLevel_Op< "typeof.expr", [VastSymbol, SingleBlock] >
  , Arguments<(ins StrAttr:$name, TypeAttr:$type)>
{
  let summary = "VAST typeof(expr) operation";
  let description = [{
    The Typeof operation serves to declare a type using type introspection.
    It evaluates its underlying expression, creates a new type symbol in the
    current scope, assigns it to the type of the underlying expression, and
    returns the type symbol to be referenced later
  }];

  let regions = (region SizedRegion<1>:$expr);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins
      "llvm::StringRef":$name,
      "Type":$type,
      CArg<"BuilderCallback", "std::nullopt">:$expr
    )>
  ];

  let assemblyFormat = [{ $name $expr `:` $type attr-dict }];
}

def EnumConstantOp
  : HighLevel_Op< "enum.const" >
  , Arguments<(ins StrAttr:$name, TypedAttrInterface:$value)>
{
  let summary = "VAST enum constant declaration";
  let description = [{
    Enumeration constant servers to link name to an enum value.
    It is required to be scoped in Enum operation. For example:

    ```
    hl.enum.const "F" = 2 : !hl.int
    ```

    A constant can have a constant expression initializer:

    ```
    hl.enum.const "G" = #core.integer<12> : !hl.int init  {
      %0 = hl.enumref "F" : !hl.int
      %1 = hl.enumref "C" : !hl.int
      %2 = hl.add %0, %1 : !hl.int
      hl.value.yield %2 : !hl.int
    }
    ```
  }];

  let regions = (region AnyRegion:$init);

  let builders = [
    OpBuilder<(ins
      "llvm::StringRef":$name,
      "Type":$type, "llvm::APSInt":$value,
      CArg< "BuilderCallback", "std::nullopt" >:$init
    ), [{
      $_state.addAttribute("name", $_builder.getStringAttr(name));
      $_state.addAttribute("value", core::IntegerAttr::get(type, value));

      InsertionGuard guard($_builder);
      build_region($_builder, $_state, init);
    }]>,
  ];

  let assemblyFormat = [{ $name `=` $value attr-dict (`init` $init^)? }];
}

def EnumRegion : Region< HasOneBlock, "enum constants region" >;

def EnumDeclOp
  : HighLevel_Op< "enum", [NoTerminator, VastSymbol] >
  // TODO(Heno): check integer like type attr
  , Arguments<(ins StrAttr:$name, OptionalAttr<TypeAttr>:$type)>
{
  let summary = "VAST enum declaration";
  let description = [{
    Enum declaration serves to declare region for enum constant declarations.
    It also defines an underlying type.
  }];

  let regions = (region EnumRegion:$constants);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins
      "llvm::StringRef":$name,
      "mlir::Type":$type,
      "BuilderCallback":$constants
    )>,
    OpBuilder<(ins
      "llvm::StringRef":$name
    )>
  ];

  let assemblyFormat = [{ $name attr-dict `:` ($type^ $constants)? }];
}

class RecordLikeDeclOp< string mnemonic, list< Trait > traits = [] >
    : HighLevel_Op<
        mnemonic,
        !listconcat(traits, [NoTerminator, VastSymbol,
                            DeclareOpInterfaceMethods< AggregateTypeDefinition >]) >
    , Arguments<(ins StrAttr:$name)>
{
  // TODO(Heno): Add region constraints.
  let regions = (region AnyRegion:$fields);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins
      "llvm::StringRef":$name,
       CArg<"BuilderCallback", "std::nullopt">:$fields
    )>
  ];

  let extraClassDefinition = [{
    // AggregateTypeDefinitionInterface

    gap::generator< mlir_type > $cppClass::getFieldTypes() {
        return hl::get_field_types(*this);
    }

    gap::generator< vast::field_info_t > $cppClass::getFieldsInfo() {
        return hl::get_fields_info(*this);
    }

    gap::generator< vast::AggregateTypeDefinitionInterface > $cppClass::getNestedDeclarations() {
        return hl::get_nested_declarations(*this);
    }

    llvm::StringRef $cppClass::getDefinedName() { return this->getName(); }

    mlir::Type $cppClass::getDefinedType() {
      return hl::RecordType::get(getContext(), getDefinedName());
    }
  }];

  let assemblyFormat = [{ $name attr-dict `:` $fields }];
}

def StructDeclOp : RecordLikeDeclOp< "struct" > {
  let summary = "VAST struct declaration";
  let description = [{ VAST struct declaration }];
}

def UnionDeclOp : RecordLikeDeclOp< "union" > {
  let summary = "VAST record declaration";
  let description = [{ VAST record declaration }];
}

def FieldDeclOp
  : HighLevel_Op< "field", [VastSymbol] >
  , Arguments<(ins StrAttr:$name, TypeAttr:$type, OptionalAttr<I32Attr>:$bits)>
{
  let summary = "VAST record field declaration";
  let description = [{ VAST record field declaration }];

  let extraClassDeclaration = [{
    vast::AggregateTypeDefinitionInterface getParentAggregate() {
        return mlir::dyn_cast< vast::AggregateTypeDefinitionInterface >((*this)->getParentOp());
    }
  }];

  let assemblyFormat = [{ $name attr-dict (`bw` $bits^)? `:` $type }];
}

def RecordMemberOp
  : HighLevel_Op< "member" >
  // TODO(Heno): add type constraints
  , Arguments<(ins AnyType:$record, StrAttr:$name)>
  , Results<(outs LValueOf<AnyType>:$element)>
{
  let summary = "VAST record element access operation";
  let description = [{ VAST record element access operation }];

  let assemblyFormat = [{
    $record `at` $name attr-dict `:` type($record) `->` type($element)
  }];
}

def CallOp
  : HighLevel_Op< "call", [DeclareOpInterfaceMethods<CallOpInterface>] >
  , Arguments<(ins
      FlatSymbolRefAttr:$callee,
      Variadic<AnyType>:$argOperands)>
  , Results<(outs Variadic<AnyType>:$results)>
{
  let summary = "VAST call operation";
  let description = [{ VAST call operation }];

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder< (ins "FuncOp":$callee, CArg< "mlir::ValueRange", "{}" >:$operands ), [{
      $_state.addOperands(operands);
      $_state.addAttribute("callee", mlir::SymbolRefAttr::get($_builder.getContext(), callee.getName()));
      $_state.addTypes(callee.getFunctionType().getResults());
    }]>,

    OpBuilder< (ins "llvm::StringRef":$callee,
                "mlir::TypeRange":$results,
                CArg<"mlir::ValueRange", "{}">:$operands),
    [{
      $_state.addOperands(operands);
      $_state.addAttribute("callee", mlir::SymbolRefAttr::get($_builder.getContext(), callee));
      $_state.addTypes(results);
    }]>];

  let assemblyFormat = [{
    $callee `(` $argOperands `)` attr-dict `:` functional-type( $argOperands, $results )
  }];
}

def IndirectCallOp
  : HighLevel_Op< "indirect_call", [
    DeclareOpInterfaceMethods<CallOpInterface>,
  ] >
  , Arguments<(ins
      // FIXME: LValueOrType<PointerLikeType>
      AnyType:$callee,
      Variadic<AnyType>:$argOperands)>
  , Results<(outs Variadic<AnyType>:$results)>
{
  let summary = "VAST call operation";
  let description = [{ VAST call operation }];

  let assemblyFormat = [{
    $callee `:` type($callee)  `(` $argOperands `)` attr-dict `:` functional-type( $argOperands, $results )
  }];

  // TODO: add verifiers to check that callee type matches arg operands
}

def ExprOp
  : HighLevel_Op< "expr", [SingleBlock] >
  , Results<(outs AnyType:$result)>
{
  let summary = "VAST expression";
  let description = [{ VAST expression }];

  let regions = (region SizedRegion<1>:$subexpr);

  let skipDefaultBuilders = 1;
  let builders = [ OpBuilder<(ins "Type":$rty, "std::unique_ptr< Region > &&":$region)> ];

  let assemblyFormat = [{
     attr-dict `:` type($result) $subexpr
  }];
}

def ReturnOp
  : HighLevel_Op< "return", [SoftTerminatorTrait, ReturnLikeTrait] >
  , Arguments<(ins Variadic<AnyType>:$result)>
{
  let assemblyFormat = "($result^ `:` type($result))? attr-dict";

  // Allow building a ReturnOp with no return operand.
  let builders = [
    OpBuilder<(ins), [{ build($_builder, $_state, std::nullopt); }]>
  ];
}

// use InferTypeOpInterface
def DeclRefOp
  : HighLevel_Op< "ref" >
  , Arguments<(ins AnyType:$decl)>
  , Results<(outs LValueOf<AnyType>:$result)>
{
  let summary = "VAST variable reference declaration";
  let description = [{ VAST variable reference declaration }];

  let assemblyFormat = "$decl attr-dict `:` functional-type(operands, results)";
}

def FuncRefOp
  : HighLevel_Op< "funcref" >
  , Arguments<(ins FlatSymbolRefAttr:$function)>
  , Results<(outs AnyType:$result)>
{
  let summary = "VAST function reference declaration";
  let description = [{ VAST function reference declaration }];

  let assemblyFormat = "$function attr-dict `:` type($result)";
}

def GlobalRefOp
  : HighLevel_Op< "globref" >
  , Arguments<(ins StrAttr:$global)>
  , Results<(outs AnyType:$result)>
{
  let summary = "VAST global variable reference declaration";
  let description = [{ VAST global variable reference declaration }];

  let assemblyFormat = "$global attr-dict `:` type($result)";
}

def EnumRefOp
  : HighLevel_Op< "enumref" >
  , Arguments<(ins StrAttr:$value)>
  , Results<(outs AnyType:$result)>
{
  let summary = "VAST variable reference declaration";
  let description = [{ VAST variable reference declaration }];

  let assemblyFormat = "$value attr-dict `:` type($result)";
}

def ConstantOp
  : HighLevel_Op< "const", [ConstantLike, AllTypesMatch< ["value", "result"] >] >
  , Arguments<(ins TypedAttrInterface:$value)>
  , Results<(outs AnyType:$result)>
{
  let summary = "VAST value constant";
  let description = [{ VAST value constant }];

  let builders = [
    OpBuilder<(ins "VoidType":$type), [{
      build($_builder, $_state, type, core::VoidAttr::get(type.getContext(), type));
    }]>,
    OpBuilder<(ins "Type":$type, "bool":$value), [{
      build($_builder, $_state, type, core::BooleanAttr::get(type, value));
    }]>,
    OpBuilder<(ins "Type":$type, "llvm::APSInt":$value), [{
      build($_builder, $_state, type, core::IntegerAttr::get(type, value));
    }]>,
    OpBuilder<(ins "Type":$type, "llvm::APFloat":$value), [{
      build($_builder, $_state, type, core::FloatAttr::get(type, value));
    }]>,
    OpBuilder<(ins "Type":$type, "llvm::Twine":$value), [{
      build($_builder, $_state, type, mlir::StringAttr::get(value, type));
    }]>
  ];

  let hasFolder = 1;

  let assemblyFormat = "$value attr-dict";
}

def UnreachableOp : HighLevel_Op<"unreachable", [Terminator]> {
  let summary = "VAST unreachable operation";
  let description = [{ VAST unreachable operation }];
  let assemblyFormat = "attr-dict";
}

class CastKindAttr< string name, int val > : I64EnumAttrCase< name, val > {}

class CastKindList< string name, string summary, list< CastKindAttr > cases >
  : I64EnumAttr< name, summary, cases > {}

def Dependent : CastKindAttr<"Dependent", 0>;
def BitCast : CastKindAttr<"BitCast", 1>;
def LValueBitCast : CastKindAttr<"LValueBitCast", 2>;
def LValueToRValueBitCast : CastKindAttr<"LValueToRValueBitCast", 3>;
def LValueToRValue : CastKindAttr<"LValueToRValue", 4>;

def NoOp : CastKindAttr<"NoOp", 5>;

def BaseToDerived : CastKindAttr<"BaseToDerived", 6>;
def DerivedToBase : CastKindAttr<"DerivedToBase", 7>;
def UncheckedDerivedToBase : CastKindAttr<"UncheckedDerivedToBase", 8>;
def Dynamic : CastKindAttr<"Dynamic", 9>;
def ToUnion : CastKindAttr<"ToUnion", 10>;

def ArrayToPointerDecay : CastKindAttr<"ArrayToPointerDecay", 11>;
def FunctionToPointerDecay : CastKindAttr<"FunctionToPointerDecay", 12>;
def NullToPointer : CastKindAttr<"NullToPointer", 13>;
def NullToMemberPointer : CastKindAttr<"NullToMemberPointer", 14>;
def BaseToDerivedMemberPointer : CastKindAttr<"BaseToDerivedMemberPointer", 15>;
def DerivedToBaseMemberPointer : CastKindAttr<"DerivedToBaseMemberPointer", 16>;
def MemberPointerToBoolean : CastKindAttr<"MemberPointerToBoolean", 17>;
def ReinterpretMemberPointer : CastKindAttr<"ReinterpretMemberPointer", 18>;
def UserDefinedConversion : CastKindAttr<"UserDefinedConversion", 19>;
def ConstructorConversion : CastKindAttr<"ConstructorConversion", 20>;

def IntegralToPointer : CastKindAttr<"IntegralToPointer", 21>;
def PointerToIntegral : CastKindAttr<"PointerToIntegral", 22>;
def PointerToBoolean  : CastKindAttr<"PointerToBoolean", 23>;

def ToVoid : CastKindAttr<"ToVoid", 24>;

def VectorSplat : CastKindAttr<"VectorSplat", 25>;

def IntegralCast : CastKindAttr<"IntegralCast", 26>;
def IntegralToBoolean : CastKindAttr<"IntegralToBoolean", 27>;
def IntegralToFloating : CastKindAttr<"IntegralToFloating", 28>;
def FloatingToFixedPoint : CastKindAttr<"FloatingToFixedPoint", 29>;
def FixedPointToFloating : CastKindAttr<"FixedPointToFloating", 30>;
def FixedPointCast : CastKindAttr<"FixedPointCast", 31>;
def FixedPointToIntegral : CastKindAttr<"FixedPointToIntegral", 32>;
def IntegralToFixedPoint : CastKindAttr<"IntegralToFixedPoint", 33>;
def FixedPointToBoolean : CastKindAttr<"FixedPointToBoolean", 34>;
def FloatingToIntegral : CastKindAttr<"FloatingToIntegral", 35>;
def FloatingToBoolean : CastKindAttr<"FloatingToBoolean", 36>;
def BooleanToSignedIntegral : CastKindAttr<"BooleanToSignedIntegral", 37>;
def FloatingCast : CastKindAttr<"FloatingCast", 38>;

def CPointerToObjCPointerCast : CastKindAttr<"CPointerToObjCPointerCast", 39>;
def BlockPointerToObjCPointerCast : CastKindAttr<"BlockPointerToObjCPointerCast", 40>;
def AnyPointerToBlockPointerCast : CastKindAttr<"AnyPointerToBlockPointerCast", 41>;
def ObjCObjectLValueCast : CastKindAttr<"ObjCObjectLValueCast", 42>;

def FloatingRealToComplex : CastKindAttr<"FloatingRealToComplex", 43>;
def FloatingComplexToReal : CastKindAttr<"FloatingComplexToReal", 44>;
def FloatingComplexToBoolean : CastKindAttr<"FloatingComplexToBoolean", 45>;
def FloatingComplexCast : CastKindAttr<"FloatingComplexCast", 46>;
def FloatingComplexToIntegralComplex : CastKindAttr<"FloatingComplexToIntegralComplex", 47>;
def IntegralRealToComplex : CastKindAttr<"IntegralRealToComplex", 48>;
def IntegralComplexToReal : CastKindAttr<"IntegralComplexToReal", 49>;
def IntegralComplexToBoolean : CastKindAttr<"IntegralComplexToBoolean", 50>;
def IntegralComplexCast : CastKindAttr<"IntegralComplexCast", 51>;
def IntegralComplexToFloatingComplex : CastKindAttr<"IntegralComplexToFloatingComplex", 52>;

def ARCProduceObject : CastKindAttr<"ARCProduceObject", 53>;
def ARCConsumeObject : CastKindAttr<"ARCConsumeObject", 54>;
def ARCReclaimReturnedObject : CastKindAttr<"ARCReclaimReturnedObject", 55>;
def ARCExtendBlockObject : CastKindAttr<"ARCExtendBlockObject", 56>;

def AtomicToNonAtomic : CastKindAttr<"AtomicToNonAtomic", 57>;
def NonAtomicToAtomic : CastKindAttr<"NonAtomicToAtomic", 58>;

def CopyAndAutoreleaseBlockObject : CastKindAttr<"CopyAndAutoreleaseBlockObject", 59>;
def BuiltinFnToFnPtr : CastKindAttr<"BuiltinFnToFnPtr", 60>;

def ZeroToOCLOpaqueType : CastKindAttr<"ZeroToOCLOpaqueType", 61>;
def AddressSpaceConversion : CastKindAttr<"AddressSpaceConversion", 62>;
def IntToOCLSampler : CastKindAttr<"IntToOCLSampler", 63>;

def MatrixCast : CastKindAttr<"MatrixCast", 64>;

let cppNamespace = "::vast::hl" in
def CastKind : CastKindList< "CastKind", "cast kind", [
  Dependent,
  BitCast,
  LValueBitCast,
  LValueToRValueBitCast,
  LValueToRValue,

  NoOp,

  BaseToDerived,
  DerivedToBase,
  UncheckedDerivedToBase,
  Dynamic,
  ToUnion,

  ArrayToPointerDecay,
  FunctionToPointerDecay,
  NullToPointer,
  NullToMemberPointer,
  BaseToDerivedMemberPointer,
  DerivedToBaseMemberPointer,
  MemberPointerToBoolean,
  ReinterpretMemberPointer,
  UserDefinedConversion,
  ConstructorConversion,

  IntegralToPointer,
  PointerToIntegral,
  PointerToBoolean ,

  ToVoid,

  VectorSplat,

  IntegralCast,
  IntegralToBoolean,
  IntegralToFloating,
  FloatingToFixedPoint,
  FixedPointToFloating,
  FixedPointCast,
  FixedPointToIntegral,
  IntegralToFixedPoint,
  FixedPointToBoolean,
  FloatingToIntegral,
  FloatingToBoolean,
  BooleanToSignedIntegral,
  FloatingCast,

  CPointerToObjCPointerCast,
  BlockPointerToObjCPointerCast,
  AnyPointerToBlockPointerCast,
  ObjCObjectLValueCast,

  FloatingRealToComplex,
  FloatingComplexToReal,
  FloatingComplexToBoolean,
  FloatingComplexCast,
  FloatingComplexToIntegralComplex,
  IntegralRealToComplex,
  IntegralComplexToReal,
  IntegralComplexToBoolean,
  IntegralComplexCast,
  IntegralComplexToFloatingComplex,

  ARCProduceObject,
  ARCConsumeObject,
  ARCReclaimReturnedObject,
  ARCExtendBlockObject,

  AtomicToNonAtomic,
  NonAtomicToAtomic,

  CopyAndAutoreleaseBlockObject,
  BuiltinFnToFnPtr,

  ZeroToOCLOpaqueType,
  AddressSpaceConversion,
  IntToOCLSampler,

  MatrixCast
] >;

class CastOp< string mnemonic, list< Trait > traits = [] >
    : HighLevel_Op< mnemonic, traits >
    , Arguments< (ins AnyType:$value, CastKind:$kind) >
    , Results< (outs AnyType:$result) >
{
    let summary = "VAST cast operation";
    let description = [{ VAST cast operation }];

    let assemblyFormat = "$value $kind attr-dict `:` type($value) `->` type($result)";
}

def ImplicitCastOp   : CastOp< "implicit_cast" >;
def CStyleCastOp     : CastOp< "cstyle_cast" >;
def BuiltinBitCastOp : CastOp< "builtin_bitcast" >;

class IsPointerCompatible< string arg > : PredTypeTrait< "type can be added/subtracted to a pointer",
   Or < [And< [IsIntegral< arg >.predicate,
           Neg< IsShort< arg >.predicate >,
           Neg< IsChar< arg >.predicate >,
           Neg< IsBool< arg >.predicate >
              ] >,
         IsBuiltinType< arg >.predicate
        ] >
>;

class IsPointerAdditive< string lhs, string rhs, string res > : PredOpTrait< "is an additive operation on pointers",
    And< [Or< [And< [IsPointer< lhs >.predicate, IsPointerCompatible< rhs >.predicate] >,
               And< [IsPointerCompatible< lhs >.predicate, IsPointer< rhs >.predicate] >] >,
         IsPointer< res >.predicate
    ] >
>;

class PtrDiffLike< string lhs, string rhs, string res > : PredOpTrait< "is a subtraction of pointers",
    And< [IsPointer< lhs >.predicate, IsPointer< rhs >.predicate,
          Or< [ IsLong< res >.predicate, IsBuiltinType< res >.predicate] >
         ] >
>;

class IsAdditive< string lhs, string rhs, string res > : PredOpTrait< "is an additive operation (types match or are ptr and integral)", Or<[
        TypesMatchOrTypedef< [lhs, rhs, res] >.predicate,
        IsPointerAdditive< lhs, rhs, res >.predicate,
    ]>
>;

class IsSub< string lhs, string rhs, string res > : PredOpTrait< "is a subtraction operation (types match or are ptr and integral)",
    Or< [
        IsAdditive< lhs, rhs, res >.predicate, PtrDiffLike< lhs, rhs, res >.predicate
    ] >
>;

class ArithBinOp< string mnemonic, list< Trait > traits = [] >
    : HighLevel_Op< mnemonic, traits >
    , Arguments<(ins AnyType:$lhs, AnyType:$rhs)>
    , Results<(outs AnyType:$result)>
{
    let summary = "VAST arithmetic binary operation";
    let description = [{
        High-level arithmetic binary operation. This operation takes two operands
        and returns one result, each of these is required to be of the same
        type.

        The custom assembly form of the operation is as follows:

        %result = <op> %lhs, %rhs  : functional-type(operands, results)
    }];

    let skipDefaultBuilders = 1;
    let builders = [ OpBuilder< (ins "Type":$rtype, "Value":$lhs, "Value":$rhs), [{
        $_state.addOperands(lhs);
        $_state.addOperands(rhs);
        $_state.addTypes(rtype);
      }] >
    ];

    let assemblyFormat = [{ $lhs `,` $rhs attr-dict `:` functional-type(operands, results) }];
}

class StandardArithBinOp< string mnemonic, list< Trait > traits = [] >
    : ArithBinOp< mnemonic,
        !listconcat(traits, [TypesMatchOrTypedef< ["lhs", "rhs", "result"] >])
      >;


def AddIOp : ArithBinOp< "add", [Commutative, IsAdditive< "lhs", "rhs", "result" >] >;
def SubIOp : ArithBinOp< "sub", [IsSub< "lhs", "rhs", "result" >] >;

def AddFOp : StandardArithBinOp< "fadd" >;
def SubFOp : StandardArithBinOp< "fsub" >;
def MulIOp : StandardArithBinOp<  "mul", [Commutative] >;
def MulFOp : StandardArithBinOp< "fmul" >;
def DivSOp : StandardArithBinOp< "sdiv" >;
def DivUOp : StandardArithBinOp< "udiv" >;
def DivFOp : StandardArithBinOp< "fdiv" >;
def RemSOp : StandardArithBinOp< "srem" >;
def RemUOp : StandardArithBinOp< "urem" >;
def RemFOp : StandardArithBinOp< "frem" >;

def BinXorOp : StandardArithBinOp< "bin.xor" >;
def BinOrOp  : StandardArithBinOp<  "bin.or" >;
def BinAndOp : StandardArithBinOp< "bin.and" >;


class LogicBinOp< string mnemonic, list< Trait > traits = [] >
    : HighLevel_Op< mnemonic, traits >
    , Results<(outs AnyType:$result)>
{
    let summary = "VAST logical binary operation";
    let description = [{
        High-level logical binary operation. This operation takes two operands
        and returns one result, each of these is required to be of the same
        type.

        The custom assembly form of the operation is as follows:

        %result = <op> %lhs, %rhs  : type
    }];

    let regions = (region ValueRegion:$lhs, ValueRegion:$rhs);

    let skipDefaultBuilders = 1;
    let builders = [
        OpBuilder<(ins
            "Type":$type,
            "BuilderCallback":$lhsbuilder,
            "BuilderCallback":$rhsbuilder
        )>
    ];

    let assemblyFormat = [{ $lhs`,` $rhs attr-dict `:` type(results) }];
}

def BinLAndOp : LogicBinOp< "bin.land" >;
def BinLOrOp  : LogicBinOp<  "bin.lor" >;

def BinComma
  : HighLevel_Op< "bin.comma" >
  , Arguments< (ins AnyType:$lhs, AnyType:$rhs) >
  , Results< (outs AnyType:$result) >
{
    let summary = "VAST binary operation";

    let assemblyFormat = [{ $lhs `,` $rhs attr-dict `:` functional-type(operands, results) }];
}

class ShiftOp< string mnemonic, list< Trait > traits = [] >
    : HighLevel_Op< mnemonic, !listconcat(traits, [
        TypesMatchOrTypedef<["lhs", "result"]>
    ]) >
    , Arguments<(ins IntegerLikeType:$lhs, IntegerLikeType:$rhs)>
    , Results<(outs IntegerLikeType:$result)>
{
    let summary = "VAST binary shift operation";
    let description = [{
        High-level binary shift operation. This operation takes two operands
        and returns one result.

        The custom assembly form of the operation is as follows:

        %result = <op> %lhs, %rhs  : functional-type(operands, results)
    }];

    let assemblyFormat = [{ $lhs `,` $rhs attr-dict `:` functional-type(operands, results) }];
}

def BinShlOp : ShiftOp<"bin.shl" >;
def BinLShrOp : ShiftOp<"bin.lshr" >;
def BinAShrOp : ShiftOp<"bin.ashr" >;

class IsLValuePointer< string arg > : PredOpTrait< "lvalue is of an integer type",
    CPred< "$" # arg # ".getType().cast< LValueType >().getElementType().isa< hl::PointerType >()" >
>;

class IsCompoundPtrAdditive< string lhs, string rhs, string res > : PredOpTrait< "is an additive operation on pointers",
    And< [IsLValuePointer< lhs >.predicate,
          IsInteger< rhs>.predicate,
          IsPointer< res >.predicate ] >
>;

class IsCompoundAdditive< string lhs, string rhs, string res > : PredOpTrait< "is an additive operation (types match or are ptr and integral)",
    Or<[
        IsCompoundPtrAdditive< lhs, rhs, res >.predicate,
        CPred< "$" # lhs # ".getType().cast< LValueType >().getElementType() == $" # res # ".getType()" >,
        ContainsTypedef< [lhs, rhs] >.predicate
    ]>
>;

class CompoundAssignOpTemplate< string mnemonic, TypeConstraint Type, list< Trait > traits = [] >
    : HighLevel_Op< mnemonic, !listconcat(traits, [
        TypesMatchWith<
          "underlying destination type match result type",
          "dst", "result",
          "$_self.cast< ElementTypeInterface >().getElementType()"
        >
    ])>
    // FIXME: add argument constraints and infer return type
    , Arguments<(ins LValueOrType< Type >:$src, AnyType:$dst)>
    , Results<(outs Type:$result)>
{
    let summary = "VAST compound assign operation";
    let description = [{
        A compound assign operation represents an assignment operation joined
        with an arithmetic operation. It requires the same types for both source
        and destination arguments.

        The custom assembly form of the operation is as follows:

        %result = <assign.op> src to dst : functional-type(operands, results)

        It represents C compound assignment statement:

        dst =<op> src;
    }];

    let skipDefaultBuilders = 1;
    let builders = [ OpBuilder<(ins "Value":$dst, "Value":$src), [{
        $_state.addOperands(src);
        $_state.addOperands(dst);
        auto type = dst.getType().cast< LValueType >();
        $_state.addTypes(type.getElementType());
      }]>
    ];

    let assemblyFormat = [{
      $src `to` $dst attr-dict `:` type(operands) `->` type(results)
    }];
}

class CompoundAssignOp< string mnemonic, list< Trait > traits = [] >
    : CompoundAssignOpTemplate< mnemonic, AnyType, traits > {}

def AssignOp     : CompoundAssignOp< "assign" >;
def AddIAssignOp : CompoundAssignOp< "assign.add", [IsCompoundAdditive< "dst", "src", "result" >] >;
def AddFAssignOp : CompoundAssignOp< "assign.fadd" >;
def SubIAssignOp : CompoundAssignOp< "assign.sub", [IsCompoundAdditive< "dst", "src", "result" >] >;
def SubFAssignOp : CompoundAssignOp< "assign.fsub" >;
def MulIAssignOp : CompoundAssignOp< "assign.mul"  >;
def MulFAssignOp : CompoundAssignOp< "assign.fmul" >;
def DivSAssignOp : CompoundAssignOp< "assign.sdiv" >;
def DivUAssignOp : CompoundAssignOp< "assign.udiv" >;
def DivFAssignOp : CompoundAssignOp< "assign.fdiv" >;
def RemSAssignOp : CompoundAssignOp< "assign.srem" >;
def RemUAssignOp : CompoundAssignOp< "assign.urem" >;
def RemFAssignOp : CompoundAssignOp< "assign.frem" >;

def BinAndAssignOp : CompoundAssignOp< "assign.bin.and" >;
def BinOrAssignOp  : CompoundAssignOp< "assign.bin.or"  >;
def BinXorAssignOp : CompoundAssignOp< "assign.bin.xor" >;

class ShiftAssignOp< string mnemonic, list< Trait > traits = [] >
    : CompoundAssignOpTemplate< mnemonic, IntegerLikeType, traits > {}

def BinShlAssignOp : ShiftAssignOp< "assign.bin.shl" >;

def BinLShrAssignOp : ShiftAssignOp< "assign.bin.lshr" >;
def BinAShrAssignOp : ShiftAssignOp< "assign.bin.ashr" >;

class PredicateAttr< string name, int val > : I64EnumAttrCase< name, val > {}

class PredicateList< string name, string summary, list<PredicateAttr> cases >
  : I64EnumAttr< name, summary, cases > {}

def ICmpPredicateEQ  : PredicateAttr< "eq", 0>;
def ICmpPredicateNE  : PredicateAttr< "ne", 1>;
def ICmpPredicateSLT : PredicateAttr<"slt", 2>;
def ICmpPredicateSLE : PredicateAttr<"sle", 3>;
def ICmpPredicateSGT : PredicateAttr<"sgt", 4>;
def ICmpPredicateSGE : PredicateAttr<"sge", 5>;
def ICmpPredicateULT : PredicateAttr<"ult", 6>;
def ICmpPredicateULE : PredicateAttr<"ule", 7>;
def ICmpPredicateUGT : PredicateAttr<"ugt", 8>;
def ICmpPredicateUGE : PredicateAttr<"uge", 9>;

let cppNamespace = "::vast::hl" in
def Predicate : PredicateList< "Predicate", "comparison predicate", [
  ICmpPredicateEQ,  ICmpPredicateNE,  ICmpPredicateSLT, ICmpPredicateSLE,
  ICmpPredicateSGT, ICmpPredicateSGE, ICmpPredicateULT, ICmpPredicateULE,
  ICmpPredicateUGT, ICmpPredicateUGE
] >;

class IsPointerCmp< string lhs, string rhs > : PredOpTrait< "is an additive operation on pointers",
    And< [IsPointer< lhs >.predicate, IsPointer< rhs >.predicate] >
>;

class IsIntegralCmp< string lhs, string rhs > : PredOpTrait< "is an additive operation on pointers",
    And< [IsIntegral< lhs >.predicate, IsIntegral< rhs >.predicate] >
>;

class IsCmp< string lhs, string rhs > : PredOpTrait< "is an additive operation (types match or are ptr and integral)",
    Or<[
        IsIntegralCmp< lhs, rhs >.predicate,
        IsPointerCmp< lhs, rhs >.predicate
    ]>
>;

def CmpOp
  : HighLevel_Op< "cmp" >
  , Arguments<(ins Predicate:$predicate, AnyType:$lhs, AnyType:$rhs)>
  , Results<(outs IntOrBoolType:$result)>
  , IsCmp< "lhs", "rhs" >
{
  let summary = "VAST comparison operation";
  let description = [{ VAST comparison operation }];

  let assemblyFormat = "$predicate $lhs `,` $rhs  attr-dict `:` type(operands) `->` type($result)";
}

class FPredicateAttr< string name, int val > : I64EnumAttrCase< name, val > {}

class FPredicateList< string name, string summary, list<FPredicateAttr> cases >
  : I64EnumAttr< name, summary, cases > {}

def FPredicateFalse : FPredicateAttr<"ffalse", 0>;
def FPredicateOEQ   : FPredicateAttr<"oeq",    1>;
def FPredicateOGT   : FPredicateAttr<"ogt",    2>;
def FPredicateOGE   : FPredicateAttr<"oge",    3>;
def FPredicateOLT   : FPredicateAttr<"olt",    4>;
def FPredicateOLE   : FPredicateAttr<"ole",    5>;
def FPredicateONE   : FPredicateAttr<"one",    6>;
def FPredicateORD   : FPredicateAttr<"ord",    7>;
def FPredicateUNO   : FPredicateAttr<"uno",    8>;
def FPredicateUEQ   : FPredicateAttr<"ueq",    9>;
def FPredicateUGT   : FPredicateAttr<"ugt",   10>;
def FPredicateUGE   : FPredicateAttr<"uge",   11>;
def FPredicateULT   : FPredicateAttr<"ult",   12>;
def FPredicateULE   : FPredicateAttr<"ule",   13>;
def FPredicateUNE   : FPredicateAttr<"une",   14>;
def FPredicateTrue  : FPredicateAttr<"ftrue", 15>;

let cppNamespace = "::vast::hl" in
def FPredicate : FPredicateList< "FPredicate", "floating point comparison predicate", [
  FPredicateFalse, FPredicateOEQ, FPredicateOGT, FPredicateOGE,
  FPredicateOLT, FPredicateOLE, FPredicateONE, FPredicateORD,
  FPredicateUNO, FPredicateUEQ, FPredicateUGT, FPredicateUGE,
  FPredicateULT, FPredicateULE, FPredicateUNE, FPredicateTrue
] >;

def FCmpOp
  : HighLevel_Op< "fcmp" >
  , Arguments<(ins FPredicate:$predicate, FloatLikeType:$lhs, FloatLikeType:$rhs)>
  , Results<(outs IntOrBoolType:$result)>
{
  let summary = "VAST flaoting point comparison operation";
  let description = [{ VAST floating point comparison operation }];

  let assemblyFormat = "$predicate $lhs `,` $rhs  attr-dict `:` type(operands) `->` type($result)";
}



class UnInplaceOp< string mnemonic, list< Trait > traits = [] >
    : HighLevel_Op< mnemonic, !listconcat(traits, [
        TypesMatchWith<
          "underlying argument type match result type",
          "arg", "result",
          "$_self.cast< LValueType >().getElementType()"
        >
      ]) >
    , Arguments<(ins LValueOf<AnyType>:$arg)>
    , Results<(outs AnyType:$result)>
{
    let summary = "VAST unary inplace operation";
    let description = [{
        Inplace high-level unary operation changes its single argument in place.
        It does not produce a new value.

        The custom assembly form of the operation is as follows:

        %result = <op> %arg : type
    }];

    let assemblyFormat = [{ $arg attr-dict `:` type($arg) `->` type($result) }];
}

def PostIncOp : UnInplaceOp< "post.inc" >;
def PostDecOp : UnInplaceOp< "post.dec" >;
def PreIncOp  : UnInplaceOp<  "pre.inc" >;
def PreDecOp  : UnInplaceOp<  "pre.dec" >;

class TypePreservingUnOp< string mnemonic, list< Trait > traits = [] >
    : HighLevel_Op< mnemonic, !listconcat(traits, [SameOperandsAndResultType]) >
    , Arguments<(ins AnyType:$arg)>
    , Results<(outs AnyType:$result)>
{
    let summary = "VAST unary type preserving operation";
    let description = [{
        Type preserving high-level unary operation assures that argument and
        result has the same type.

        The custom assembly form of the operation is as follows:

        %result = <op> %arg : type
    }];

    let assemblyFormat = [{ $arg attr-dict `:` type($result) }];
}

def PlusOp  : TypePreservingUnOp< "plus" >;
def MinusOp : TypePreservingUnOp< "minus" >;
def NotOp   : TypePreservingUnOp< "not" >;

class LogicalUnOp< string mnemonic, list< Trait > traits = [] >
    : HighLevel_Op< mnemonic, traits >
    , Arguments< (ins AnyType:$arg) >
    , Results< (outs IntOrBoolType:$res) >
{
    let summary = "VAST unary logical operation";
    let description = [{
        High-level unary logical operation assures that
        result has the right type.

        The custom assembly form of the operation is as follows:

        %result = <op> %arg : type -> ret_type
    }];

    let assemblyFormat = [{ $arg attr-dict `:` type($arg) `->` type($res) }];
}

def LNotOp  : LogicalUnOp< "lnot", [] >;

def AddressOf
  : HighLevel_Op< "addressof", [] >
  // TODO(Heno): parameter constraints
  , Arguments<(ins LValueOf<AnyType>:$value)>
  , Results<(outs AnyType:$result)>
{
    let summary = "VAST addressof operation";
    let description = [{ VAST addressof operation }];

    let assemblyFormat = [{ $value attr-dict `:` type($value) `->` type($result) }];
}

def Deref
  : HighLevel_Op< "deref" >
  // TODO(Heno): check dereferencable
  , Arguments<(ins AnyType:$addr)>
  , Results<(outs LValueOf<AnyType>:$result)>
{
    let summary = "VAST deref operation";
    let description = [{ VAST deref operation }];

    let assemblyFormat = [{ $addr attr-dict `:` type($addr) `->` type($result) }];
}

def AddrLabelExpr
  : HighLevel_Op< "labeladdr" >
  , Arguments<(ins LabelType:$label)>
  , Results<(outs LValueOf<PointerLikeType>:$result)>
{
    let summary = "VAST address of label extension";
    let description = [{ VAST address of label extension }];

    let assemblyFormat = [{ $label attr-dict `:` type($result) }];
}

def InitListExpr
  : HighLevel_Op< "initlist" >
  , Arguments<(ins Variadic<AnyType>:$elements)>
  , Results<(outs Variadic<AnyType>)>
{
  let summary = "VAST initializer list expression";
  let description = [{ VAST initializer list expression }];

  let assemblyFormat = "$elements attr-dict `:` functional-type($elements, results)";
}

def SubscriptOp
  : HighLevel_Op< "subscript" >
  , Arguments<(ins
      LValueOrType<SubscriptableType>:$array,
      IntegerLikeType:$index)>
  , Results<(outs LValueOf<AnyType>:$result)>
{
  let summary = "VAST array subscript operator";
  let description = [{ VAST array subscript operator }];

  let assemblyFormat = [{
    $array `at` ` ` `[` $index `:` type($index) `]` attr-dict
      `:` type($array) `->` type($result)
  }];
}

class TypeTraitOp< string mnemonic, list< Trait > traits = [] >
  : HighLevel_Op< mnemonic, traits >
  , Arguments<(ins TypeAttr:$arg)>
  , Results<(outs IntegerLikeType:$result)>
{
  let assemblyFormat = [{ $arg attr-dict `->` type($result) }];
}

def SizeOfTypeOp : TypeTraitOp< "sizeof.type" > {
  let summary = "VAST type sizeof operator";
  let description = [{ VAST type sizeof operator }];
}

def AlignOfTypeOp : TypeTraitOp< "alignof.type" > {
  let summary = "VAST type alignof operator";
  let description = [{ VAST type alignof operator }];
}

class ExprTraitOp< string mnemonic, list< Trait > traits = [] >
  : HighLevel_Op< mnemonic, traits >
  , Results<(outs IntegerLikeType:$result)>
{
  let regions = (region SizedRegion<1>:$expr);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "Type":$type, "BuilderCallback":$exprBuilder)>
  ];

  let assemblyFormat = [{ attr-dict `->` type($result) $expr }];
}

def SizeOfExprOp : ExprTraitOp< "sizeof.expr" > {
  let summary = "VAST expr sizeof operator";
  let description = [{ VAST expr sizeof operator }];
}

def AlignOfExprOp : ExprTraitOp< "alignof.expr" > {
  let summary = "VAST expr alignof operator";
  let description = [{ VAST expr alignof operator }];
}

def StmtExprOp
  : HighLevel_Op< "stmt.expr", [SingleBlock, DeclareOpInterfaceMethods<RegionKindInterface>] >
  , Results<(outs AnyType:$result)>
{
  let summary = "VAST statement expression";
  let description = [{ VAST statement expression }];

  let regions = (region SizedRegion<1>:$substmt);

  let skipDefaultBuilders = 1;
  let builders = [ OpBuilder<(ins "Type":$rty, "std::unique_ptr< Region > &&":$region)> ];

  let assemblyFormat = [{
     attr-dict `:` type($result) $substmt
  }];
}

class IdentKindAttr< string name, int val > : I64EnumAttrCase< name, val > {}

class IdentKindList< string name, string summary, list< IdentKindAttr > cases >
  : I64EnumAttr< name, summary, cases > {}

def Func : IdentKindAttr<"Func", 0>;
def Function : IdentKindAttr<"Function", 1>;
def LFunction : IdentKindAttr<"LFunction", 2>;
def FuncDName : IdentKindAttr<"FuncDName", 3>;
def FuncSig : IdentKindAttr<"FuncSig", 4>;
def LFuncSig : IdentKindAttr<"LFuncSig", 5>;
def PrettyFunction : IdentKindAttr<"PrettyFunction", 6>;
def PrettyFunctionNoVirtual : IdentKindAttr<"PrettyFunctionNoVirtual", 7>;

let cppNamespace = "::vast::hl" in
def IdentKind : IdentKindList< "IdentKind", "ident kind", [
    Func,
    Function,
    LFunction,
    FuncDName,
    FuncSig,
    LFuncSig,
    PrettyFunction,
    PrettyFunctionNoVirtual
] >;

def PredefinedExpr
  : HighLevel_Op< "predefined.expr" >
    , Arguments<(ins AnyType:$value, IdentKind:$kind)>
    , Results<(outs AnyType:$result)>
{
    let summary = "VAT predefined expr ( such as __func__ )";
    let description = [{ VAT predefined expr ( such as __func__ ) }];

    let assemblyFormat = "$value $kind attr-dict `:` type($value) `->` type($result)";
}

def ExtensionOp
  : HighLevel_Op< "gnu.extension" >
    , Arguments<(ins AnyType:$value)>
    , Results<(outs AnyType:$result)>
{
    let summary = "VAST extension (__extension__) keyword";
    let description = [{ VAST op corresponding to GNU __extension__ keyword. }];

    let assemblyFormat = "$value attr-dict `:` type($value) `->` type($result)";
}

def HighLevel_EmptyDeclOp
  : HighLevel_Op< "empty.decl", [] >
{
  let assemblyFormat = [{ attr-dict }];
}

def AsmOp
  : HighLevel_Op< "asm", [AttrSizedOperandSegments] >
  , Arguments<(ins
      StrAttr:$asm_template,
      UnitAttr:$is_volatile,
      UnitAttr:$has_goto,
      // UnitAttr:$has_inline, // Clang doesn't have this qualifier? issue #454
      Variadic< AnyType >:$asm_outputs,
      Variadic< AnyType >:$asm_inputs,
      OptionalAttr< ArrayAttr >:$output_names,
      OptionalAttr< ArrayAttr >:$input_names,
      OptionalAttr< ArrayAttr >:$output_constraints,
      OptionalAttr< ArrayAttr >:$input_constraints,
      OptionalAttr< ArrayAttr >:$clobbers,
      Variadic< AnyType >:$labels
    )>
{
  let summary = "VAST operation for inline assembly";
  let description = [{ VAST operation mirroring the GCCAsmStmt in clang AST. It prints
                       a name for every operand (either its id or user-supplied string).}];

  let skipDefaultBuilders = 1;

  let builders = [
  OpBuilder< (ins
      "mlir::StringAttr":$asm_template,
      "bool":$is_volatile,
      "bool":$has_goto,
      CArg< "llvm::ArrayRef< mlir::Value >", "{}" >:$outs,
      CArg< "llvm::ArrayRef< mlir::Value >", "{}" >:$ins,
      CArg< "mlir::ArrayAttr", "{}" >:$out_names,
      CArg< "mlir::ArrayAttr", "{}" >:$in_names,
      CArg< "mlir::ArrayAttr", "{}" >:$out_constraints,
      CArg< "mlir::ArrayAttr", "{}" >:$in_constraints,
      CArg< "mlir::ArrayAttr", "{}" >:$clobbers,
      CArg< "llvm::ArrayRef< mlir::Value >", "{}" >:$labels
    ) >
  ];

  let assemblyFormat = [{attr-dict $asm_template `(`($output_names $asm_outputs^ `:` $output_constraints)? `)` `(` (`ins` `:`$input_names $asm_inputs^ `:` $input_constraints)? `)` `(`( $clobbers^)?`)` `(`( $labels^)?`)` `:` functional-type(operands, results)}];
}

def VAArgExpr
  : HighLevel_Op< "va_arg_expr" >
  , Arguments< (ins AnyType: $arg_list) >
  , Results< (outs AnyType: $result) >
{
  let summary = "Fetch next variadic element.";

  let assemblyFormat = "$arg_list `:` functional-type(operands, $result) attr-dict";
}

#endif // VAST_DIALECT_HIGHLEVEL_IR_HIGHLEVELOPS

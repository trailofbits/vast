// Copyright (c) 2021-present, Trail of Bits, Inc.

#ifndef VAST_DIALECT_HIGHLEVEL_IR_HIGHLEVELOPS
#define VAST_DIALECT_HIGHLEVEL_IR_HIGHLEVELOPS

include "mlir/IR/OpBase.td"

include "mlir/IR/FunctionInterfaces.td"

include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/CastInterfaces.td"
include "vast/Interfaces/SymbolInterface.td"

include "vast/Dialect/HighLevel/HighLevelLinkage.td"


def TranslationUnitOp
  : HighLevel_Op< "translation_unit", [NoTerminator, SymbolTable, IsolatedFromAbove] >
{
  let summary = "VAST translation unit";
  let description = [{ VAST tranaslation unit }];

  let regions = (region AnyRegion:$body);

  let assemblyFormat = [{ $body attr-dict }];
}

def ScopeOp : HighLevel_Op< "scope", [NoTerminator] >
{
  let summary = "VAST scope declaration";
  let description = [{
    Scope operation servers to represent explicitly high-level code scope.
    Other control flow operations represent scopes implicitly.  It is a
    single-region operation.
  }];

  let regions = (region AnyRegion:$body);

  let assemblyFormat = [{ $body attr-dict }];
}

def FuncOp : HighLevel_Op< "func", [
  AutomaticAllocationScope, CallableOpInterface, FunctionOpInterface,
  IsolatedFromAbove, Symbol
] > {
  let summary = "VAST high-level function definintion or declaration";

  let description = [{
    Inspired by `cir::FuncOp` and `mlir::func::FuncOp`:
    > Operations within the function cannot implicitly capture values defined
    > outside of the function, i.e. Functions are `IsolatedFromAbove`. All
    > external references must use function arguments or attributes that establish
    > a symbolic connection (e.g. symbols referenced by name via a string
    > attribute like SymbolRefAttr). An external function declaration (used when
    > referring to a function declared in some other module) has no body.

    The function linkage information is specified by `linkage`, as defined by
    `GlobalLinkageKind` attribute.
  }];

  let arguments =(ins
    SymbolNameAttr:$sym_name,
    TypeAttrOf<FunctionType>:$function_type,
    DefaultValuedAttr<GlobalLinkageKind, "GlobalLinkageKind::ExternalLinkage">:$linkage,
    OptionalAttr<StrAttr>:$sym_visibility
  );

  let regions = (region AnyRegion:$body);
  let skipDefaultBuilders = 1;

  let builders = [OpBuilder<(ins
    "llvm::StringRef":$name,
    "mlir::FunctionType":$type,
    CArg< "GlobalLinkageKind", "GlobalLinkageKind::ExternalLinkage" >:$linkage,
    CArg< "llvm::ArrayRef<mlir::NamedAttribute>", "{}" >:$attrs,
    CArg< "llvm::ArrayRef<mlir::DictionaryAttr>", "{}" >:$arg_attrs,
    CArg< "BuilderCallback", "std::nullopt" >:$body), [{
      Builder::InsertionGuard guard($_builder);
      detail::build_region($_builder, $_state, body);

      $_state.addAttribute(
        mlir::SymbolTable::getSymbolAttrName(), $_builder.getStringAttr(name)
      );
      $_state.addAttribute(getTypeAttrName(), mlir::TypeAttr::get(type));
      $_state.addAttribute(
        "linkage", GlobalLinkageKindAttr::get($_builder.getContext(), linkage)
      );
      $_state.attributes.append(attrs.begin(), attrs.end());

      if (arg_attrs.empty())
        return;

      mlir::function_interface_impl::addArgAndResultAttrs(
        $_builder, $_state, arg_attrs, /* results */ llvm::None
      );
    }] >
  ];

  let extraClassDeclaration = [{
    //===------------------------------------------------------------------===//
    // CallableOpInterface
    //===------------------------------------------------------------------===//

    /// Returns the region on the current operation that is callable. This may
    /// return null in the case of an external callable object, e.g. an external
    /// function.
    ::mlir::Region *getCallableRegion() {
      return isExternal() ? nullptr : &getBody();
    }

    /// Returns the results types that the callable region produces when
    /// executed.
    llvm::ArrayRef<Type> getCallableResults() {
      return getFunctionType().getResults();
    }

    //===------------------------------------------------------------------===//
    // FunctionOpInterface Methods
    //===------------------------------------------------------------------===//

    /// Returns the argument types of this function.
    llvm::ArrayRef<Type> getArgumentTypes() { return getFunctionType().getInputs(); }

    /// Returns the result types of this function.
    llvm::ArrayRef<Type> getResultTypes() { return getFunctionType().getResults(); }

    //===------------------------------------------------------------------===//
    // SymbolOpInterface Methods
    //===------------------------------------------------------------------===//

    bool isDeclaration() { return isExternal(); }
  }];

  let assemblyFormat = [{
    $linkage $sym_name custom< FunctionSignaruteAndBody >($function_type, attr-dict, $body)
  }];

  let hasVerifier = 1;
}

def TypeDeclOp
  : HighLevel_Op< "type", [NoSideEffect, VastSymbol] >
  , Arguments<(ins StrAttr:$name)>
{
  let summary = "VAST type declaration";
  let description = [{ VAST type declaration }];

  let assemblyFormat = [{ $name attr-dict }];
}

def TypeDefOp
  : HighLevel_Op< "typedef", [NoSideEffect, VastSymbol] >
  , Arguments<(ins StrAttr:$name, TypeAttr:$type)>
{
  let summary = "VAST typedef operation";
  let description = [{
    Typedef operation servers to declare named types.
    It creates a new type symbol in the current scope to
    be referenced as NamedType later.
  }];

  let assemblyFormat = [{ $name attr-dict `:` $type }];
}

def EnumConstantOp
  : HighLevel_Op< "enum.const", [NoSideEffect] >
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
    hl.enum.const "G" = #hl.integer<12> : !hl.int init  {
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
      CArg<"BuilderCallback", "std::nullopt">:$init
    ), [{
      $_state.addAttribute("name", $_builder.getStringAttr(name));
      $_state.addAttribute("value", IntegerAttr::get(type, value));

      Builder::InsertionGuard guard($_builder);
      detail::build_region($_builder, $_state, init);
    }]>,
  ];

  let assemblyFormat = [{ $name `=` $value attr-dict (`init` $init^)? }];
}

def EnumRegion : Region< HasOneBlock, "enum constants region" >;

def EnumDeclOp
  : HighLevel_Op< "enum", [NoSideEffect, NoTerminator, VastSymbol] >
  // TODO(Heno): check integer like type attr
  , Arguments<(ins StrAttr:$name, TypeAttr:$type)>
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
    )>
  ];

  let assemblyFormat = [{ $name attr-dict `:` $type $constants }];
}

class RecordLikeDeclOp< string mnemonic, list< Trait > traits = [] >
    : HighLevel_Op< mnemonic, !listconcat(traits, [NoSideEffect, NoTerminator, VastSymbol]) >
    , Arguments<(ins StrAttr:$name)>
{
  // TODO(Heno): add region constraints
  let regions = (region AnyRegion:$fields);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins
      "llvm::StringRef":$name,
       CArg<"BuilderCallback", "std::nullopt">:$fields
    )>
  ];

  let assemblyFormat = [{ $name attr-dict `:` $fields }];
}

def StructDeclOp : RecordLikeDeclOp<"struct"> {
  let summary = "VAST struct declaration";
  let description = [{ VAST struct declaration }];
}

def UnionDeclOp : RecordLikeDeclOp<"union"> {
  let summary = "VAST record declaration";
  let description = [{ VAST record declaration }];
}

def FieldDeclOp
  : HighLevel_Op< "field", [NoSideEffect, VastSymbol] >
  , Arguments<(ins StrAttr:$name, TypeAttr:$type, OptionalAttr<I32Attr>:$bits)>
{
  let summary = "VAST record field declaration";
  let description = [{ VAST record field declaration }];

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
  , Results<(outs AnyType:$result)>
{
  let summary = "VAST call operation";
  let description = [{ VAST call operation }];

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder< (ins "FuncOp":$callee, CArg<"mlir::ValueRange", "{}">:$operands ), [{
      $_state.addOperands(operands);
      $_state.addAttribute("callee", mlir::SymbolRefAttr::get($_builder.getContext(), callee.getName()));
      $_state.addTypes(callee.getFunctionType().getResults());
    }]>
  ];

  let assemblyFormat = [{
    $callee `(` $argOperands `)` attr-dict `:` functional-type( $argOperands, $result )
  }];
}

def IndirectCallOp
  : HighLevel_Op< "indirect_call", [
    DeclareOpInterfaceMethods<CallOpInterface>,
  ]>
  , Arguments<(ins
      // FIXME: LValueOrType<PointerLikeType>
      AnyType:$callee,
      Variadic<AnyType>:$argOperands)>
  , Results<(outs AnyType:$result)>
{
  let summary = "VAST call operation";
  let description = [{ VAST call operation }];

  let assemblyFormat = [{
    $callee `:` type($callee)  `(` $argOperands `)` attr-dict `:` functional-type( $argOperands, $result )
  }];

  // TODO: add verifiers to check that callee type matches arg operands
}

def ExprOp
  : HighLevel_Op< "expr", [RecursiveSideEffects, SingleBlock] >
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
  : HighLevel_Op< "return", [Terminator, NoSideEffect] >
  , Arguments<(ins Variadic<AnyType>:$result)>
{
  let assemblyFormat = "($result^ `:` type($result))? attr-dict";

  // Allow building a ReturnOp with no return operand.
  let builders = [
    OpBuilder<(ins), [{ build($_builder, $_state, llvm::None); }]>
  ];
}

def DeclRefOp
  : HighLevel_Op< "ref", [NoSideEffect, SameOperandsAndResultType] >
  , Arguments<(ins AnyType:$decl)>
  , Results<(outs AnyType:$result)>
{
  let summary = "VAST variable reference declaration";
  let description = [{ VAST variable reference declaration }];

  let assemblyFormat = "$decl attr-dict `:` type($decl)";
}

def FuncRefOp
  : HighLevel_Op< "funcref", [NoSideEffect] >
  , Arguments<(ins FlatSymbolRefAttr:$function)>
  , Results<(outs AnyType:$result)>
{
  let summary = "VAST function reference declaration";
  let description = [{ VAST function reference declaration }];

  let assemblyFormat = "$function attr-dict `:` type($result)";
}

def GlobalRefOp
  : HighLevel_Op< "globref", [NoSideEffect] >
  , Arguments<(ins StrAttr:$global)>
  , Results<(outs AnyType:$result)>
{
  let summary = "VAST global variable reference declaration";
  let description = [{ VAST global variable reference declaration }];

  let assemblyFormat = "$global attr-dict `:` type($result)";
}

def EnumRefOp
  : HighLevel_Op< "enumref", [NoSideEffect] >
  , Arguments<(ins StrAttr:$value)>
  , Results<(outs AnyType:$result)>
{
  let summary = "VAST variable reference declaration";
  let description = [{ VAST variable reference declaration }];

  let assemblyFormat = "$value attr-dict `:` type($result)";
}

def ConstantOp
  : HighLevel_Op< "const", [ConstantLike, NoSideEffect, AllTypesMatch< ["value", "result"]>] >
  , Arguments<(ins TypedAttrInterface:$value)>
  , Results<(outs AnyType:$result)>
{
  let summary = "VAST integral constant";
  let description = [{ VAST integral constant }];

  let builders = [
    OpBuilder<(ins "Type":$type, "bool":$value), [{
      build($_builder, $_state, type, BooleanAttr::get(type, value));
    }]>,
    OpBuilder<(ins "Type":$type, "llvm::APSInt":$value), [{
      build($_builder, $_state, type, IntegerAttr::get(type, value));
    }]>,
    OpBuilder<(ins "Type":$type, "llvm::APFloat":$value), [{
      build($_builder, $_state, type, FloatAttr::get(type, value));
    }]>,
    OpBuilder<(ins "Type":$type, "llvm::StringRef":$value), [{
      build($_builder, $_state, type, StringAttr::get(value, type));
    }]>
  ];

  let hasFolder = 1;

  let assemblyFormat = "$value attr-dict";
}

def UnreachableOp : HighLevel_Op<"unreachable", [Terminator, NoSideEffect]> {
  let summary = "VAST unreachable operation";
  let description = [{ VAST unreachable operation }];
  let assemblyFormat = "attr-dict";
}

class CastKindAttr< string name, int val > : I64EnumAttrCase< name, val > {}

class CastKindList< string name, string summary, list<CastKindAttr> cases >
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
    : HighLevel_Op< mnemonic, !listconcat(traits, [NoSideEffect]) >
    , Arguments<(ins AnyType:$value, CastKind:$kind)>
    , Results<(outs AnyType:$result)>
{
    let summary = "VAST cast operation";
    let description = [{ VAST cast operation }];

    let assemblyFormat = "$value $kind attr-dict `:` type($value) `->` type($result)";
}

def ImplicitCastOp   : CastOp< "implicit_cast", [] >;
def CStyleCastOp     : CastOp< "cstyle_cast", [] >;
def BuiltinBitCastOp : CastOp< "builtin_bitcast", [] >;

class ArithBinOp< string mnemonic, list< Trait > traits = [] >
    : HighLevel_Op< mnemonic, !listconcat(traits, [NoSideEffect, SameOperandsAndResultType]) >
    , Arguments<(ins AnyType:$lhs, AnyType:$rhs)>
    , Results<(outs AnyType:$result)>
{
    let summary = "VAST arithmetic binary operation";
    let description = [{
        High-level arithmetic binary operation. This operation takes two operands
        and returns one result, each of these is required to be of the same
        type.

        The custom assembly form of the operation is as follows:

        %result = <op> %lhs, %rhs  : type
    }];

    let assemblyFormat = [{ $lhs `,` $rhs attr-dict `:` type($result) }];
}

def AddIOp : ArithBinOp< "add", [Commutative]>;
def AddFOp : ArithBinOp<"fadd", []>;
def SubIOp : ArithBinOp< "sub", []>;
def SubFOp : ArithBinOp<"fsub", []>;
def MulIOp : ArithBinOp< "mul", [Commutative]>;
def MulFOp : ArithBinOp<"fmul", []>;
def DivSOp : ArithBinOp<"sdiv", []>;
def DivUOp : ArithBinOp<"udiv", []>;
def DivFOp : ArithBinOp<"fdiv", []>;
def RemSOp : ArithBinOp<"srem", []>;
def RemUOp : ArithBinOp<"urem", []>;
def RemFOp : ArithBinOp<"frem", []>;

def BinXorOp : ArithBinOp<"bin.xor", []>;
def BinOrOp  : ArithBinOp< "bin.or", []>;
def BinAndOp : ArithBinOp<"bin.and", []>;


class LogicBinOp< string mnemonic, list< Trait > traits = [] >
    : HighLevel_Op< mnemonic, !listconcat(traits, [NoSideEffect]) >
    , Arguments<(ins AnyType:$lhs, AnyType:$rhs)>
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

    let assemblyFormat = [{ $lhs `,` $rhs attr-dict `:` functional-type(operands, results) }];
}

def BinLAndOp : LogicBinOp<"bin.land", []>;
def BinLOrOp  : LogicBinOp< "bin.lor", []>;

def BinComma
  : HighLevel_Op< "bin.comma", [NoSideEffect] >
  , Arguments<(ins AnyType:$lhs, AnyType:$rhs)>
  , Results<(outs AnyType:$result)>
{
    let summary = "VAST binary operation";

    let assemblyFormat = [{ $lhs `,` $rhs attr-dict `:` functional-type(operands, results) }];
}

class ShiftOp< string mnemonic, list< Trait > traits = [] >
    : HighLevel_Op< mnemonic, !listconcat(traits, [
        NoSideEffect, AllTypesMatch<["lhs", "result"]>
    ])>
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

def BinShlOp : ShiftOp<"bin.shl", []>;
def BinShrOp : ShiftOp<"bin.shr", []>;


class CompoundAssignOpTemplate< string mnemonic, TypeConstraint Type, list< Trait > traits = [] >
    : HighLevel_Op< mnemonic, !listconcat(traits, [
        TypesMatchWith<
          "underlying destination type match result type",
          "dst", "result",
          "$_self.cast< LValueType >().getElementType()"
        >
    ])>
    , Arguments<(ins LValueOrType<Type>:$src, LValueOf<Type>:$dst)>
    , Results<(outs Type:$result)>
{
    let summary = "VAST compound assign operation";
    let description = [{
        A compound assign operation represents an assignment operation joined
        with an arithmetic operation. It requires the same types for both source
        and destination arguments.

        The custom assembly form of the operation is as follows:

        %result = <assign.op> src to dst : type(src)

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
def AddIAssignOp : CompoundAssignOp< "assign.add"  >;
def AddFAssignOp : CompoundAssignOp< "assign.fadd" >;
def SubIAssignOp : CompoundAssignOp< "assign.sub"  >;
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
def BinShrAssignOp : ShiftAssignOp< "assign.bin.shr" >;

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


def CmpOp
  : HighLevel_Op< "cmp", [NoSideEffect] >
  , Arguments<(ins Predicate:$predicate, AnyType:$lhs, AnyType:$rhs)>
  , Results<(outs IntOrBoolType:$result)>
{
  let summary = "VAST comparison operation";
  let description = [{ VAST comparison operation }];

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

def PostIncOp : UnInplaceOp< "post.inc", [] >;
def PostDecOp : UnInplaceOp< "post.dec", [] >;
def PreIncOp  : UnInplaceOp<  "pre.inc", [] >;
def PreDecOp  : UnInplaceOp<  "pre.dec", [] >;

class TypePreservingUnOp< string mnemonic, list< Trait > traits = [] >
    : HighLevel_Op< mnemonic, !listconcat(traits, [NoSideEffect, SameOperandsAndResultType]) >
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

def PlusOp  : TypePreservingUnOp< "plus", [] >;
def MinusOp : TypePreservingUnOp< "minus", [] >;
def NotOp   : TypePreservingUnOp< "not", [] >;
def LNotOp  : TypePreservingUnOp< "lnot", [] >;

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
  : HighLevel_Op< "deref", [] >
  // TODO(Heno): check dereferencable
  , Arguments<(ins AnyType:$addr)>
  , Results<(outs LValueOf<AnyType>:$result)>
{
    let summary = "VAST deref operation";
    let description = [{ VAST deref operation }];

    let assemblyFormat = [{ $addr attr-dict `:` type($addr) `->` type($result) }];
}

def AddrLabelExpr
  : HighLevel_Op< "labeladdr", [] >
  , Arguments<(ins LabelType:$label)>
  , Results<(outs LValueOf<PointerLikeType>:$result)>
{
    let summary = "VAST address of label extension";
    let description = [{ VAST address of label extension }];

    let assemblyFormat = [{ $label attr-dict `:` type($result) }];
}

def InitListExpr
  : HighLevel_Op< "initlist", [NoSideEffect] >
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
  : HighLevel_Op< mnemonic, !listconcat(traits, [NoSideEffect]) >
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
  : HighLevel_Op< mnemonic, !listconcat(traits, [RecursiveSideEffects]) >
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

#endif // VAST_DIALECT_HIGHLEVEL_IR_HIGHLEVELOPS

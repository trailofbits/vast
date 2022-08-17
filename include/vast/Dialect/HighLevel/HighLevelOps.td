// Copyright (c) 2021-present, Trail of Bits, Inc.

#ifndef VAST_DIALECT_HIGHLEVEL_IR_HIGHLEVELOPS
#define VAST_DIALECT_HIGHLEVEL_IR_HIGHLEVELOPS

include "mlir/Interfaces/CallInterfaces.td"
include "vast/Interfaces/SymbolInterface.td"

def HighLevel_TranslationUnitOp
  : HighLevel_Op< "translation.unit", [NoTerminator, SymbolTable, IsolatedFromAbove] >
{
  let summary = "VAST translation unit";
  let description = [{ VAST tranaslation unit }];

  let regions = (region AnyRegion:$body);

  let assemblyFormat = [{ $body attr-dict }];
}

def HighLevel_ScopeOp
  : HighLevel_Op< "scope", [NoTerminator] >
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

def HighLevel_TypeDeclOp
  : HighLevel_Op< "type.decl", [NoSideEffect, VastSymbol] >
  , Arguments<(ins StrAttr:$name)>
{
  let summary = "VAST type declaration";
  let description = [{ VAST type declaration }];

  let assemblyFormat = [{ $name attr-dict }];
}

def HighLevel_TypeDefOp
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

def HighLevel_EnumConstantOp
  : HighLevel_Op< "enum.const", [NoSideEffect] >
{
  let summary = "VAST enum constant declaration";
  let description = [{
    Enumeration constant servers to link name to an enum value.
    It is required to be scoped in Enum operation. For example:

    ```
    hl.enum.const "F" = 2 : si32
    ```

    A constant can have a constant expression initializer:

    ```
    hl.enum.const "G" = 12 : si32 init  {
      %0 = hl.enumref "F" : !hl.int
      %1 = hl.enumref "C" : !hl.int
      %2 = hl.add %0, %1 : !hl.int
      hl.value.yield %2 : !hl.int
    }
    ```
  }];

  let arguments = (ins
    StrAttr:$name,
    APIntAttr:$value
  );

  let regions = (region AnyRegion:$init);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins
      // TODO use identifier
      "llvm::StringRef":$name,
      "llvm::APSInt":$value,
      CArg<"BuilderCallback", "std::nullopt">:$init
    )>
  ];

  let assemblyFormat = [{ $name `=` $value attr-dict (`init` $init^)? }];
}

def EnumRegion : Region< HasOneBlock, "enum constants region" >;

def HighLevel_EnumDeclOp
  : HighLevel_Op< "enum.decl", [NoSideEffect, NoTerminator, VastSymbol] >
{
  let summary = "VAST enum declaration";
  let description = [{
    Enum declaration serves to declare region for enum constant declarations.
    It also defines an underlying type.
  }];

  let arguments = (ins
    StrAttr:$name,
    TypeAttr:$type // TODO(Heno): check integer like type attr
  );

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

class HighLevel_RecordLikeDeclOp< string mnemonic, list< Trait > traits = [] >
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

def HighLevel_StructDeclOp : HighLevel_RecordLikeDeclOp<"struct"> {
  let summary = "VAST struct declaration";
  let description = [{ VAST struct declaration }];
}

def HighLevel_UnionDeclOp : HighLevel_RecordLikeDeclOp<"union"> {
  let summary = "VAST record declaration";
  let description = [{ VAST record declaration }];
}

def HighLevel_FieldDeclOp
  : HighLevel_Op< "field", [NoSideEffect, VastSymbol] >
  , Arguments<(ins StrAttr:$name, TypeAttr:$type)>
{
  let summary = "VAST record field declaration";
  let description = [{ VAST record field declaration }];

  let assemblyFormat = [{ $name attr-dict `:` $type }];
}

def HighLevel_RecordMemberOp
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

def HighLevel_CallOp : HighLevel_Op< "call", [DeclareOpInterfaceMethods<CallOpInterface>] >
{
  let summary = "VAST call operation";
  let description = [{ VAST call operation }];

  let arguments = (ins
    FlatSymbolRefAttr:$callee,
    Variadic<AnyType>:$operands
  );

  let results = (outs AnyType:$result);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder< (ins "mlir::FuncOp":$callee, CArg<"mlir::ValueRange", "{}">:$operands ), [{
      $_state.addOperands(operands);
      $_state.addAttribute("callee", mlir::SymbolRefAttr::get($_builder.getContext(), callee.getName()));
      $_state.addTypes(callee.getType().getResults());
    }]>
  ];

  let assemblyFormat = [{
    $callee `(` $operands `)` attr-dict `:` functional-type( $operands, $result )
  }];
}

def HighLevel_IndirectCallOp : HighLevel_Op< "indirect_call", [
  DeclareOpInterfaceMethods<CallOpInterface>,
  TypesMatchWith<
    "callee input types match argument types",
    "callee", "operands",
    "getFunctionType($_self).getInputs()"
  >,
  TypesMatchWith<
    "callee result type match result type",
    "callee", "result",
    "getFunctionType($_self).getResult(0)"
  >
] > {
  let summary = "VAST call operation";
  let description = [{ VAST call operation }];

  let arguments = (ins
    PointerLikeType:$callee,
    Variadic<AnyType>:$operands
  );

  let results = (outs AnyType:$result);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder< (ins "mlir::Value":$callee, CArg<"mlir::ValueRange", "{}">:$operands ), [{
      $_state.addOperands(callee);
      $_state.addOperands(operands);
      $_state.addTypes(getFunctionType(callee.getType()).getResult(0));
    }]>
  ];

  let assemblyFormat = [{
    $callee `(` $operands `)` attr-dict `:` type( $callee )
  }];
}

def HighLevel_ExprOp
  : HighLevel_Op< "expr", [RecursiveSideEffects, SingleBlock] >
{
  let summary = "VAST expression";
  let description = [{ VAST expression }];

  let regions = (region SizedRegion<1>:$subexpr);
  let results = (outs AnyType:$result);

  let skipDefaultBuilders = 1;
  let builders = [ OpBuilder<(ins "Type":$rty, "std::unique_ptr< Region > &&":$region)> ];

  let assemblyFormat = [{
     attr-dict `:` type($result) $subexpr
  }];
}

def HighLevel_ReturnOp
  : HighLevel_Op< "return", [Terminator, NoSideEffect] >
{
  let arguments = (ins Variadic<AnyType>:$result);

  let assemblyFormat = "($result^ `:` type($result))? attr-dict";

  // Allow building a ReturnOp with no return operand.
  let builders = [
    OpBuilder<(ins), [{ build($_builder, $_state, llvm::None); }]>
  ];
}

def HighLevel_DeclRefOp
  : HighLevel_Op< "decl.ref", [NoSideEffect, SameOperandsAndResultType] >
  , Arguments<(ins AnyType:$decl)>
  , Results<(outs AnyType:$result)>
{
  let summary = "VAST variable reference declaration";
  let description = [{ VAST variable reference declaration }];

  let assemblyFormat = "$decl attr-dict `:` type($decl)";
}

def HighLevel_GlobalRefOp
  : HighLevel_Op< "global.ref", [NoSideEffect] >
  , Arguments<(ins StrAttr:$global)>
  , Results<(outs AnyType:$result)>
{
  let summary = "VAST global variable reference declaration";
  let description = [{ VAST global variable reference declaration }];

  let assemblyFormat = "$global attr-dict `:` type($result)";
}

def HighLevel_EnumRefOp
  : HighLevel_Op< "enumref", [NoSideEffect] >
  , Arguments<(ins StrAttr:$value)>
  , Results<(outs AnyType:$result)>
{
  let summary = "VAST variable reference declaration";
  let description = [{ VAST variable reference declaration }];

  let assemblyFormat = "$value attr-dict `:` type($result)";
}

class HighLevel_ConstantOp< string mnemonic, list< Trait > traits = [] >
  : HighLevel_Op< mnemonic, !listconcat(traits, [ConstantLike, NoSideEffect]) >
{
  let extraClassDeclaration = [{
    Attribute getValue() { return (*this)->getAttr("value"); }
  }];

  let parser = [{ return parse$cppClass(parser, result); }];
  let printer = [{ print$cppClass(p, *this); }];

  let hasFolder = 1;
}

def ConstantIntOp : HighLevel_ConstantOp< "constant.int" >
  , Arguments<(ins APSIntAttr:$value)>
  , Results<(outs IntOrBoolType:$result)>
{
  let summary = "VAST integral constant";
  let description = [{ VAST integral constant }];

  let builders = [
    OpBuilder<(ins "Type":$type, "llvm::APInt":$value), [{
      build($_builder, $_state, type, llvm::APSInt(value, isSigned(type)));
    }]>
  ];
}

def HighLevel_ConstantFloatOp : HighLevel_ConstantOp< "constant.float" >
  , Arguments<(ins APFloatAttr:$value)>
  , Results<(outs FloatLikeType:$result)>
{
  let summary = "VAST floating constant";
  let description = [{ VAST floating constant }];
}

def HighLevel_ConstantArrayOp : HighLevel_ConstantOp< "constant.array" >
  , Arguments<(ins ArrayAttr:$value)>
  , Results<(outs ArrayType:$result)>
{
  let summary = "VAST constant array";
  let description = [{ VAST constant array }];
}

def HighLevel_ConstantStringOp : HighLevel_ConstantOp< "constant.string" >
  , Arguments<(ins StrAttr:$value)>
  , Results<(outs ArrayType:$result)>
{
  let summary = "VAST constant string";
  let description = [{ VAST constant string }];
}

def HighLevel_UnreachableOp : HighLevel_Op<"unreachable", [Terminator, NoSideEffect]> {
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

class HighLevel_CastOp< string mnemonic, list< Trait > traits = [] >
    : HighLevel_Op< mnemonic, !listconcat(traits, [NoSideEffect]) >
{
    let summary = "VAST cast operation";
    let description = [{ VAST cast operation }];

    let arguments = (ins AnyType:$value, CastKind:$kind);
    let results = (outs AnyType:$result);

    let assemblyFormat = "$value $kind attr-dict `:` type($value) `->` type($result)";
}

def HighLevel_ImplicitCastOp   : HighLevel_CastOp< "implicit_cast", [] >;
def HighLevel_CStyleCastOp     : HighLevel_CastOp< "cstyle_cast", [] >;
def HighLevel_BuiltinBitCastOp : HighLevel_CastOp< "builtin_bitcast", [] >;


class HighLevel_ArithBinOp< string mnemonic, list< Trait > traits = [] >
    : HighLevel_Op< mnemonic, !listconcat(traits, [NoSideEffect, SameOperandsAndResultType]) >
{
    let summary = "VAST arithmetic binary operation";
    let description = [{
        High-level arithmetic binary operation. This operation takes two operands
        and returns one result, each of these is required to be of the same
        type.

        The custom assembly form of the operation is as follows:

        %result = <op> %lhs, %rhs  : type
    }];

    let arguments = (ins AnyType:$lhs, AnyType:$rhs);
    let results = (outs AnyType:$result);

    let assemblyFormat = [{ $lhs `,` $rhs attr-dict `:` type($result) }];
}

def HighLevel_AddIOp : HighLevel_ArithBinOp< "add", [Commutative]>;
def HighLevel_AddFOp : HighLevel_ArithBinOp<"fadd", []>;
def HighLevel_SubIOp : HighLevel_ArithBinOp< "sub", []>;
def HighLevel_SubFOp : HighLevel_ArithBinOp<"fsub", []>;
def HighLevel_MulIOp : HighLevel_ArithBinOp< "mul", [Commutative]>;
def HighLevel_MulFOp : HighLevel_ArithBinOp<"fmul", []>;
def HighLevel_DivSOp : HighLevel_ArithBinOp<"sdiv", []>;
def HighLevel_DivUOp : HighLevel_ArithBinOp<"udiv", []>;
def HighLevel_DivFOp : HighLevel_ArithBinOp<"fdiv", []>;
def HighLevel_RemSOp : HighLevel_ArithBinOp<"srem", []>;
def HighLevel_RemUOp : HighLevel_ArithBinOp<"urem", []>;
def HighLevel_RemFOp : HighLevel_ArithBinOp<"frem", []>;

def HighLevel_BinXorOp : HighLevel_ArithBinOp<"bin.xor", []>;
def HighLevel_BinOrOp  : HighLevel_ArithBinOp< "bin.or", []>;
def HighLevel_BinAndOp : HighLevel_ArithBinOp<"bin.and", []>;


class HighLevel_LogicBinOp< string mnemonic, list< Trait > traits = [] >
    : HighLevel_Op< mnemonic, !listconcat(traits, [NoSideEffect, SameOperandsAndResultType]) >
{
    let summary = "VAST logical binary operation";
    let description = [{
        High-level logical binary operation. This operation takes two operands
        and returns one result, each of these is required to be of the same
        type.

        The custom assembly form of the operation is as follows:

        %result = <op> %lhs, %rhs  : type
    }];

    let arguments = (ins AnyType:$lhs, AnyType:$rhs);
    let results = (outs AnyType:$result);

    let assemblyFormat = [{ $lhs `,` $rhs attr-dict `:` type($result) }];
}

def HighLevel_BinLAndOp : HighLevel_LogicBinOp<"bin.land", []>;
def HighLevel_BinLOrOp  : HighLevel_LogicBinOp< "bin.lor", []>;

def HighLevel_BinComma
    : HighLevel_Op< "bin.comma", [NoSideEffect] >
{
    let summary = "VAST binary operation";

    let arguments = (ins AnyType:$lhs, AnyType:$rhs);
    let results = (outs AnyType:$result);

    let assemblyFormat = [{ $lhs `,` $rhs attr-dict `:` functional-type(operands, results) }];
}

class HighLevel_ShiftOp< string mnemonic, list< Trait > traits = [] >
    : HighLevel_Op< mnemonic, !listconcat(traits, [
        NoSideEffect, AllTypesMatch<["lhs", "result"]>
    ])>
{
    let summary = "VAST binary shift operation";
    let description = [{
        High-level binary shift operation. This operation takes two operands
        and returns one result.

        The custom assembly form of the operation is as follows:

        %result = <op> %lhs, %rhs  : functional-type(operands, results)
    }];

    let arguments = (ins IntegerLikeType:$lhs, IntegerLikeType:$rhs);
    let results = (outs IntegerLikeType:$result);

    let assemblyFormat = [{ $lhs `,` $rhs attr-dict `:` functional-type(operands, results) }];
}

def HighLevel_BinShlOp : HighLevel_ShiftOp<"bin.shl", []>;
def HighLevel_BinShrOp : HighLevel_ShiftOp<"bin.shr", []>;

class HighLevel_CompoundAssignOpTemplate< string mnemonic, TypeConstraint Type, list< Trait > traits = [] >
    : HighLevel_Op< mnemonic, !listconcat(traits, [
      TypesMatchWith<
        "underlying lvalue type is same as source type",
        "src", "dst", "LValueType::get( $_ctxt, $_self )"
      >,
      AllTypesMatch< ["src", "result"] >
    ])>
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

    let arguments = (ins Type:$src, LValueOf<Type>:$dst);
    let results = (outs Type:$result);

    let skipDefaultBuilders = 1;
    let builders = [ OpBuilder<(ins "Value":$dst, "Value":$src), [{
      $_state.addOperands(src);
      $_state.addOperands(dst);
      $_state.addTypes(src.getType());
    }]>
  ];

    let assemblyFormat = [{ $src `to` $dst attr-dict `:` type($src) }];
}

class HighLevel_CompoundAssignOp< string mnemonic, list< Trait > traits = [] >
    : HighLevel_CompoundAssignOpTemplate< mnemonic, AnyType, traits > {}

def HighLevel_AssignOp     : HighLevel_CompoundAssignOp< "assign" >;
def HighLevel_AddIAssignOp : HighLevel_CompoundAssignOp< "assign.add"  >;
def HighLevel_AddFAssignOp : HighLevel_CompoundAssignOp< "assign.fadd" >;
def HighLevel_SubIAssignOp : HighLevel_CompoundAssignOp< "assign.sub"  >;
def HighLevel_SubFAssignOp : HighLevel_CompoundAssignOp< "assign.fsub" >;
def HighLevel_MulIAssignOp : HighLevel_CompoundAssignOp< "assign.mul"  >;
def HighLevel_MulFAssignOp : HighLevel_CompoundAssignOp< "assign.fmul" >;
def HighLevel_DivSAssignOp : HighLevel_CompoundAssignOp< "assign.sdiv" >;
def HighLevel_DivUAssignOp : HighLevel_CompoundAssignOp< "assign.udiv" >;
def HighLevel_DivFAssignOp : HighLevel_CompoundAssignOp< "assign.fdiv" >;
def HighLevel_RemSAssignOp : HighLevel_CompoundAssignOp< "assign.srem" >;
def HighLevel_RemUAssignOp : HighLevel_CompoundAssignOp< "assign.urem" >;
def HighLevel_RemFAssignOp : HighLevel_CompoundAssignOp< "assign.frem" >;

def HighLevel_BinAndAssignOp : HighLevel_CompoundAssignOp< "assign.bin.and" >;
def HighLevel_BinOrAssignOp  : HighLevel_CompoundAssignOp< "assign.bin.or"  >;
def HighLevel_BinXorAssignOp : HighLevel_CompoundAssignOp< "assign.bin.xor" >;

class HighLevel_ShiftAssignOp< string mnemonic, list< Trait > traits = [] >
    : HighLevel_CompoundAssignOpTemplate< mnemonic, IntegerLikeType, traits > {}

def HighLevel_BinShlAssignOp : HighLevel_ShiftAssignOp< "assign.bin.shl" >;
def HighLevel_BinShrAssignOp : HighLevel_ShiftAssignOp< "assign.bin.shr" >;

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


def HighLevel_CmpOp : HighLevel_Op< "cmp", [NoSideEffect, SameTypeOperands, Commutative] >
{
  let summary = "VAST comparison operation";
  let description = [{ VAST comparison operation }];

  let arguments = (ins Predicate:$predicate, AnyType:$lhs, AnyType:$rhs);
  let results = (outs IntOrBoolType:$result);

  let assemblyFormat = "$predicate type($lhs) $lhs `,` $rhs  attr-dict `->` type($result)";
}


class HighLevel_UnInplaceOp< string mnemonic, list< Trait > traits = [] >
    : HighLevel_Op< mnemonic, !listconcat(traits, [SameOperandsAndResultType]) >
{
    let summary = "VAST unary inplace operation";
    let description = [{
        Inplace high-level unary operation changes its single argument in place.
        It does not produce a new value.

        The custom assembly form of the operation is as follows:

        %result = <op> %arg : type
    }];

    let arguments = (ins AnyType:$arg);
    let results = (outs AnyType:$result);

    let assemblyFormat = [{ $arg attr-dict `:` type($result) }];
}

def HighLevel_PostIncOp : HighLevel_UnInplaceOp< "post.inc", [] >;
def HighLevel_PostDecOp : HighLevel_UnInplaceOp< "post.dec", [] >;
def HighLevel_PreIncOp  : HighLevel_UnInplaceOp<  "pre.inc", [] >;
def HighLevel_PreDecOp  : HighLevel_UnInplaceOp<  "pre.dec", [] >;


class HighLevel_TypePreservingUnOp< string mnemonic, list< Trait > traits = [] >
    : HighLevel_Op< mnemonic, !listconcat(traits, [NoSideEffect, SameOperandsAndResultType]) >
{
    let summary = "VAST unary type preserving operation";
    let description = [{
        Type preserving high-level unary operation assures that argument and
        result has the same type.

        The custom assembly form of the operation is as follows:

        %result = <op> %arg : type
    }];

    let arguments = (ins AnyType:$arg);
    let results = (outs AnyType:$result);

    let assemblyFormat = [{ $arg attr-dict `:` type($result) }];
}

def HighLevel_PlusOp  : HighLevel_TypePreservingUnOp< "plus", [] >;
def HighLevel_MinusOp : HighLevel_TypePreservingUnOp< "minus", [] >;
def HighLevel_NotOp   : HighLevel_TypePreservingUnOp< "not", [] >;
def HighLevel_LNotOp  : HighLevel_TypePreservingUnOp< "lnot", [] >;

def HighLevel_AddressOf
  : HighLevel_Op< "addressof", [] >
  // TODO(Heno): parameter constraints
  , Arguments<(ins LValueOf<AnyType>:$value)>
  , Results<(outs AnyType:$result)>
{
    let summary = "VAST addressof operation";
    let description = [{ VAST addressof operation }];

    let assemblyFormat = [{ $value attr-dict `:` type($value) `->` type($result) }];
}

def HighLevel_Deref
  : HighLevel_Op< "deref", [] >
  // TODO(Heno): check dereferencable
  , Arguments<(ins AnyType:$addr)>
  , Results<(outs LValueOf<AnyType>:$result)>
{
    let summary = "VAST deref operation";
    let description = [{ VAST deref operation }];

    let assemblyFormat = [{ $addr attr-dict `:` type($addr) `->` type($result) }];
}

def HighLevel_InitListExpr : HighLevel_Op< "initlist",
  [NoSideEffect]
> {
  let summary = "VAST initializer list expression";
  let description = [{ VAST initializer list expression }];

  let arguments = (ins Variadic<AnyType>:$elements);
  let results = (outs Variadic<AnyType>);

  let assemblyFormat = "$elements attr-dict `:` functional-type($elements, results)";
}

def HighLevel_SubscriptOp : HighLevel_Op< "subscript" >
{
  let summary = "VAST array subscript operator";
  let description = [{ VAST array subscript operator }];

  let arguments = (ins
    LValueOf<SubscriptableType>:$array,
    IntegerLikeType:$index
  );

  let results = (outs LValueOf<AnyType>:$result);

  let assemblyFormat = [{
    $array `at` ` ` `[` $index `:` type($index) `]` attr-dict
      `:` type($array) `->` type($result)
  }];
}

class HighLevel_TypeTraitOp< string mnemonic, list< Trait > traits = [] >
  : HighLevel_Op< mnemonic, !listconcat(traits, [NoSideEffect]) >
{
  let arguments = (ins TypeAttr:$type);
  let results   = (outs IntegerLikeType:$result);

  let assemblyFormat = [{ $type attr-dict `->` type($result) }];
}

def HighLevel_SizeOfTypeOp : HighLevel_TypeTraitOp< "sizeof.type" > {
  let summary = "VAST type sizeof operator";
  let description = [{ VAST type sizeof operator }];
}

def HighLevel_AlignOfTypeOp : HighLevel_TypeTraitOp< "alignof.type" > {
  let summary = "VAST type alignof operator";
  let description = [{ VAST type alignof operator }];
}

class HighLevel_ExprTraitOp< string mnemonic, list< Trait > traits = [] >
  : HighLevel_Op< mnemonic, !listconcat(traits, [RecursiveSideEffects]) >
{
  let regions = (region SizedRegion<1>:$expr);
  let results = (outs IntegerLikeType:$result);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "Type":$type, "BuilderCallback":$exprBuilder)>
  ];

  let assemblyFormat = [{ attr-dict `->` type($result) $expr }];
}

def HighLevel_SizeOfExprOp : HighLevel_ExprTraitOp< "sizeof.expr" > {
  let summary = "VAST expr sizeof operator";
  let description = [{ VAST expr sizeof operator }];
}

def HighLevel_AlignOfExprOp : HighLevel_ExprTraitOp< "alignof.expr" > {
  let summary = "VAST expr alignof operator";
  let description = [{ VAST expr alignof operator }];
}

#endif // VAST_DIALECT_HIGHLEVEL_IR_HIGHLEVELOPS

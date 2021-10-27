// Copyright (c) 2021-present, Trail of Bits, Inc.

#ifndef VAST_DIALECT_HIGHLEVEL_IR_HIGHLEVELOPS
#define VAST_DIALECT_HIGHLEVEL_IR_HIGHLEVELOPS

def HighLevel_ScopeOp
  : HighLevel_Op< "scope", [NoTerminator] >
{
  let arguments = (ins);
  let results = (outs);
  let regions = (region AnyRegion:$body);

  let assemblyFormat = [{ $body attr-dict }];
}


def HighLevel_ReturnOp
  : HighLevel_Op< "return", [Terminator, NoSideEffect] >
{
  let arguments = (ins Variadic<AnyType>:$result);

  let assemblyFormat = "($result^ `:` type($result))? attr-dict ";

  // Allow building a ReturnOp with no return operand.
  let builders = [
    OpBuilder<(ins), [{ build($_builder, $_state, llvm::None); }]>
  ];
}

def HighLevel_VarOp : HighLevel_Op< "var", [] > {
  let summary = "VAST variable declaration";
  let description = [{ VAST variable declaration }];

  let arguments = (ins
    StrAttr:$name,
    Optional<AnyType>:$initializer
  );

  let results = (outs AnyType:$result);

  let builders = [
    OpBuilder< (ins "mlir::Type":$result, "llvm::StringRef":$name),
    [{
      build($_builder, $_state, result, $_builder.getStringAttr(name));
    }] >,
    OpBuilder< (ins "mlir::Type":$result, "mlir::StringAttr":$name),
    [{
      build($_builder, $_state, result, name, mlir::Value());
    }] >
  ];

  let verifier = ? ;
  let printer = [{
    p << getOperationName()
      << "( " <<  name();
    if (auto init = initializer())
      p << ", " << init;
    p << " )";
    p << ": " << getResult().getType();
  }];
}


def HighLevel_DeclRefOp : HighLevel_Op< "declref", [] > {
  let summary = "VAST reference declaration";
  let description = [{ VAST reference declaration }];

  let arguments = (ins SymbolRefAttr:$decl);
  let results = (outs AnyType:$result);

  let builders = [
    OpBuilder< (ins "mlir::Type":$rty, "llvm::StringRef":$name),
    [{
      build($_builder, $_state, rty, $_builder.getSymbolRefAttr(name));
    }] >
  ];

  let assemblyFormat = "$decl attr-dict `:` type($result) ";
}


def HighLevel_ConstantOp : HighLevel_Op< "constant", [] > {
  let summary = "VAST constant";
  let description = [{ VAST constant }];

  let arguments = (ins AnyAttr:$value);

  // FIXME: More specific type.
  let results = (outs AnyType);

  let builders = [OpBuilder< (
      ins "mlir::Attribute":$value
    ) >
  ];

  let verifier = ? ;
  let printer = [{ p << getOperationName()
                     << "( " <<  value() << " )";
                   p << ": " << getResult().getType();
  }];
}


def HighLevel_VoidOp : HighLevel_Op<"void", []> {
  let summary = "VAST void operation";
  let description = [{ VAST void operation }];

  let results = (outs Void);

  let builders = [
    OpBuilder<(ins),
    [{
      build($_builder, $_state, VoidType::get($_builder.getContext()));
    }] >
  ];

  let verifier = ?;
  let printer = [{ p << getOperationName(); }];
}


def HighLevel_UnreachableOp : HighLevel_Op<"unreachable", [Terminator]> {
  let summary = "VAST unreachable operation";
  let description = [{ VAST unreachable operation }];

  let arguments = (ins Variadic<AnyType>:$operands);

  let builders = [OpBuilder<(ins), [{ }]>];

  let verifier = ?;
  let printer = [{
    p << getOperationName() << " ";
    if (!operands().empty())
      p << *operand_begin();
  }];
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

class HighLevel_CastOp< string mnemonic, list< OpTrait > traits = [] >
    : HighLevel_Op< mnemonic, !listconcat(traits, [NoSideEffect]) >
{
    let summary = "VAST cast operation";
    let description = [{ VAST cast operation }];

    let arguments = (ins AnyType:$value, CastKind:$kind);
    let results = (outs AnyType:$res);

    let verifier = ? ;

    let printer = [{
        p << getOperationName()
        << "( " << value() << " ) " << stringifyCastKind(kind()) << ": "
        << getResult().getType();
    }];
}


def HighLevel_ImplicitCastOp   : HighLevel_CastOp< "implicit_cast", [] >;
def HighLevel_CStyleCastOp     : HighLevel_CastOp< "cstyle_cast", [] >;
def HighLevel_BuiltinBitCastOp : HighLevel_CastOp< "builtin_bit_cast", [] >;


def HighLevel_AddIOp : HighLevel_BinOp< "add", [Commutative]>;
def HighLevel_AddFOp : HighLevel_BinOp<"fadd", []>;
def HighLevel_SubIOp : HighLevel_BinOp< "sub", []>;
def HighLevel_SubFOp : HighLevel_BinOp<"fsub", []>;
def HighLevel_MulIOp : HighLevel_BinOp< "mul", [Commutative]>;
def HighLevel_MulFOp : HighLevel_BinOp<"fmul", []>;
def HighLevel_DivSOp : HighLevel_BinOp<"sdiv", []>;
def HighLevel_DivUOp : HighLevel_BinOp<"udiv", []>;
def HighLevel_DivFOp : HighLevel_BinOp<"fdiv", []>;
def HighLevel_RemSOp : HighLevel_BinOp<"srem", []>;
def HighLevel_RemUOp : HighLevel_BinOp<"urem", []>;
def HighLevel_RemFOp : HighLevel_BinOp<"frem", []>;


def HighLevel_AssignOp     : HighLevel_CompoundAssignOp< "assign", []>;
def HighLevel_AddIAssignOp : HighLevel_CompoundAssignOp< "assign.add",  []>;
def HighLevel_AddFAssignOp : HighLevel_CompoundAssignOp< "assign.fadd", []>;
def HighLevel_SubIAssignOp : HighLevel_CompoundAssignOp< "assign.sub",  []>;
def HighLevel_SubFAssignOp : HighLevel_CompoundAssignOp< "assign.fsub", []>;
def HighLevel_MulIAssignOp : HighLevel_CompoundAssignOp< "assign.mul",  []>;
def HighLevel_MulFAssignOp : HighLevel_CompoundAssignOp< "assign.fmul", []>;
def HighLevel_DivSAssignOp : HighLevel_CompoundAssignOp< "assign.sdiv", []>;
def HighLevel_DivUAssignOp : HighLevel_CompoundAssignOp< "assign.udiv", []>;
def HighLevel_DivFAssignOp : HighLevel_CompoundAssignOp< "assign.fdiv", []>;
def HighLevel_RemSAssignOp : HighLevel_CompoundAssignOp< "assign.srem", []>;
def HighLevel_RemUAssignOp : HighLevel_CompoundAssignOp< "assign.urem", []>;
def HighLevel_RemFAssignOp : HighLevel_CompoundAssignOp< "assign.frem", []>;


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


def HighLevel_CmpOp : HighLevel_Op< "cmp", [NoSideEffect, SameTypeOperands] >
{
  let summary = "VAST comparison operation";
  let description = [{ VAST comparison operation }];

  let arguments = (ins Predicate:$predicate, AnyType:$lhs, AnyType:$rhs);
  let results = (outs Bool:$result);

  let assemblyFormat = "$predicate $lhs `,` $rhs  attr-dict `:` type($lhs)";

  let builders = [
    OpBuilder<(ins "Predicate":$pred, "Value":$lhs, "Value":$rhs), [{
      auto boolty = BoolType::get($_builder.getContext());
      return build($_builder, $_state, boolty, pred, lhs, rhs);
    }]>
  ];
}


def HighLevel_PostIncOp : HighLevel_UnInplaceOp< "post.inc", [] >;
def HighLevel_PostDecOp : HighLevel_UnInplaceOp< "post.dec", [] >;
def HighLevel_PreIncOp  : HighLevel_UnInplaceOp<  "pre.inc", [] >;
def HighLevel_PreDecOp  : HighLevel_UnInplaceOp<  "pre.dec", [] >;


def HighLevel_PlusOp  : HighLevel_UnOp< "plus", [] >;
def HighLevel_MinusOp : HighLevel_UnOp< "minus", [] >;
def HighLevel_NotOp   : HighLevel_UnOp< "not", [] >;
def HighLevel_LNotOp  : HighLevel_UnOp< "lnot", [] >;

#endif // VAST_DIALECT_HIGHLEVEL_IR_HIGHLEVELOPS

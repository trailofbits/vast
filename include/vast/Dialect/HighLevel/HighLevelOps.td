// Copyright (c) 2021-present, Trail of Bits, Inc.

#ifndef VAST_DIALECT_HIGHLEVEL_IR_HIGHLEVELOPS
#define VAST_DIALECT_HIGHLEVEL_IR_HIGHLEVELOPS

include "mlir/IR/OpBase.td"
include "mlir/IR/BuiltinAttributes.td"

include "mlir/Interfaces/CastInterfaces.td"

include "vast/Interfaces/AggregateTypeDefinitionInterface.td"
include "vast/Interfaces/SymbolInterface.td"
include "vast/Interfaces/SymbolTableInterface.td"

include "vast/Dialect/HighLevel/HighLevelCF.td"

include "vast/Dialect/Core/CommonAttrConstraints.td"
include "vast/Dialect/Core/CoreTraits.td"
include "vast/Dialect/Core/Func.td"
include "vast/Dialect/Core/Utils.td"

include "vast/Interfaces/AST/DeclInterface.td"


def HighLevel_TranslationUnitOp
  : HighLevel_Op< "translation_unit", [
      NoTerminator,
      ShadowingSymbolTable< [[FuncSymbol, VarSymbol, TypeSymbol], [ElaboratedTypeSymbol]] >,
      IsolatedFromAbove
  ] >
{
  let summary = "VAST translation unit";
  let description = [{ VAST tranaslation unit }];

  let regions = (region AnyRegion:$body);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins
      "builder_callback_ref":$bodyBuilder
    )>
  ];

  let assemblyFormat = [{ $body attr-dict }];
}

def HighLevel_FuncOp
  : Core_FuncBaseOp< HighLevel_Dialect, "func", [
    FuncSymbol,
    ShadowingSymbolTable< [[VarSymbol, TypeSymbol], [ElaboratedTypeSymbol]] >,
    IsolatedFromAbove
  ] >
{
  let skipDefaultBuilders = 1;

  let builders = [OpBuilder< (ins
    "llvm::StringRef":$name,
    "core::FunctionType":$type,
    CArg< "std::optional< core::GlobalLinkageKind >", "{ core::GlobalLinkageKind::ExternalLinkage }" >:$linkage,
    CArg< "llvm::ArrayRef<mlir::NamedAttribute>", "{}" >:$attrs,
    CArg< "llvm::ArrayRef<mlir::DictionaryAttr>", "{}" >:$arg_attrs,
    CArg< "llvm::ArrayRef<mlir::DictionaryAttr>", "{}" >:$res_attrs,
    CArg< "maybe_builder_callback_ref", "std::nullopt" >:$body), [{
      InsertionGuard guard($_builder);
      build_region($_builder, $_state, body);

      $_state.addAttribute(
        mlir::SymbolTable::getSymbolAttrName(), $_builder.getStringAttr(name)
      );
      $_state.addAttribute(getFunctionTypeAttrName($_state.name), mlir::TypeAttr::get(type));
      if (linkage) {
        $_state.addAttribute(
          "linkage", core::GlobalLinkageKindAttr::get($_builder.getContext(), linkage.value())
        );
      }
      $_state.attributes.append(attrs.begin(), attrs.end());

      mlir::function_interface_impl::addArgAndResultAttrs(
        $_builder, $_state, arg_attrs, res_attrs,
        getArgAttrsAttrName($_state.name), getResAttrsAttrName($_state.name)
      );
    }] >
  ];

  let hasVerifier = 1;
}

def HighLevel_TypeDeclOp
  : HighLevel_Op< "type", [TypeSymbol] >
  , Arguments<(ins StrAttr:$name)>
{
  let summary = "VAST type declaration";
  let description = [{ VAST type declaration }];

  let extraClassDeclaration = [{
    mlir_type getDefinedType();
  }];

  let assemblyFormat = [{ $name attr-dict }];
}

def HighLevel_TypeDefOp
  : HighLevel_Op< "typedef", [TypeSymbol] >
  , Arguments<(ins SymbolNameAttr:$sym_name, TypeAttr:$type)>
{
  let summary = "VAST typedef operation";
  let description = [{
    Typedef operation servers to declare named types.
    It creates a new type symbol in the current scope to
    be referenced as NamedType later.

    This operation corresponds to `clang::TypedefDecl`.
  }];

  let extraClassDeclaration = [{
    mlir_type getDefinedType();
  }];

  let assemblyFormat = [{ $sym_name attr-dict `:` $type }];
}

def HighLevel_TypeAliasOp
  : HighLevel_Op< "alias", [TypeSymbol] >
  , Arguments<(ins StrAttr:$name, TypeAttr:$type)>
{
  let summary = "VAST type alias operation";
  let description = [{
    This operation corresponds to `clang::TypeAliasDecl`.
  }];

  let extraClassDeclaration = [{
    mlir_type getDefinedType();
  }];

  let assemblyFormat = [{ $name attr-dict `:` $type }];
}

def HighLevel_TypeOfExprOp
  : HighLevel_Op< "typeof.expr", [SingleBlock] >
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
      CArg<"maybe_builder_callback_ref", "std::nullopt">:$expr
    )>
  ];

  let assemblyFormat = [{ $name $expr `:` $type attr-dict }];
}

def HighLevel_EnumConstantOp
  : HighLevel_Op< "enum.const", [EnumConstantSymbol] >
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
      CArg< "maybe_builder_callback_ref", "std::nullopt" >:$init
    ), [{
      $_state.addAttribute("name", $_builder.getStringAttr(name));
      $_state.addAttribute("value", core::IntegerAttr::get(type, value));

      InsertionGuard guard($_builder);
      build_region($_builder, $_state, init);
    }]>,
  ];

  let assemblyFormat = [{ $name `=` $value attr-dict (`init` $init^)? }];
}

def HighLevel_EnumRegion : Region< HasOneBlock, "enum constants region" >;

def HighLevel_EnumDeclOp
  : HighLevel_Op< "enum", [NoTerminator, TypeSymbol] >
  , Arguments<(ins StrAttr:$name, OptionalAttr<TypeAttr>:$type)>
{
  let summary = "VAST enum declaration";
  let description = [{
    Enum declaration serves to declare region for enum constant declarations.
    It also defines an underlying type.
  }];

  let regions = (region HighLevel_EnumRegion:$constants);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins
      "llvm::StringRef":$name,
      "mlir::Type":$type,
      "builder_callback_ref":$constants
    )>,
    OpBuilder<(ins
      "llvm::StringRef":$name
    )>
  ];

  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    bool isComplete();
    mlir::Block& getConstantsBlock();
  }];

  let extraClassDefinition = [{
    bool $cppClass::isComplete() { return !getConstantsBlock().empty(); }

    mlir::Block& $cppClass::getConstantsBlock() { return getConstants().front(); }
  }];
}

class HighLevel_RecordLikeDeclOp< string mnemonic, list< Trait > traits = [] >
    : HighLevel_Op< mnemonic,
        !listconcat(traits, [
          NoTerminator, TypeSymbol,
          DeclareOpInterfaceMethods< AggregateTypeDefinition >
        ])
      >
    , Arguments<(ins StrAttr:$name)>
{
  let regions = (region AnyRegion:$fields);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins
      "llvm::StringRef":$name,
       CArg< "maybe_builder_callback_ref", "std::nullopt" >:$fields
    )>
  ];

  let extraClassDeclaration = [{
    bool isCompleteDefinition();
    mlir::Block& getFieldsBlock();
  }];

  let extraClassDefinition = [{
    bool $cppClass::isCompleteDefinition() { return !getFieldsBlock().empty(); }

    mlir::Block& $cppClass::getFieldsBlock() { return getFields().front(); }

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

def HighLevel_StructDeclOp
  : HighLevel_RecordLikeDeclOp< "struct", [
    DeclareOpInterfaceMethods< VAST_RecordDeclInterface >
  ] >
{
  let summary = "VAST struct declaration";
  let description = [{ VAST struct declaration }];
}

def HighLevel_UnionDeclOp
  : HighLevel_RecordLikeDeclOp< "union" >
{
  let summary = "VAST record declaration";
  let description = [{ VAST record declaration }];
}

def HighLevel_FieldDeclOp
  : HighLevel_Op< "field", [MemberVarSymbol] >
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

def HighLevel_RecordMemberOp
  : HighLevel_Op< "member" >
  , Arguments<(ins AnyType:$record, StrAttr:$name)>
  , Results<(outs AnyType:$element)>
{
  let summary = "VAST record element access operation";
  let description = [{ VAST record element access operation }];

  let assemblyFormat = [{
    $record `at` $name attr-dict `:` type($record) `->` type($element)
  }];
}

def HighLevel_CallOp
  : HighLevel_Op< "call", [
      CallOpInterface,
      DeclareOpInterfaceMethods<CallOpInterface, ["resolveCallable", "resolveCallableInTable"]>,
      VastCallOpInterface,
      DeclareOpInterfaceMethods<VastCallOpInterface, ["resolveCallable", "resolveCallableInTable"]>
    ] >
  , Arguments<(ins
      FlatSymbolRefAttr:$callee,
      Variadic<AnyType>:$argOperands)>
  , Results<(outs Variadic<AnyType>:$results)>
{
  let summary = "VAST call operation";
  let description = [{ VAST call operation }];

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder< (ins
      "mlir::FlatSymbolRefAttr":$callee,
      "mlir::TypeRange":$results,
      CArg<"mlir::ValueRange", "{}">:$operands
    ), [{
      $_state.addOperands(operands);
      $_state.addAttribute("callee", callee);
      $_state.addTypes(results);
    }]>,
    OpBuilder< (ins
      "mlir::StringAttr":$callee,
      "mlir::TypeRange":$results,
      CArg<"mlir::ValueRange", "{}">:$operands
    ), [{
      build($_builder, $_state, mlir::FlatSymbolRefAttr::get($_builder.getContext(), callee), results, operands);
    }]>,
    OpBuilder< (ins
      "llvm::StringRef":$callee,
      "mlir::TypeRange":$results,
      CArg<"mlir::ValueRange", "{}">:$operands
    ), [{
      build($_builder, $_state, mlir::SymbolRefAttr::get($_builder.getContext(), callee), results, operands);
    }]>,
    OpBuilder< (ins "FuncOp":$callee, CArg< "mlir::ValueRange", "{}" >:$operands), [{
      build($_builder, $_state, callee.getName(), callee.getFunctionType().getResults(), operands);
    }]>
  ];

  let assemblyFormat = [{
    $callee `(` $argOperands `)` attr-dict `:` functional-type( $argOperands, $results )
  }];
}

def HighLevel_IndirectCallOp
  : HighLevel_Op< "indirect_call", [
    CallOpInterface, DeclareOpInterfaceMethods<CallOpInterface>
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

def HighLevel_ExprOp
  : HighLevel_Op< "expr", [SingleBlock] >
  , Results<(outs AnyType:$result)>
{
  let summary = "VAST expression";
  let description = [{ VAST expression }];

  let regions = (region SizedRegion<1>:$subexpr);

  let skipDefaultBuilders = 1;
  let builders = [
      OpBuilder<(ins
          "Type":$type,
          "builder_callback_ref":$subexpr_builder
      )>
  ];

  let assemblyFormat = [{
     attr-dict `:` type($result) $subexpr
  }];
}

def HighLevel_ReturnOp
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
def HighLevel_DeclRefOp
  : HighLevel_Op< "ref" >
  , Arguments<(ins AnyType:$decl)>
  , Results<(outs HighLevel_LValueOf<AnyType>:$result)>
{
  let summary = "VAST variable reference declaration";
  let description = [{ VAST variable reference declaration }];

  let assemblyFormat = "$decl attr-dict `:` functional-type(operands, results)";
}

def HighLevel_FuncRefOp
  : HighLevel_Op< "funcref" >
  , Arguments<(ins FlatSymbolRefAttr:$function)>
  , Results<(outs AnyType:$result)>
{
  let summary = "VAST function reference declaration";
  let description = [{ VAST function reference declaration }];

  let assemblyFormat = "$function attr-dict `:` type($result)";
}

def HighLevel_GlobalRefOp
  : HighLevel_Op< "globref" >
  , Arguments<(ins StrAttr:$global)>
  , Results<(outs AnyType:$result)>
{
  let summary = "VAST global variable reference declaration";
  let description = [{ VAST global variable reference declaration }];

  let assemblyFormat = "$global attr-dict `:` type($result)";
}

def HighLevel_EnumRefOp
  : HighLevel_Op< "enumref" >
  , Arguments<(ins StrAttr:$value)>
  , Results<(outs AnyType:$result)>
{
  let summary = "VAST variable reference declaration";
  let description = [{ VAST variable reference declaration }];

  let assemblyFormat = "$value attr-dict `:` type($result)";
}

def HighLevel_ConstantOp
  : HighLevel_Op< "const", [ConstantLike, Pure, AllTypesMatch< ["value", "result"] >] >
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

  let extraClassDeclaration = [{
    static bool isBuildableWith(mlir_attr value, mlir_type type);
  }];

  let assemblyFormat = "$value attr-dict";
}

def HighLevel_InitializedConstantOp
  : HighLevel_Op< "const.init", [ConstantLike, Pure] >
  , Results< (outs AnyType:$result) >
{
  let summary = "VAST constant with non-trivial initialization";
  let description = [{
    InitializedConstantOp is used to represent a constant with non-trivial
    initialization. The initialization is represented by a region that
    represents initialization expression.
  }];

  let regions = (region ValueRegion:$init);

  let skipDefaultBuilders = 1;
  let builders = [
      OpBuilder<(ins
          "Type":$type,
          "builder_callback":$initbuilder
      )>
  ];

  let assemblyFormat = [{ attr-dict `:` type($result) `=` $init }];
}

def HighLevel_UnreachableOp
  : HighLevel_Op< "unreachable", [Terminator] >
{
  let summary = "VAST unreachable operation";
  let description = [{ VAST unreachable operation }];
  let assemblyFormat = "attr-dict";
}

def HighLevel_CompoundLiteralOp
  : HighLevel_Op< "compound_literal" >
  , Results< (outs AnyType:$result) >
{
  let summary = "VAST compound literals operation";
  let description = [{
    Creates a compound literal that represents an unnamed object and is
    initialzied by initializer-list.

    Is used to represetnt `clang::CompoundLiteralExpr`.
  }];

  let regions = (region ValueRegion:$init);

  let skipDefaultBuilders = 1;
  let builders = [
      OpBuilder<(ins
          "Type":$type,
          "builder_callback":$initbuilder
      )>
  ];

  let assemblyFormat = [{ attr-dict `:` type($result) `=` $init }];
}

class HighLevel_CastKindAttr< string name, int val > : I64EnumAttrCase< name, val > {}

class HighLevel_CastKindList< string name, string summary, list< HighLevel_CastKindAttr > cases >
  : I64EnumAttr< name, summary, cases > {}

def HighLevel_Dependent : HighLevel_CastKindAttr<"Dependent", 0>;
def HighLevel_BitCast : HighLevel_CastKindAttr<"BitCast", 1>;
def HighLevel_LValueBitCast : HighLevel_CastKindAttr<"LValueBitCast", 2>;
def HighLevel_LValueToRValueBitCast : HighLevel_CastKindAttr<"LValueToRValueBitCast", 3>;
def HighLevel_LValueToRValue : HighLevel_CastKindAttr<"LValueToRValue", 4>;

def HighLevel_NoOp : HighLevel_CastKindAttr<"NoOp", 5>;

def HighLevel_BaseToDerived : HighLevel_CastKindAttr<"BaseToDerived", 6>;
def HighLevel_DerivedToBase : HighLevel_CastKindAttr<"DerivedToBase", 7>;
def HighLevel_UncheckedDerivedToBase : HighLevel_CastKindAttr<"UncheckedDerivedToBase", 8>;
def HighLevel_Dynamic : HighLevel_CastKindAttr<"Dynamic", 9>;
def HighLevel_ToUnion : HighLevel_CastKindAttr<"ToUnion", 10>;

def HighLevel_ArrayToPointerDecay : HighLevel_CastKindAttr<"ArrayToPointerDecay", 11>;
def HighLevel_FunctionToPointerDecay : HighLevel_CastKindAttr<"FunctionToPointerDecay", 12>;
def HighLevel_NullToPointer : HighLevel_CastKindAttr<"NullToPointer", 13>;
def HighLevel_NullToMemberPointer : HighLevel_CastKindAttr<"NullToMemberPointer", 14>;
def HighLevel_BaseToDerivedMemberPointer : HighLevel_CastKindAttr<"BaseToDerivedMemberPointer", 15>;
def HighLevel_DerivedToBaseMemberPointer : HighLevel_CastKindAttr<"DerivedToBaseMemberPointer", 16>;
def HighLevel_MemberPointerToBoolean : HighLevel_CastKindAttr<"MemberPointerToBoolean", 17>;
def HighLevel_ReinterpretMemberPointer : HighLevel_CastKindAttr<"ReinterpretMemberPointer", 18>;
def HighLevel_UserDefinedConversion : HighLevel_CastKindAttr<"UserDefinedConversion", 19>;
def HighLevel_ConstructorConversion : HighLevel_CastKindAttr<"ConstructorConversion", 20>;

def HighLevel_IntegralToPointer : HighLevel_CastKindAttr<"IntegralToPointer", 21>;
def HighLevel_PointerToIntegral : HighLevel_CastKindAttr<"PointerToIntegral", 22>;
def HighLevel_PointerToBoolean  : HighLevel_CastKindAttr<"PointerToBoolean", 23>;

def HighLevel_ToVoid : HighLevel_CastKindAttr<"ToVoid", 24>;

def HighLevel_VectorSplat : HighLevel_CastKindAttr<"VectorSplat", 25>;

def HighLevel_IntegralCast : HighLevel_CastKindAttr<"IntegralCast", 26>;
def HighLevel_IntegralToBoolean : HighLevel_CastKindAttr<"IntegralToBoolean", 27>;
def HighLevel_IntegralToFloating : HighLevel_CastKindAttr<"IntegralToFloating", 28>;
def HighLevel_FloatingToFixedPoint : HighLevel_CastKindAttr<"FloatingToFixedPoint", 29>;
def HighLevel_FixedPointToFloating : HighLevel_CastKindAttr<"FixedPointToFloating", 30>;
def HighLevel_FixedPointCast : HighLevel_CastKindAttr<"FixedPointCast", 31>;
def HighLevel_FixedPointToIntegral : HighLevel_CastKindAttr<"FixedPointToIntegral", 32>;
def HighLevel_IntegralToFixedPoint : HighLevel_CastKindAttr<"IntegralToFixedPoint", 33>;
def HighLevel_FixedPointToBoolean : HighLevel_CastKindAttr<"FixedPointToBoolean", 34>;
def HighLevel_FloatingToIntegral : HighLevel_CastKindAttr<"FloatingToIntegral", 35>;
def HighLevel_FloatingToBoolean : HighLevel_CastKindAttr<"FloatingToBoolean", 36>;
def HighLevel_BooleanToSignedIntegral : HighLevel_CastKindAttr<"BooleanToSignedIntegral", 37>;
def HighLevel_FloatingCast : HighLevel_CastKindAttr<"FloatingCast", 38>;

def HighLevel_CPointerToObjCPointerCast : HighLevel_CastKindAttr<"CPointerToObjCPointerCast", 39>;
def HighLevel_BlockPointerToObjCPointerCast : HighLevel_CastKindAttr<"BlockPointerToObjCPointerCast", 40>;
def HighLevel_AnyPointerToBlockPointerCast : HighLevel_CastKindAttr<"AnyPointerToBlockPointerCast", 41>;
def HighLevel_ObjCObjectLValueCast : HighLevel_CastKindAttr<"ObjCObjectLValueCast", 42>;

def HighLevel_FloatingRealToComplex : HighLevel_CastKindAttr<"FloatingRealToComplex", 43>;
def HighLevel_FloatingComplexToReal : HighLevel_CastKindAttr<"FloatingComplexToReal", 44>;
def HighLevel_FloatingComplexToBoolean : HighLevel_CastKindAttr<"FloatingComplexToBoolean", 45>;
def HighLevel_FloatingComplexCast : HighLevel_CastKindAttr<"FloatingComplexCast", 46>;
def HighLevel_FloatingComplexToIntegralComplex : HighLevel_CastKindAttr<"FloatingComplexToIntegralComplex", 47>;
def HighLevel_IntegralRealToComplex : HighLevel_CastKindAttr<"IntegralRealToComplex", 48>;
def HighLevel_IntegralComplexToReal : HighLevel_CastKindAttr<"IntegralComplexToReal", 49>;
def HighLevel_IntegralComplexToBoolean : HighLevel_CastKindAttr<"IntegralComplexToBoolean", 50>;
def HighLevel_IntegralComplexCast : HighLevel_CastKindAttr<"IntegralComplexCast", 51>;
def HighLevel_IntegralComplexToFloatingComplex : HighLevel_CastKindAttr<"IntegralComplexToFloatingComplex", 52>;

def HighLevel_ARCProduceObject : HighLevel_CastKindAttr<"ARCProduceObject", 53>;
def HighLevel_ARCConsumeObject : HighLevel_CastKindAttr<"ARCConsumeObject", 54>;
def HighLevel_ARCReclaimReturnedObject : HighLevel_CastKindAttr<"ARCReclaimReturnedObject", 55>;
def HighLevel_ARCExtendBlockObject : HighLevel_CastKindAttr<"ARCExtendBlockObject", 56>;

def HighLevel_AtomicToNonAtomic : HighLevel_CastKindAttr<"AtomicToNonAtomic", 57>;
def HighLevel_NonAtomicToAtomic : HighLevel_CastKindAttr<"NonAtomicToAtomic", 58>;

def HighLevel_CopyAndAutoreleaseBlockObject : HighLevel_CastKindAttr<"CopyAndAutoreleaseBlockObject", 59>;
def HighLevel_BuiltinFnToFnPtr : HighLevel_CastKindAttr<"BuiltinFnToFnPtr", 60>;

def HighLevel_ZeroToOCLOpaqueType : HighLevel_CastKindAttr<"ZeroToOCLOpaqueType", 61>;
def HighLevel_AddressSpaceConversion : HighLevel_CastKindAttr<"AddressSpaceConversion", 62>;
def HighLevel_IntToOCLSampler : HighLevel_CastKindAttr<"IntToOCLSampler", 63>;

def HighLevel_MatrixCast : HighLevel_CastKindAttr<"MatrixCast", 64>;

let cppNamespace = "::vast::hl" in
def HighLevel_CastKind : HighLevel_CastKindList< "CastKind", "cast kind", [
  HighLevel_Dependent,
  HighLevel_BitCast,
  HighLevel_LValueBitCast,
  HighLevel_LValueToRValueBitCast,
  HighLevel_LValueToRValue,

  HighLevel_NoOp,

  HighLevel_BaseToDerived,
  HighLevel_DerivedToBase,
  HighLevel_UncheckedDerivedToBase,
  HighLevel_Dynamic,
  HighLevel_ToUnion,

  HighLevel_ArrayToPointerDecay,
  HighLevel_FunctionToPointerDecay,
  HighLevel_NullToPointer,
  HighLevel_NullToMemberPointer,
  HighLevel_BaseToDerivedMemberPointer,
  HighLevel_DerivedToBaseMemberPointer,
  HighLevel_MemberPointerToBoolean,
  HighLevel_ReinterpretMemberPointer,
  HighLevel_UserDefinedConversion,
  HighLevel_ConstructorConversion,

  HighLevel_IntegralToPointer,
  HighLevel_PointerToIntegral,
  HighLevel_PointerToBoolean,

  HighLevel_ToVoid,

  HighLevel_VectorSplat,

  HighLevel_IntegralCast,
  HighLevel_IntegralToBoolean,
  HighLevel_IntegralToFloating,
  HighLevel_FloatingToFixedPoint,
  HighLevel_FixedPointToFloating,
  HighLevel_FixedPointCast,
  HighLevel_FixedPointToIntegral,
  HighLevel_IntegralToFixedPoint,
  HighLevel_FixedPointToBoolean,
  HighLevel_FloatingToIntegral,
  HighLevel_FloatingToBoolean,
  HighLevel_BooleanToSignedIntegral,
  HighLevel_FloatingCast,

  HighLevel_CPointerToObjCPointerCast,
  HighLevel_BlockPointerToObjCPointerCast,
  HighLevel_AnyPointerToBlockPointerCast,
  HighLevel_ObjCObjectLValueCast,

  HighLevel_FloatingRealToComplex,
  HighLevel_FloatingComplexToReal,
  HighLevel_FloatingComplexToBoolean,
  HighLevel_FloatingComplexCast,
  HighLevel_FloatingComplexToIntegralComplex,
  HighLevel_IntegralRealToComplex,
  HighLevel_IntegralComplexToReal,
  HighLevel_IntegralComplexToBoolean,
  HighLevel_IntegralComplexCast,
  HighLevel_IntegralComplexToFloatingComplex,

  HighLevel_ARCProduceObject,
  HighLevel_ARCConsumeObject,
  HighLevel_ARCReclaimReturnedObject,
  HighLevel_ARCExtendBlockObject,

  HighLevel_AtomicToNonAtomic,
  HighLevel_NonAtomicToAtomic,

  HighLevel_CopyAndAutoreleaseBlockObject,
  HighLevel_BuiltinFnToFnPtr,

  HighLevel_ZeroToOCLOpaqueType,
  HighLevel_AddressSpaceConversion,
  HighLevel_IntToOCLSampler,

  HighLevel_MatrixCast
] >;

class HighLevel_CastOp< string mnemonic, list< Trait > traits = [] >
    : HighLevel_Op< mnemonic, traits >
    , Arguments< (ins AnyType:$value, HighLevel_CastKind:$kind) >
    , Results< (outs AnyType:$result) >
{
    let summary = "VAST cast operation";
    let description = [{ VAST cast operation }];

    let assemblyFormat = "$value $kind attr-dict `:` type($value) `->` type($result)";
}

def HighLevel_ImplicitCastOp   : HighLevel_CastOp< "implicit_cast" > {
    let hasFolder = 1;
}

def HighLevel_CStyleCastOp     : HighLevel_CastOp< "cstyle_cast" >;
def HighLevel_BuiltinBitCastOp : HighLevel_CastOp< "builtin_bitcast" >;

class HighLevel_IsPointerCompatible< string arg >
  : PredTypeTrait< "type can be added/subtracted to a pointer",
   Or < [
    And< [
      IsIntegral< arg >.predicate,
      Neg< IsShort< arg >.predicate >,
      Neg< IsChar< arg >.predicate >,
      Neg< IsBool< arg >.predicate >
    ] >,
    IsBuiltinType< arg >.predicate
  ] >
>;

class HighLevel_IsPointerAdditive< string lhs, string rhs, string res >
  : PredOpTrait< "is an additive operation on pointers",
    And< [
      Or< [
        And< [
          IsPointer< lhs >.predicate,
          HighLevel_IsPointerCompatible< rhs >.predicate
        ] >,
        And< [
          HighLevel_IsPointerCompatible< lhs >.predicate,
          IsPointer< rhs >.predicate
        ] >
      ] >,
      IsPointer< res >.predicate
    ] >
>;

class HighLevel_PtrDiffLike< string lhs, string rhs, string res >
  : PredOpTrait< "is a subtraction of pointers",
    And< [
      IsPointer< lhs >.predicate,
      IsPointer< rhs >.predicate,
      Or< [
        IsIntegral< res >.predicate,
        IsBuiltinType< res >.predicate
      ] >
    ] >
>;

class HighLevel_IsAdditive< string lhs, string rhs, string res >
  : PredOpTrait< "is an additive operation (types match or are ptr and integral)",
    Or< [
        HighLevel_TypesMatchOrUnresolved< [lhs, rhs, res] >.predicate,
        HighLevel_IsPointerAdditive< lhs, rhs, res >.predicate,
    ]>
>;

class HighLevel_IsSub< string lhs, string rhs, string res >
: PredOpTrait< "is a subtraction operation (types match or are ptr and integral)",
    Or< [
        HighLevel_IsAdditive< lhs, rhs, res >.predicate,
        HighLevel_PtrDiffLike< lhs, rhs, res >.predicate
    ] >
>;

class HighLevel_ArithBinOp< string mnemonic, list< Trait > traits = [] >
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

    let hasFolder = 1;
}

class HighLevel_StandardArithBinOp< string mnemonic, list< Trait > traits = [] >
    : HighLevel_ArithBinOp< mnemonic,
        !listconcat(traits, [HighLevel_TypesMatchOrUnresolved< ["lhs", "rhs", "result"] >])
      >;

class HighLevel_FloatArithBinOp< string mnemonic, list< Trait > traits = [] >
    : HighLevel_ArithBinOp< mnemonic, traits >
{
    let hasVerifier = 1;
    let extraClassDefinition = [{
        logical_result $cppClass::verify() {
            return verify_float_arith_op(this->getOperation());
        }
    }];
}

def HighLevel_AddIOp : HighLevel_ArithBinOp< "add", [Commutative, HighLevel_IsAdditive< "lhs", "rhs", "result" >] >;
def HighLevel_SubIOp : HighLevel_ArithBinOp< "sub", [HighLevel_IsSub< "lhs", "rhs", "result" >] >;

def HighLevel_MulIOp : HighLevel_StandardArithBinOp<  "mul", [Commutative] >;
def HighLevel_DivSOp : HighLevel_StandardArithBinOp< "sdiv" >;
def HighLevel_DivUOp : HighLevel_StandardArithBinOp< "udiv" >;
def HighLevel_RemSOp : HighLevel_StandardArithBinOp< "srem" >;
def HighLevel_RemUOp : HighLevel_StandardArithBinOp< "urem" >;

def HighLevel_BinXorOp : HighLevel_StandardArithBinOp< "bin.xor" >;
def HighLevel_BinOrOp  : HighLevel_StandardArithBinOp<  "bin.or" >;
def HighLevel_BinAndOp : HighLevel_StandardArithBinOp< "bin.and" >;

def HighLevel_AddFOp : HighLevel_FloatArithBinOp< "fadd" >;
def HighLevel_SubFOp : HighLevel_FloatArithBinOp< "fsub" >;
def HighLevel_MulFOp : HighLevel_FloatArithBinOp< "fmul" >;
def HighLevel_DivFOp : HighLevel_FloatArithBinOp< "fdiv" >;
def HighLevel_RemFOp : HighLevel_FloatArithBinOp< "frem" >;

class HighLevel_ComplexUnaryOp< string mnemonic, list< Trait > traits = [] >
  : HighLevel_Op< mnemonic, traits >
  , Arguments< (ins AnyType:$arg) >
  , Results< (outs AnyType:$result) >
{
  let assemblyFormat = [{ $arg attr-dict `:` functional-type(operands, results) }];
}

def HighLevel_RealOp : HighLevel_ComplexUnaryOp< "real" >;
def HighLevel_ImagOp : HighLevel_ComplexUnaryOp< "imag" >;

class HighLevel_LogicBinOp< string mnemonic, list< Trait > traits = [] >
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
            "builder_callback_ref":$lhsbuilder,
            "builder_callback_ref":$rhsbuilder
        )>
    ];

    let assemblyFormat = [{ $lhs`,` $rhs attr-dict `:` type(results) }];

    let hasFolder = 1;
}

def HighLevel_BinLAndOp : HighLevel_LogicBinOp< "bin.land" >;
def HighLevel_BinLOrOp  : HighLevel_LogicBinOp<  "bin.lor" >;

def HighLevel_BinComma
  : HighLevel_Op< "bin.comma" >
  , Arguments< (ins AnyType:$lhs, AnyType:$rhs) >
  , Results< (outs AnyType:$result) >
{
    let summary = "VAST binary operation";

    let assemblyFormat = [{ $lhs `,` $rhs attr-dict `:` functional-type(operands, results) }];

    let hasFolder = 1;
}

class HighLevel_ShiftOp< string mnemonic, list< Trait > traits = [] >
    : HighLevel_Op< mnemonic, !listconcat(traits, [
        HighLevel_TypesMatchOrUnresolved<["lhs", "result"]>
    ]) >
    , Arguments<(ins HighLevel_IntegerLikeType:$lhs, HighLevel_IntegerLikeType:$rhs)>
    , Results<(outs HighLevel_IntegerLikeType:$result)>
{
    let summary = "VAST binary shift operation";
    let description = [{
        High-level binary shift operation. This operation takes two operands
        and returns one result.

        The custom assembly form of the operation is as follows:

        %result = <op> %lhs, %rhs  : functional-type(operands, results)
    }];

    let assemblyFormat = [{ $lhs `,` $rhs attr-dict `:` functional-type(operands, results) }];

    let hasFolder = 1;
}

def HighLevel_BinShlOp : HighLevel_ShiftOp<"bin.shl" >;
def HighLevel_BinLShrOp : HighLevel_ShiftOp<"bin.lshr" >;
def HighLevel_BinAShrOp : HighLevel_ShiftOp<"bin.ashr" >;

class HighLevel_IsLValuePointer< string arg >
  : PredOpTrait< "lvalue is of an integer type",
    CPred< "mlir::isa< LValueType >($" # arg # ".getType()) && mlir::isa< hl::PointerType >(mlir::cast< LValueType >($" # arg # ".getType()).getElementType())" >
>;

class HighLevel_IsCompoundPtrAdditive< string lhs, string rhs, string res >
  : PredOpTrait< "is an additive operation on pointers",
    And< [HighLevel_IsLValuePointer< lhs >.predicate,
          IsInteger< rhs>.predicate,
          IsPointer< res >.predicate ] >
>;

class HighLevel_IsCompoundAdditive< string lhs, string rhs, string res >
  : PredOpTrait< "is an additive operation (types match or are ptr and integral)",
    Or<[
        HighLevel_IsCompoundPtrAdditive< lhs, rhs, res >.predicate,
        CPred< "mlir::isa< LValueType >($" # lhs # ".getType()) && mlir::cast< LValueType >($" # lhs # ".getType()).getElementType() == $" # res # ".getType()" >,
        HighLevel_TypesMatchOrUnresolved< [lhs, rhs] >.predicate
    ]>
>;

class HighLevel_CompoundAssignOpTemplate< string mnemonic, TypeConstraint Type, list< Trait > traits = [] >
    : HighLevel_Op< mnemonic, !listconcat(traits, [
        TypesMatchWith<
          "underlying destination type match result type",
          "dst", "result",
          "mlir::isa< ElementTypeInterface >($_self) ? mlir::cast< ElementTypeInterface >($_self).getElementType() : mlir::Type()"
        >
    ])>
    // FIXME: add argument constraints and infer return type
    , Arguments<(ins HighLevel_LValueOrType< Type >:$src, AnyType:$dst)>
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

class HighLevel_CompoundAssignOp< string mnemonic, list< Trait > traits = [] >
    : HighLevel_CompoundAssignOpTemplate< mnemonic, AnyType, traits > {}

def HighLevel_AssignOp     : HighLevel_CompoundAssignOp< "assign" >;
def HighLevel_AddIAssignOp : HighLevel_CompoundAssignOp< "assign.add", [HighLevel_IsCompoundAdditive< "dst", "src", "result" >] >;
def HighLevel_AddFAssignOp : HighLevel_CompoundAssignOp< "assign.fadd" >;
def HighLevel_SubIAssignOp : HighLevel_CompoundAssignOp< "assign.sub", [HighLevel_IsCompoundAdditive< "dst", "src", "result" >] >;
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
    : HighLevel_CompoundAssignOpTemplate< mnemonic, HighLevel_IntegerLikeType, traits > {}

def HighLevel_BinShlAssignOp : HighLevel_ShiftAssignOp< "assign.bin.shl" >;

def HighLevel_BinLShrAssignOp : HighLevel_ShiftAssignOp< "assign.bin.lshr" >;
def HighLevel_BinAShrAssignOp : HighLevel_ShiftAssignOp< "assign.bin.ashr" >;

class HighLevel_PredicateAttr< string name, int val > : I64EnumAttrCase< name, val > {}

class HighLevel_PredicateList< string name, string summary, list< HighLevel_PredicateAttr > cases >
  : I64EnumAttr< name, summary, cases > {}

def HighLevel_ICmpPredicateEQ  : HighLevel_PredicateAttr< "eq", 0>;
def HighLevel_ICmpPredicateNE  : HighLevel_PredicateAttr< "ne", 1>;
def HighLevel_ICmpPredicateSLT : HighLevel_PredicateAttr<"slt", 2>;
def HighLevel_ICmpPredicateSLE : HighLevel_PredicateAttr<"sle", 3>;
def HighLevel_ICmpPredicateSGT : HighLevel_PredicateAttr<"sgt", 4>;
def HighLevel_ICmpPredicateSGE : HighLevel_PredicateAttr<"sge", 5>;
def HighLevel_ICmpPredicateULT : HighLevel_PredicateAttr<"ult", 6>;
def HighLevel_ICmpPredicateULE : HighLevel_PredicateAttr<"ule", 7>;
def HighLevel_ICmpPredicateUGT : HighLevel_PredicateAttr<"ugt", 8>;
def HighLevel_ICmpPredicateUGE : HighLevel_PredicateAttr<"uge", 9>;

let cppNamespace = "::vast::hl" in
def HighLevel_Predicate : HighLevel_PredicateList< "Predicate", "comparison predicate", [
  HighLevel_ICmpPredicateEQ,  HighLevel_ICmpPredicateNE,
  HighLevel_ICmpPredicateSLT, HighLevel_ICmpPredicateSLE,
  HighLevel_ICmpPredicateSGT, HighLevel_ICmpPredicateSGE,
  HighLevel_ICmpPredicateULT, HighLevel_ICmpPredicateULE,
  HighLevel_ICmpPredicateUGT, HighLevel_ICmpPredicateUGE
] >;

class HighLevel_IsPointerCmp< string lhs, string rhs >
  : PredOpTrait< "is an additive operation on pointers",
    And< [
      IsPointer< lhs >.predicate,
      IsPointer< rhs >.predicate
    ] >
>;

class HighLevel_IsIntegralCmp< string lhs, string rhs >
  : PredOpTrait< "is an additive operation on pointers",
    And< [
      IsIntegral< lhs >.predicate,
      IsIntegral< rhs >.predicate
    ] >
>;

class HighLevel_IsCmp< string lhs, string rhs >
  : PredOpTrait< "is an additive operation (types match or are ptr and integral)",
    Or<[
        HighLevel_IsIntegralCmp< lhs, rhs >.predicate,
        HighLevel_IsPointerCmp< lhs, rhs >.predicate
    ]>
>;

def HighLevel_CmpOp
  : HighLevel_Op< "cmp" >
  , Arguments<(ins HighLevel_Predicate:$predicate, AnyType:$lhs, AnyType:$rhs)>
  , Results<(outs HighLevel_IntOrBoolType:$result)>
  , HighLevel_IsCmp< "lhs", "rhs" >
{
  let summary = "VAST comparison operation";
  let description = [{ VAST comparison operation }];

  let assemblyFormat = "$predicate $lhs `,` $rhs  attr-dict `:` type(operands) `->` type($result)";
}

class HighLevel_FPredicateAttr< string name, int val > : I64EnumAttrCase< name, val > {}

class HighLevel_FPredicateList< string name, string summary, list< HighLevel_FPredicateAttr > cases >
  : I64EnumAttr< name, summary, cases > {}

def HighLevel_FPredicateFalse : HighLevel_FPredicateAttr<"ffalse", 0>;
def HighLevel_FPredicateOEQ   : HighLevel_FPredicateAttr<"oeq",    1>;
def HighLevel_FPredicateOGT   : HighLevel_FPredicateAttr<"ogt",    2>;
def HighLevel_FPredicateOGE   : HighLevel_FPredicateAttr<"oge",    3>;
def HighLevel_FPredicateOLT   : HighLevel_FPredicateAttr<"olt",    4>;
def HighLevel_FPredicateOLE   : HighLevel_FPredicateAttr<"ole",    5>;
def HighLevel_FPredicateONE   : HighLevel_FPredicateAttr<"one",    6>;
def HighLevel_FPredicateORD   : HighLevel_FPredicateAttr<"ord",    7>;
def HighLevel_FPredicateUNO   : HighLevel_FPredicateAttr<"uno",    8>;
def HighLevel_FPredicateUEQ   : HighLevel_FPredicateAttr<"ueq",    9>;
def HighLevel_FPredicateUGT   : HighLevel_FPredicateAttr<"ugt",   10>;
def HighLevel_FPredicateUGE   : HighLevel_FPredicateAttr<"uge",   11>;
def HighLevel_FPredicateULT   : HighLevel_FPredicateAttr<"ult",   12>;
def HighLevel_FPredicateULE   : HighLevel_FPredicateAttr<"ule",   13>;
def HighLevel_FPredicateUNE   : HighLevel_FPredicateAttr<"une",   14>;
def HighLevel_FPredicateTrue  : HighLevel_FPredicateAttr<"ftrue", 15>;

let cppNamespace = "::vast::hl" in
def HighLevel_FPredicate : HighLevel_FPredicateList< "FPredicate", "floating point comparison predicate", [
  HighLevel_FPredicateFalse, HighLevel_FPredicateOEQ, HighLevel_FPredicateOGT, HighLevel_FPredicateOGE,
  HighLevel_FPredicateOLT, HighLevel_FPredicateOLE, HighLevel_FPredicateONE, HighLevel_FPredicateORD,
  HighLevel_FPredicateUNO, HighLevel_FPredicateUEQ, HighLevel_FPredicateUGT, HighLevel_FPredicateUGE,
  HighLevel_FPredicateULT, HighLevel_FPredicateULE, HighLevel_FPredicateUNE, HighLevel_FPredicateTrue
] >;

def HighLevel_FCmpOp
  : HighLevel_Op< "fcmp" >
  , Arguments<(ins HighLevel_FPredicate:$predicate, AnyType:$lhs, AnyType:$rhs)>
  , Results<(outs HighLevel_IntOrBoolType:$result)>
{
  let summary = "VAST flaoting point comparison operation";
  let description = [{ VAST floating point comparison operation }];

  let hasVerifier = 1;

  let assemblyFormat = "$predicate $lhs `,` $rhs  attr-dict `:` type(operands) `->` type($result)";
}

class HighLevel_UnInplaceOp< string mnemonic, list< Trait > traits = [] >
    : HighLevel_Op< mnemonic, !listconcat(traits, [
        TypesMatchWith<
          "underlying argument type match result type",
          "arg", "result",
          "$_self.cast< LValueType >().getElementType()"
        >
      ]) >
    , Arguments<(ins HighLevel_LValueOf<AnyType>:$arg)>
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

def HighLevel_PostIncOp : HighLevel_UnInplaceOp< "post.inc" >;
def HighLevel_PostDecOp : HighLevel_UnInplaceOp< "post.dec" >;
def HighLevel_PreIncOp  : HighLevel_UnInplaceOp<  "pre.inc" >;
def HighLevel_PreDecOp  : HighLevel_UnInplaceOp<  "pre.dec" >;

class HighLevel_TypePreservingUnOp< string mnemonic, list< Trait > traits = [] >
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

def HighLevel_PlusOp  : HighLevel_TypePreservingUnOp< "plus" >;
def HighLevel_MinusOp : HighLevel_TypePreservingUnOp< "minus" >;
def HighLevel_NotOp   : HighLevel_TypePreservingUnOp< "not" >;

class HighLevel_LogicalUnOp< string mnemonic, list< Trait > traits = [] >
    : HighLevel_Op< mnemonic, traits >
    , Arguments< (ins AnyType:$arg) >
    , Results< (outs HighLevel_IntOrBoolType:$res) >
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

def HighLevel_LNotOp  : HighLevel_LogicalUnOp< "lnot", [] >;

def HighLevel_AddressOf
  : HighLevel_Op< "addressof" >
  , Arguments<(ins AnyType:$value)>
  , Results<(outs AnyType:$result)>
{
    let summary = "VAST addressof operation";
    let description = [{ VAST addressof operation }];

    let assemblyFormat = [{ $value attr-dict `:` type($value) `->` type($result) }];
}

def HighLevel_Deref
  : HighLevel_Op< "deref" >
  , Arguments<(ins AnyType:$addr)>
  , Results<(outs HighLevel_LValueOf<AnyType>:$result)>
{
    let summary = "VAST deref operation";
    let description = [{ VAST deref operation }];

    let assemblyFormat = [{ $addr attr-dict `:` type($addr) `->` type($result) }];
}

def HighLevel_AddrLabelExpr
  : HighLevel_Op< "labeladdr" >
  , Arguments<(ins HighLevel_LabelType:$label)>
  , Results<(outs HighLevel_PointerLikeType:$result)>
{
    let summary = "VAST address of label extension";
    let description = [{ VAST address of label extension }];

    let assemblyFormat = [{ $label attr-dict `:` type($result) }];
}

def HighLevel_InitListExpr
  : HighLevel_Op< "initlist" >
  , Arguments<(ins Variadic<AnyType>:$elements)>
  , Results<(outs Variadic<AnyType>)>
{
  let summary = "VAST initializer list expression";
  let description = [{ VAST initializer list expression }];

  let assemblyFormat = "$elements attr-dict `:` functional-type($elements, results)";
}

def HighLevel_SubscriptOp
  : HighLevel_Op< "subscript" >
  , Arguments<(ins
      HighLevel_LValueOrType<HighLevel_SubscriptableType>:$array,
      HighLevel_IntegerLikeType:$index)>
  , Results<(outs HighLevel_LValueOf<AnyType>:$result)>
{
  let summary = "VAST array subscript operator";
  let description = [{ VAST array subscript operator }];

  let assemblyFormat = [{
    $array `at` ` ` `[` $index `:` type($index) `]` attr-dict
      `:` type($array) `->` type($result)
  }];
}

class HighLevel_TypeTraitOp< string mnemonic, list< Trait > traits = [] >
  : HighLevel_Op< mnemonic, !listconcat([Pure], traits) >
  , Arguments<(ins TypeAttr:$arg)>
  , Results<(outs HighLevel_IntegerLikeType:$result)>
{
  let assemblyFormat = [{ $arg attr-dict `->` type($result) }];
}

def HighLevel_SizeOfTypeOp
  : HighLevel_TypeTraitOp< "sizeof.type", [Pure] >
{
  let summary = "VAST type sizeof operator";
  let description = [{ Corresponds to sizeof(<type>) expression. }];

  let extraClassDeclaration = [{ std::size_t getValue(); }];

  let hasFolder = 1;
}

class HighLevel_AlignOfTypeOpBase< string mnemonic, list< Trait > traits = [] >
    : HighLevel_TypeTraitOp< mnemonic, traits >
{
  let summary = "VAST type alignof operator";
  let description = [{ VAST type alignof operator }];
}

def HighLevel_AlignOfTypeOp          : HighLevel_AlignOfTypeOpBase< "alignof.type" >;
def HighLevel_PreferredAlignOfTypeOp : HighLevel_AlignOfTypeOpBase< "preferred_alignof.type" >;

class HighLevel_ExprTraitOp< string mnemonic, list< Trait > traits = [] >
  : HighLevel_Op< mnemonic, !listconcat([Pure], traits) >
  , Results<(outs HighLevel_IntegerLikeType:$result)>
{
  let regions = (region SizedRegion<1>:$expr);

  let skipDefaultBuilders = 1;
  let builders = [
      OpBuilder<(ins
          "Type":$type,
          "builder_callback_ref":$exprBuilder
      )>
  ];

  let assemblyFormat = [{ attr-dict `->` type($result) $expr }];
}

def HighLevel_SizeOfExprOp
  : HighLevel_ExprTraitOp< "sizeof.expr" >
{
  let summary = "VAST expr sizeof operator";
  let description = [{ VAST expr sizeof operator }];

  let extraClassDeclaration = [{ std::size_t getValue(); }];
  let hasFolder = 1;
}

class HighLevel_AlignOfExprOpBase< string mnemonic, list< Trait > traits = [] >
    : HighLevel_ExprTraitOp< mnemonic, traits >
{
  let summary = "VAST expr alignof operator";
  let description = [{ VAST expr alignof operator }];
}

def HighLevel_AlignOfExprOp          : HighLevel_AlignOfExprOpBase< "alignof.expr" >;
def HighLevel_PreferredAlignOfExprOp : HighLevel_AlignOfExprOpBase< "preferred_alignof.expr" >;

def HighLevel_OffsetOfNodeAttr
  : HighLevel_Attr< "OffsetOfNode", "offset_of_node" > {
    let parameters = (ins
      "std::variant< unsigned int, ::mlir::StringAttr >":$value
      //TODO base class specifier
    );

  let summary = "VAST expr offsetofnode ";
  let description = [{
      This attribute corresponds to OffsetOfExprNode used by Clang AST.
      It can containt either an index into the list of expressions(regions) that index arrays
      or a field identifier or base class specifier.
  }];

  let extraClassDeclaration = [{
    static mlir_attr parseAttrWithInteger(mlir::AsmParser &parser);
    static mlir_attr parseAttrWithString(mlir::AsmParser &parser);
  }];

  let hasCustomAssemblyFormat = 1;
}

def HighLevel_OffsetOfExprOp
  : HighLevel_Op< "offsetof.expr" >
  , Arguments< (ins TypeAttr:$source, ArrayAttr:$components) >
  , Results< (outs AnyType:$result) >
{
  let summary = "VAST expr offsetof operator";
  let description = [{
      VAST expr offsetof operator.
      The operation holds an array of components - identifiers or indexes of array indexing expressions
      and an array of regions - each corresponding to one indexing expression.
  }];
  let regions = (region VariadicRegion< ValueRegion >:$array_index_exprs);

  let skipDefaultBuilders = 1;
  let builders = [
      OpBuilder< (ins
        "Type":$type,
        "mlir::TypeAttr":$source,
        "mlir::ArrayAttr":$components,
        "const std::vector< builder_callback > &":$builders
      ) >,
      OpBuilder< (ins
        "Type":$type,
        "mlir::Type":$source,
        "mlir::ArrayAttr":$components,
        "const std::vector< builder_callback > &":$builders
      ) >
  ];

  let assemblyFormat = [{ attr-dict `type` `:` $source `,` `member` `:` $components `:` type($result) $array_index_exprs }];
}

def HighLevel_StmtExprOp
  : HighLevel_Op< "stmt.expr", [SingleBlock, DeclareOpInterfaceMethods<RegionKindInterface>] >
  , Results<(outs AnyType:$result)>
{
  let summary = "VAST statement expression";
  let description = [{ VAST statement expression }];

  let regions = (region SizedRegion<1>:$substmt);

  let skipDefaultBuilders = 1;
  let builders = [
      OpBuilder<(ins
          "Type":$type,
          "builder_callback_ref":$subexpr_builder
      )>
  ];

  let assemblyFormat = [{
     attr-dict `:` type($result) $substmt
  }];
}

class HighLevel_IdentKindAttr< string name, int val > : I64EnumAttrCase< name, val > {}

class HighLevel_IdentKindList< string name, string summary, list< HighLevel_IdentKindAttr > cases >
  : I64EnumAttr< name, summary, cases > {}

def HighLevel_Func : HighLevel_IdentKindAttr<"Func", 0>;
def HighLevel_Function : HighLevel_IdentKindAttr<"Function", 1>;
def HighLevel_LFunction : HighLevel_IdentKindAttr<"LFunction", 2>;
def HighLevel_FuncDName : HighLevel_IdentKindAttr<"FuncDName", 3>;
def HighLevel_FuncSig : HighLevel_IdentKindAttr<"FuncSig", 4>;
def HighLevel_LFuncSig : HighLevel_IdentKindAttr<"LFuncSig", 5>;
def HighLevel_PrettyFunction : HighLevel_IdentKindAttr<"PrettyFunction", 6>;
def HighLevel_PrettyFunctionNoVirtual : HighLevel_IdentKindAttr<"PrettyFunctionNoVirtual", 7>;

let cppNamespace = "::vast::hl" in
def HighLevel_IdentKind : HighLevel_IdentKindList< "IdentKind", "ident kind", [
    HighLevel_Func,
    HighLevel_Function,
    HighLevel_LFunction,
    HighLevel_FuncDName,
    HighLevel_FuncSig,
    HighLevel_LFuncSig,
    HighLevel_PrettyFunction,
    HighLevel_PrettyFunctionNoVirtual
] >;

def HighLevel_PredefinedExpr
  : HighLevel_Op< "predefined.expr" >
  , Arguments<(ins AnyType:$value, HighLevel_IdentKind:$kind)>
  , Results<(outs AnyType:$result)>
{
    let summary = "VAT predefined expr ( such as __func__ )";
    let description = [{ VAT predefined expr ( such as __func__ ) }];

    let assemblyFormat = "$value $kind attr-dict `:` type($value) `->` type($result)";
}

def HighLevel_ExtensionOp
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

def HighLevel_AsmOp
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
  let description = [{
    VAST operation mirroring the GCCAsmStmt in clang AST. It prints a name for
    every operand (either its id or user-supplied string).
  }];

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

  let assemblyFormat = [{
    attr-dict $asm_template `(`($output_names $asm_outputs^ `:` $output_constraints)? `)` `(` (`ins` `:`$input_names $asm_inputs^ `:` $input_constraints)? `)` `(`( $clobbers^)?`)` `(`( $labels^)?`)` `:` functional-type(operands, results)
  }];
}

def HighLevel_VAArgExpr
  : HighLevel_Op< "va_arg_expr" >
  , Arguments< (ins AnyType: $arg_list) >
  , Results< (outs AnyType: $result) >
{
  let summary = "Fetch next variadic element.";

  let assemblyFormat = "$arg_list `:` functional-type(operands, $result) attr-dict";
}

def HighLevel_OpaqueValueExpr
  : HighLevel_Op< "opaque_expr" >
  , Arguments< (ins Variadic< AnyType >: $arg) >
  , Results< (outs AnyType: $result) >
{
  let summary = "Opaque value expression from clang AST.";
  let description = [{
    This op takes an argument and returns it. According to clang documentation
    it is "usually" a copy operation with no special semantings.
    The current use in VAST is to have the opaque value computed in some region and
    then using this op to extract the computed value from region argument (see BinaryCondOp).
  }];

  let assemblyFormat = "$arg `:` functional-type(operands, $result) attr-dict";
}

#endif // VAST_DIALECT_HIGHLEVEL_IR_HIGHLEVELOPS

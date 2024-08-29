// Copyright (c) 2023-present, Trail of Bits, Inc.

#ifndef VAST_DIALECT_CORE_UTILS
#define VAST_DIALECT_CORE_UTILS

include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/DialectBase.td"

def Core_HasOneBlock   : CPred<"$_self.hasOneBlock()">;

// TODO(Heno): constraints on regions
def Core_CondRegion  : Region< Core_HasOneBlock, "condition region" >;
def Core_CasesRegion : Region< Core_HasOneBlock, "cases region" >;
def Core_ValueRegion : Region< Core_HasOneBlock, "value region" >;

//
// Type traits
//
// The user dialect also has to include vast/Dialect/Core/TypeTraits.hpp
// Adding new traits requires class definition in vast/Dialect/Core/TypeTraits.hpp
//

class Core_NativeTypeTrait< string name > : NativeTypeTrait< name > {
    let cppNamespace = "::vast::core";
}

def Core_IntegralTypeTrait : Core_NativeTypeTrait< "IntegralTypeTrait" >;
def Core_BoolTypeTrait     : Core_NativeTypeTrait< "BoolTypeTrait" >;
def Core_CharTypeTrait     : Core_NativeTypeTrait< "CharTypeTrait" >;
def Core_ShortTypeTrait    : Core_NativeTypeTrait< "ShortTypeTrait" >;
def Core_IntegerTypeTrait  : Core_NativeTypeTrait< "IntegerTypeTrait" >;
def Core_LongTypeTrait     : Core_NativeTypeTrait< "LongTypeTrait" >;
def Core_LongLongTypeTrait : Core_NativeTypeTrait< "LongLongTypeTrait" >;
def Core_Int128TypeTrait   : Core_NativeTypeTrait< "Int128TypeTrait" >;

def Core_FloatingTypeTrait : Core_NativeTypeTrait< "FloatingTypeTrait" >;

def Core_PointerTypeTrait  : Core_NativeTypeTrait< "PointerTypeTrait" >;

def Core_TypedefTrait      : Core_NativeTypeTrait< "TypedefTrait" >;

def Core_TypeOfTrait       : Core_NativeTypeTrait< "TypeOfTrait" >;

def Core_AutoTrait         : Core_NativeTypeTrait< "AutoTrait" >;

def Core_VoidTrait         : Core_NativeTypeTrait< "VoidTrait" >;

//
// Type trait predicates
//

class Core_IsIntegral< string arg > : PredOpTrait< "value is of an integral type",
    CPred< "$" # arg # ".getType().hasTrait< core::IntegralTypeTrait >()" >
>;

class Core_IsBool< string arg > : PredOpTrait< "value is of a bool type",
    CPred< "$" # arg # ".getType().hasTrait< core::BoolTypeTrait >()" >
>;

class Core_IsChar< string arg > : PredOpTrait< "value is of a char type",
    CPred< "$" # arg # ".getType().hasTrait< core::CharTypeTrait >()" >
>;

class Core_IsShort< string arg > : PredOpTrait< "value is of a short type",
    CPred< "$" # arg # ".getType().hasTrait< core::ShortTypeTrait >()" >
>;

class Core_IsInteger< string arg > : PredOpTrait< "value is of an integer type",
    CPred< "$" # arg # ".getType().hasTrait< core::IntegerTypeTrait >()" >
>;

class Core_IsLong< string arg > : PredOpTrait< "value is of a long type",
    CPred< "$" # arg # ".getType().hasTrait< core::LongTypeTrait >()" >
>;

class Core_IsLongLong< string arg > : PredOpTrait< "value is of a long long type",
    CPred< "$" # arg # ".getType().hasTrait< core::LongLongTypeTrait >()" >
>;

class Core_IsInt128< string arg > : PredOpTrait< "value is of a int128 type",
    CPred< "$" # arg # ".getType().hasTrait< core::Int128TypeTrait >()" >
>;


class Core_IsPointer< string arg > : PredOpTrait< "value is of a pointer type",
    CPred< "$" # arg # ".getType().hasTrait< core::PointerTypeTrait >()" >
>;


class Core_IsTypedef< string arg > : PredOpTrait< "value is of a typedefed type",
    CPred< "$" # arg # ".getType().hasTrait< core::TypedefTrait >()" >
>;

class Core_IsTypeOf< string arg > : PredOpTrait< "value is of a typeofed type",
    CPred< "$" # arg # ".getType().hasTrait< core::TypeOfTrait >()" >
>;

class Core_IsBuiltinType< string arg > : PredOpTrait< "is a type from the builtin dialect",
    CPred< "mlir::isa< mlir::BuiltinDialect >($" # arg # ".getType().getDialect())" >
>;

class Core_IsVoid< string arg > : PredOpTrait< "value is of a void type",
  CPred< "$" # arg # ".getType().hasTrait< core::VoidTrait >()" >
>;

def Core_VoidLikeType : Type<
    CPred<"$_self.hasTrait< core::VoidTrait >()">,
    "Type with VoidTrit"
>;


#endif // VAST_DIALECT_CORE_UTILS

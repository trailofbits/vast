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

def Core_IntegralTypeTrait : NativeTypeTrait< "IntegralTypeTrait" >;
def Core_BoolTypeTrait     : NativeTypeTrait< "BoolTypeTrait" >;
def Core_CharTypeTrait     : NativeTypeTrait< "CharTypeTrait" >;
def Core_ShortTypeTrait    : NativeTypeTrait< "ShortTypeTrait" >;
def Core_IntegerTypeTrait  : NativeTypeTrait< "IntegerTypeTrait" >;
def Core_LongTypeTrait     : NativeTypeTrait< "LongTypeTrait" >;
def Core_LongLongTypeTrait : NativeTypeTrait< "LongLongTypeTrait" >;
def Core_Int128TypeTrait   : NativeTypeTrait< "Int128TypeTrait" >;

def Core_FloatingTypeTrait : NativeTypeTrait< "FloatingTypeTrait" >;

def Core_PointerTypeTrait  : NativeTypeTrait< "PointerTypeTrait" >;

def Core_TypedefTrait      : NativeTypeTrait< "TypedefTrait" >;

def Core_TypeOfTrait       : NativeTypeTrait< "TypeOfTrait" >;

def Core_VoidTrait         : NativeTypeTrait< "VoidTrait" >;

//
// Type trait predicates
//

class Core_IsIntegral< string arg > : PredOpTrait< "value is of an integral type",
    CPred< "$" # arg # ".getType().hasTrait< mlir::TypeTrait::IntegralTypeTrait >()" >
>;

class Core_IsBool< string arg > : PredOpTrait< "value is of a bool type",
    CPred< "$" # arg # ".getType().hasTrait< mlir::TypeTrait::BoolTypeTrait >()" >
>;

class Core_IsChar< string arg > : PredOpTrait< "value is of a char type",
    CPred< "$" # arg # ".getType().hasTrait< mlir::TypeTrait::CharTypeTrait >()" >
>;

class Core_IsShort< string arg > : PredOpTrait< "value is of a short type",
    CPred< "$" # arg # ".getType().hasTrait< mlir::TypeTrait::ShortTypeTrait >()" >
>;

class Core_IsInteger< string arg > : PredOpTrait< "value is of an integer type",
    CPred< "$" # arg # ".getType().hasTrait< mlir::TypeTrait::IntegerTypeTrait >()" >
>;

class Core_IsLong< string arg > : PredOpTrait< "value is of a long type",
    CPred< "$" # arg # ".getType().hasTrait< mlir::TypeTrait::LongTypeTrait >()" >
>;

class Core_IsLongLong< string arg > : PredOpTrait< "value is of a long long type",
    CPred< "$" # arg # ".getType().hasTrait< mlir::TypeTrait::LongLongTypeTrait >()" >
>;

class Core_IsInt128< string arg > : PredOpTrait< "value is of a int128 type",
    CPred< "$" # arg # ".getType().hasTrait< mlir::TypeTrait::Int128TypeTrait >()" >
>;


class Core_IsPointer< string arg > : PredOpTrait< "value is of a pointer type",
    CPred< "$" # arg # ".getType().hasTrait< mlir::TypeTrait::PointerTypeTrait >()" >
>;


class Core_IsTypedef< string arg > : PredOpTrait< "value is of a typedefed type",
    CPred< "$" # arg # ".getType().hasTrait< mlir::TypeTrait::TypedefTrait >()" >
>;

class Core_IsTypeOf< string arg > : PredOpTrait< "value is of a typeofed type",
    CPred< "$" # arg # ".getType().hasTrait< mlir::TypeTrait::TypeOfTrait >()" >
>;

class IsBuiltinType< string arg > : PredOpTrait< "is a type from the builtin dialect",
    CPred< "mlir::isa< mlir::BuiltinDialect >($" # arg # ".getType().getDialect())" >
>;

class Core_IsVoid< string arg > : PredOpTrait< "value is of a void type",
  CPred< "$" # arg # ".getType().hasTrait< mlir::TypeTrait::VoidTrait >()" >
>;

def Core_VoidLikeType : Type<
    CPred<"$_self.hasTrait< mlir::TypeTrait::VoidTrait >()">,
    "Type with VoidTrit"
>;


#endif // VAST_DIALECT_CORE_UTILS

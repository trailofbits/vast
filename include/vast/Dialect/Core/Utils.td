// Copyright (c) 2023-present, Trail of Bits, Inc.

#ifndef VAST_DIALECT_CORE_UTILS
#define VAST_DIALECT_CORE_UTILS

include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/DialectBase.td"

def HasOneBlock   : CPred<"$_self.hasOneBlock()">;

// TODO(Heno): constraints on regions
def CondRegion  : Region< HasOneBlock, "condition region" >;
def CasesRegion : Region< HasOneBlock, "cases region" >;
def ValueRegion : Region< HasOneBlock, "value region" >;

//
// Type traits
//
// The user dialect also has to include vast/Dialect/Core/TypeTraits.hpp
// Adding new traits requires class definition in vast/Dialect/Core/TypeTraits.hpp
//

def IntegralTypeTrait : NativeTypeTrait< "IntegralTypeTrait" >;
def BoolTypeTrait     : NativeTypeTrait< "BoolTypeTrait" >;
def CharTypeTrait     : NativeTypeTrait< "CharTypeTrait" >;
def ShortTypeTrait    : NativeTypeTrait< "ShortTypeTrait" >;
def IntegerTypeTrait  : NativeTypeTrait< "IntegerTypeTrait" >;
def LongTypeTrait     : NativeTypeTrait< "LongTypeTrait" >;
def LongLongTypeTrait : NativeTypeTrait< "LongLongTypeTrait" >;
def Int128TypeTrait   : NativeTypeTrait< "Int128TypeTrait" >;

def FloatingTypeTrait : NativeTypeTrait< "FloatingTypeTrait" >;

def PointerTypeTrait  : NativeTypeTrait< "PointerTypeTrait" >;

def TypedefTrait      : NativeTypeTrait< "TypedefTrait" >;

def TypeOfTrait       : NativeTypeTrait< "TypeOfTrait" >;

def VoidTrait         : NativeTypeTrait< "VoidTrait" >;

//
// Type trait predicates
//

class IsIntegral< string arg > : PredOpTrait< "value is of an integral type",
    CPred< "$" # arg # ".getType().hasTrait< mlir::TypeTrait::IntegralTypeTrait >()" >
>;

class IsBool< string arg > : PredOpTrait< "value is of a bool type",
    CPred< "$" # arg # ".getType().hasTrait< mlir::TypeTrait::BoolTypeTrait >()" >
>;

class IsChar< string arg > : PredOpTrait< "value is of a char type",
    CPred< "$" # arg # ".getType().hasTrait< mlir::TypeTrait::CharTypeTrait >()" >
>;

class IsShort< string arg > : PredOpTrait< "value is of a short type",
    CPred< "$" # arg # ".getType().hasTrait< mlir::TypeTrait::ShortTypeTrait >()" >
>;

class IsInteger< string arg > : PredOpTrait< "value is of an integer type",
    CPred< "$" # arg # ".getType().hasTrait< mlir::TypeTrait::IntegerTypeTrait >()" >
>;

class IsLong< string arg > : PredOpTrait< "value is of a long type",
    CPred< "$" # arg # ".getType().hasTrait< mlir::TypeTrait::LongTypeTrait >()" >
>;

class IsLongLong< string arg > : PredOpTrait< "value is of a long long type",
    CPred< "$" # arg # ".getType().hasTrait< mlir::TypeTrait::LongLongTypeTrait >()" >
>;

class IsInt128< string arg > : PredOpTrait< "value is of a int128 type",
    CPred< "$" # arg # ".getType().hasTrait< mlir::TypeTrait::Int128TypeTrait >()" >
>;


class IsPointer< string arg > : PredOpTrait< "value is of a pointer type",
    CPred< "$" # arg # ".getType().hasTrait< mlir::TypeTrait::PointerTypeTrait >()" >
>;


class IsTypedef< string arg > : PredOpTrait< "value is of a typedefed type",
    CPred< "$" # arg # ".getType().hasTrait< mlir::TypeTrait::TypedefTrait >()" >
>;

class IsTypeOf< string arg > : PredOpTrait< "value is of a typeofed type",
    CPred< "$" # arg # ".getType().hasTrait< mlir::TypeTrait::TypeOfTrait >()" >
>;

class IsBuiltinType< string arg > : PredOpTrait< "is a type from the builtin dialect",
    CPred< "mlir::isa< mlir::BuiltinDialect >($" # arg # ".getType().getDialect())" >
>;

class IsVoid< string arg > : PredOpTrait< "value is of a void type",
  CPred< "$" # arg # ".getType().hasTrait< mlir::TypeTrait::VoidTrait >()" >
>;

#endif // VAST_DIALECT_CORE_UTILS


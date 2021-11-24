// Copyright (c) 2021-present, Trail of Bits, Inc.

#ifndef VAST_DIALECT_HIGHLEVEL_IR_HIGHLEVELTYPES
#define VAST_DIALECT_HIGHLEVEL_IR_HIGHLEVELTYPES

def HighLevelType : DialectType< HighLevel_Dialect, CPred< "$_self.isa<HighLevelType>()" >,
  "HighLevelType", "::vast::hl::HighLevelType" >;


def VoidType : DialectType< HighLevel_Dialect,
  CPred< "$_self.isa<VoidType>()" >, "VoidType", "::vast::hl::VoidType" >,
  BuildableType< "VoidType::get($_builder.getContext())"
>;

def BoolType : DialectType< HighLevel_Dialect,
  CPred< "$_self.isa<BoolType>()" >, "BoolType", "::vast::hl::BoolType" >,
  BuildableType< "BoolType::get($_builder.getContext())"
>;

def IntegerType : DialectType< HighLevel_Dialect,
  CPred< "$_self.isa<IntegerType>()" >, "IntegerType", "::vast::hl::IntegerType"
>;

def FloatingType : DialectType< HighLevel_Dialect,
  CPred< "$_self.isa<FloatingType>()" >, "FloatingType", "::vast::hl::FloatingType"
>;

def PointerType : DialectType< HighLevel_Dialect,
  CPred< "$_self.isa<PointerType>()">, "PointerType", "::vast::hl::PointerType"
>;

def AnyPointer : TypeConstraint< PointerType.predicate, "any pointer" >;

#endif // VAST_DIALECT_HIGHLEVEL_IR_HIGHLEVELTYPES

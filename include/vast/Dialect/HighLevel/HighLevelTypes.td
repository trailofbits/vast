// Copyright (c) 2021-present, Trail of Bits, Inc.

#ifndef VAST_DIALECT_HIGHLEVEL_IR_HIGHLEVELTYPES
#define VAST_DIALECT_HIGHLEVEL_IR_HIGHLEVELTYPES

def HighLevelType : DialectType< HighLevel_Dialect,
  CPred< "$_self.isa<HighLevelType>()" >, "HighLevelType", "::vast::hl::HighLevelType"
>;

class HighLevelType< string name, list<Trait> traits = []>
  : TypeDef< HighLevel_Dialect, name , traits >;

def VoidType : HighLevelType< "VoidType" > {
  let mnemonic = "void";
}


def HLBoolType : DialectType< HighLevel_Dialect,
    CPred< "$_self.isa<BoolType>()" >, "BoolType", "::vast::hl::BoolType"
  >,
  BuildableType< "BoolType::get($_builder.getContext())"
>;

def HLIntegerType : DialectType< HighLevel_Dialect,
  CPred< "$_self.isa<IntegerType>()" >,
  "IntegerType", "::vast::hl::IntegerType"
>;

def HLFloatingType : DialectType< HighLevel_Dialect,
  CPred< "$_self.isa<FloatingType>()" >,
  "FloatingType", "::vast::hl::FloatingType"
>;

def HLPointerType : DialectType< HighLevel_Dialect,
  CPred< "$_self.isa<PointerType>()">,
  "PointerType", "::vast::hl::PointerType"
>;

def PointerLikeType : TypeConstraint<
  HLPointerType.predicate,
  "pointer like type"
>;

def BoolLikeType : TypeConstraint<
  Or< [HLBoolType.predicate, AnyI1.predicate] >,
  "bool like type"
>;

def IntegerLikeType : TypeConstraint<
  Or< [HLIntegerType.predicate, AnyInteger.predicate] >,
  "integer like type"
>;

def FloatLikeType : TypeConstraint<
  Or< [HLFloatingType.predicate, AnyFloat.predicate] >,
  "float like type"
>;

#endif // VAST_DIALECT_HIGHLEVEL_IR_HIGHLEVELTYPES

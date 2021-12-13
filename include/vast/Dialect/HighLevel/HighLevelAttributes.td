// Copyright (c) 2021-present, Trail of Bits, Inc.

#ifndef VAST_DIALECT_HIGHLEVEL_IR_HIGHLEVELATTRIBUTES
#define VAST_DIALECT_HIGHLEVEL_IR_HIGHLEVELATTRIBUTES

include "mlir/IR/OpBase.td"
include "vast/Dialect/HighLevel/HighLevelTypes.td"

class HighLevel_Attr<string name, string attr_mnemonic>
    : AttrDef< HighLevel_Dialect, name >
{
  let mnemonic = attr_mnemonic;
}

class HLIntegerAttr<string name, string attr_mnemonic, Type type >
  : HighLevel_Attr< name, attr_mnemonic >
{
  let valueType = type;
  let parameters = (ins
    "llvm::APInt":$value
  );

  let parser  = [{
    return parse_integer_attr<}] # cppClassName # [{>($_ctxt, parser);
  }];

  let printer = [{ return print_integer_attr(*this, printer); }];
}

def CharAttr     : HLIntegerAttr< "Char",     "char",     CharType >;
def ShortAttr    : HLIntegerAttr< "Short",    "shor",     ShortType >;
def IntAttr      : HLIntegerAttr< "Int",      "int",      IntType >;
def LongAttr     : HLIntegerAttr< "Long",     "long",     LongType >;
def LongLongAttr : HLIntegerAttr< "LongLong", "longlong", LongLongType >;
def Int128Attr   : HLIntegerAttr< "Int128",   "int128",   Int128Type >;

class HLFloatingAttr<string name, string attr_mnemonic, Type type >
  : HighLevel_Attr< name, attr_mnemonic >
{
  let valueType = type;
  let parameters = (ins
    "llvm::APFloat":$value
  );

  let parser  = [{
    return parse_floating_attr<}] # cppClassName # [{>($_ctxt, parser);
  }];

  let printer = [{ return print_floating_attr(*this, printer); }];
}

def FloatAttr      : HLFloatingAttr< "Float",      "float",      FloatType >;
def DoubleAttr     : HLFloatingAttr< "Double",     "double",     DoubleType >;
def LongDoubleAttr : HLFloatingAttr< "LongDouble", "longdouble", LongDoubleType >;


#endif // VAST_DIALECT_HIGHLEVEL_IR_HIGHLEVELATTRIBUTES

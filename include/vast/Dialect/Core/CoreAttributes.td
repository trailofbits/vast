// Copyright (c) 2023, Trail of Bits, Inc.

#ifndef VAST_DIALECT_CORE_COREATTRIBUTES
#define VAST_DIALECT_CORE_COREATTRIBUTES

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"

class Core_Attr<string name, string attr_mnemonic, list<Trait> traits = []>
    : AttrDef< Core_Dialect, name, traits >
{
  let mnemonic = attr_mnemonic;
}

class APSIntParameter<string desc> :
    AttrOrTypeParameter<"::llvm::APSInt", desc> {
  let comparator = [{
    $_lhs.getBitWidth() == $_rhs.getBitWidth() &&
    $_lhs.isSigned() == $_rhs.isSigned() &&
    $_lhs == $_rhs
  }];
}

def BooleanAttr : Core_Attr<"Boolean", "bool", [TypedAttrInterface] > {
  let summary = "An Attribute containing a boolean value";

  let description = [{
    An boolean attribute is a literal attribute that represents a boolean value.
  }];

  let parameters = (ins AttributeSelfTypeParameter<"">:$type, "bool":$value);

  let builders = [
    AttrBuilderWithInferredContext<(ins "Type":$type, "bool":$value), [{
      return $_get(type.getContext(), type, value);
    }]>
  ];

  let assemblyFormat = "`<` $value `>`";

  // let genVerifyDecl = 1;
}

def IntegerAttr : Core_Attr<"Integer", "integer", [TypedAttrInterface] > {
  let summary = "An Attribute containing a integer value";

  let description = [{
    An integer attribute is a literal attribute that represents an integral
    value of the specified integer type.
  }];

  let parameters = (ins AttributeSelfTypeParameter<"">:$type, APSIntParameter<"">:$value);

  let builders = [
    AttrBuilderWithInferredContext<(ins "Type":$type, "const llvm::APSInt &":$value), [{
      return $_get(type.getContext(), type, value);
    }]>
  ];

  let assemblyFormat = "`<` $value `>`";

  // let genVerifyDecl = 1;
}

def FloatAttr : Core_Attr<"Float", "float", [TypedAttrInterface] > {
  let summary = "An Attribute containing a floating point value";

  let description = [{
    A float attribute is a literal attribute that represents a floating point
    value of the specified floating point type.
  }];

  let parameters = (ins AttributeSelfTypeParameter<"">:$type, APFloatParameter<"">:$value);

  let builders = [
    AttrBuilderWithInferredContext<(ins "Type":$type, "const llvm::APFloat &":$value), [{
      return $_get(type.getContext(), type, value);
    }]>
  ];

  let assemblyFormat = "`<` $value `>`";

  // let genVerifyDecl = 1;
}

def StringLiteralAttr : Core_Attr< "StringLiteral", "strlit", [TypedAttrInterface] > {
  let summary = "An Attribute containing an escaped string";
  let hasCustomAssemblyFormat = 1;

  let description = [{
    Syntax:

    ```
    string-attribute ::= string-literal (`:` type)?
    ```

    A string literal attribute is an attribute that represents an escaped string literal value.

    Examples:

    ```mlir
    "An important string"
    "string with a type" : !dialect.string
    ```
  }];

  let parameters = (ins StringRefParameter<"">:$value, AttributeSelfTypeParameter<"">:$type);

  let builders = [
    AttrBuilderWithInferredContext<(ins "Type":$type, "const llvm::Twine &":$value), [{
      llvm::SmallVector<char, 32> tempStr;
      return Base::get(type.getContext(), escapeString(value.toStringRef(tempStr)), type);
    }]>
  ];

  let extraClassDeclaration = [{
    static char escapeChar(char c) {
        std::pair< char, char > const pairs[] {
            { '\a', 'a' },
            { '\b', 'b' },
            { '\f', 'f' },
            { '\n', 'n' },
            { '\r', 'r' },
            { '\t', 't' },
            { '\v', 'v' },
        };
        for (auto [fst, snd] : pairs) {
            if (c == fst)
                return snd;
        }
        return c;
    }

    static std::string escapeString(llvm::StringRef src) {
        std::string res;
        llvm::raw_string_ostream stream(res);

        for (unsigned char C : src) {
            if (C == '\\')
                stream << '\\' << C;
            else if (llvm::sys::locale::isPrint(C) && C != '"')
                stream << C;
            else
                stream << '\\' << escapeChar(C);
        }
        return res;
    }
  }];
}

def VoidAttr : Core_Attr<"Void", "void", [TypedAttrInterface]> {
  let summary = "Attribute to represent void value.";
  let description = [{
    The VoidAttr is used to return void from statements uniformly.
  }];

  let parameters = (ins AttributeSelfTypeParameter< "" >:$type);
  let assemblyFormat = [{}];
}

#endif // VAST_DIALECT_CORE_COREATTRIBUTES

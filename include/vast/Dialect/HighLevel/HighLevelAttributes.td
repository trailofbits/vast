// Copyright (c) 2021-present, Trail of Bits, Inc.

#ifndef VAST_DIALECT_HIGHLEVEL_IR_HIGHLEVELATTRIBUTES
#define VAST_DIALECT_HIGHLEVEL_IR_HIGHLEVELATTRIBUTES

include "mlir/IR/OpBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"

class HighLevel_Attr<string name, string attr_mnemonic, list<Trait> traits = []>
    : AttrDef< HighLevel_Dialect, name, traits >
{
  let mnemonic = attr_mnemonic;
}

class HighLevel_NameAttr<string name, string attr_mnemonic>
  : HighLevel_Attr< name, attr_mnemonic >
{
  let parameters = (ins "::mlir::StringAttr":$name);

  let builders = [
    AttrBuilderWithInferredContext<(ins "::mlir::StringAttr":$name), [{
      return get(name.getContext(), name);
    }]>,
    AttrBuilder<(ins "::mlir::StringRef":$name), [{
      return get(mlir::StringAttr::get($_ctxt, name));
    }]>,
  ];

  let assemblyFormat = "`<` $name `>`";
}

class HighLevel_SymbolReferringAttr<string name, string attr_mnemonic>
  : HighLevel_Attr< name, attr_mnemonic >
{
  let parameters = (ins "::mlir::SymbolRefAttr":$symbol);

  let builders = [
    AttrBuilderWithInferredContext<(ins "::mlir::SymbolRefAttr":$symbol), [{
      return get(symbol.getContext(), symbol);
    }]>,
    AttrBuilder<(ins "::mlir::StringRef":$symbol), [{
      return get(mlir::SymbolRefAttr::get($_ctxt, symbol));
    }]>,
  ];

  let assemblyFormat = "`<` $symbol `>`";
}

class HighLevel_FlatSymbolReferringAttr<string name, string attr_mnemonic>
  : HighLevel_Attr< name, attr_mnemonic >
{
  let parameters = (ins "::mlir::FlatSymbolRefAttr":$symbol);

  let builders = [
    AttrBuilderWithInferredContext<(ins "::mlir::FlatSymbolRefAttr":$symbol), [{
      return get(symbol.getContext(), symbol);
    }]>,
    AttrBuilder<(ins "::mlir::StringRef":$symbol), [{
      return get(mlir::FlatSymbolRefAttr::get($_ctxt, symbol));
    }]>,
  ];

  let assemblyFormat = "`<` $symbol `>`";
}

def HighLevel_AnnotationAttr       : HighLevel_NameAttr< "Annotation", "annotation" >;
def HighLevel_FormatAttr           : HighLevel_NameAttr< "Format", "format" >;
def HighLevel_SectionAttr          : HighLevel_NameAttr< "Section", "section" >;
def HighLevel_AliasAttr            : HighLevel_NameAttr< "Alias", "alias" >;
def HighLevel_ErrorAttr            : HighLevel_NameAttr< "Error", "error" >;
def HighLevel_CountedByAttr        : HighLevel_Attr< "CountedBy", "counted_by" >;
def HighLevel_CleanupAttr          : HighLevel_SymbolReferringAttr< "Cleanup", "cleanup" >;
// TODO(#595): Make aligned attribute keep the alignment value/expr
def HighLevel_AlignedAttr          : HighLevel_Attr< "Aligned", "aligned" >;
def HighLevel_AlwaysInlineAttr     : HighLevel_Attr< "AlwaysInline", "always_inline" >;
def HighLevel_NoInlineAttr         : HighLevel_Attr< "NoInline", "noinline" >;
def HighLevel_ConstAttr            : HighLevel_Attr< "Const", "const" >;
def HighLevel_LoaderUninitAttr     : HighLevel_Attr< "LoaderUninitialized", "loader_uninitialized" >;
def HighLevel_NoInstrumentAttr     : HighLevel_Attr< "NoInstrumentFunction", "no_instrument_function" >;
def HighLevel_PackedAttr           : HighLevel_Attr< "Packed", "packed" >;
def HighLevel_PureAttr             : HighLevel_Attr< "Pure", "pure" >;
def HighLevel_WarnUnusedResAttr    : HighLevel_Attr< "WarnUnusedResult", "warn_unused_result" >;
def HighLevel_RestrictAttr         : HighLevel_Attr< "Restrict", "restrict" >;
def HighLevel_NoThrowAttr          : HighLevel_Attr< "NoThrow", "nothrow" >;
def HighLevel_NonNullAttr          : HighLevel_Attr< "NonNull", "nonnull" >;
def HighLevel_LeafAttr             : HighLevel_Attr< "Leaf", "leaf" >;
def HighLevel_ColdAttr             : HighLevel_Attr< "Cold", "cold" >;
def HighLevel_TransparentUnionAttr : HighLevel_Attr< "TransparentUnion", "transparent_union" >;
def HighLevel_ReturnsTwiceAttr     : HighLevel_Attr< "ReturnsTwice", "returns_twice" >;
def HighLevel_MayAliasAttr         : HighLevel_Attr< "MayAlias", "may_alias" >;
def HighLevel_UnusedAttr           : HighLevel_Attr< "Unused", "unused" >;
def HighLevel_UsedAttr             : HighLevel_Attr< "Used", "used" >;
def HighLevel_GNUInlineAttr        : HighLevel_Attr< "GNUInline", "gnu_inline" >;
def HighLevel_NoCfCheckAttr        : HighLevel_Attr< "NoCfCheck", "nocf_check" >;
def HighLevel_AvailableOnlyInDefaultEvalMethodAttr
  : HighLevel_Attr< "AvailableOnlyInDefaultEvalMethod", "available_only_in_default_eval_method" >;
def HighLevel_AvailabilityAttr     : HighLevel_Attr< "AvailabilityAttr", "availability" >;
def HighLevel_FallthroughAttr      : HighLevel_Attr< "Fallthrough", "fallthrough" >;
def HighLevel_NoProfileInstrumentFunctionAttr
  : HighLevel_Attr< "NoProfileInstrumentFunction", "no_profile_instrument_function" >;

def HighLevel_AsmLabelAttr
  : HighLevel_Attr< "AsmLabel", "asm" >
{
  let parameters = (ins "::mlir::StringAttr":$label, "bool":$isLiteral);

  let builders = [
    AttrBuilderWithInferredContext<(ins "::mlir::StringAttr":$label, "bool":$literal), [{
      return get(label.getContext(), label, literal);
    }]>,
    AttrBuilder<(ins "::mlir::StringRef":$label, "bool":$literal), [{
      return get(mlir::StringAttr::get($_ctxt, label), literal);
    }]>,
  ];

  let assemblyFormat = "`<` $label `,` $isLiteral `>`";
}

def HighLevel_ModeAttr
  : HighLevel_Attr< "Mode", "mode" >
{
  let parameters = (ins "::mlir::StringAttr":$mode);

  let builders = [
    AttrBuilderWithInferredContext<(ins "::mlir::StringAttr":$mode), [{
      return get(mode.getContext(), mode);
    }]>,
    AttrBuilder<(ins "::mlir::StringRef":$mode), [{
      return get(mlir::StringAttr::get($_ctxt, mode));
    }]>,
  ];

  let assemblyFormat = "`<` $mode `>`";
}

def HighLevel_BuiltinAttr
  : HighLevel_Attr< "Builtin", "builtin" >
{
  let parameters = (ins "unsigned":$ID);
  let assemblyFormat = "`<` $ID `>`";
}

def HighLevel_AllocAlignAttr
  : HighLevel_Attr< "AllocAlign", "alloc_align" >
{
  let parameters = (ins "int":$alignment);
  let assemblyFormat = "`<` $alignment `>`";
}

def HighLevel_AllocSizeAttr
  : HighLevel_Attr< "AllocSize", "alloc_size" >
{
  let parameters = (ins "int":$size_arg_pos, OptionalParameter< "int" >:$num_arg_pos);
  let assemblyFormat = "`<` `size_pos` `:` $size_arg_pos (`,` `num_pos` `:` $num_arg_pos^)? `>`";
}

def HighLevel_DeprecatedAttr
  : HighLevel_Attr< "Deprecated", "deprecated" >
{
  let parameters = (ins
    "::mlir::StringAttr":$message,
    "::mlir::StringAttr":$fixit
  );

  let builders = [
    AttrBuilder<(ins "::mlir::StringRef":$msg, "::mlir::StringRef":$fix), [{
      return get($_ctxt, mlir::StringAttr::get($_ctxt, msg), mlir::StringAttr::get($_ctxt, fix));
    }]>,
  ];

  let assemblyFormat = "`<` `msg` `:` $message `,` `fix` `:` $fixit `>`";
}

def HighLevel_MaxFieldAlignmentAttr
  : HighLevel_Attr< "MaxFieldAlignment", "max_field_alignment" >
{
  let parameters = (ins "unsigned": $alignment);
  let assemblyFormat = "`<` $alignment `>`";
}

def HighLevel_Visibility : I32EnumAttr< "Visibility", "visibility",
  [
    I32EnumAttrCase<"Default", 0, "default">,
    I32EnumAttrCase<"Hidden", 1, "hidden">,
    I32EnumAttrCase<"Protected", 2, "protected">
] > {
  let cppNamespace = "::vast::hl";
  let genSpecializedAttr = 0;
}

def HighLevel_VisibilityAttr : EnumAttr< HighLevel_Dialect, HighLevel_Visibility, "visibility" > {
  let summary = "Visibility attribute";
  }

def HighLevel_AssumeAlignedAttr
  : HighLevel_Attr< "AssumeAligned", "assume_aligned" >
{
  // TODO: Add support for arbitrary subexpressions
  let parameters = (ins "llvm::APInt": $alignment, "llvm::APInt": $offset);
  let assemblyFormat = "`<` $alignment`,` $offset `>`";
}

#endif // VAST_DIALECT_HIGHLEVEL_IR_HIGHLEVELATTRIBUTES

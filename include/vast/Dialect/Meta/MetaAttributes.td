// Copyright (c) 2022-present, Trail of Bits, Inc.

#ifndef VAST_DIALECT_META_IR_METAATTRIBUTES
#define VAST_DIALECT_META_IR_METAATTRIBUTES

include "mlir/IR/OpBase.td"

class Meta_Attr<string name, string attr_mnemonic>
    : AttrDef< Meta_Dialect, name >
{
  let mnemonic = attr_mnemonic;
}

def Meta_IdentifierAttr : Meta_Attr< "Id", "id" > {
    let summary = "A metadata identifier.";

    let description = [{
        A metadata identifier can be used to relate operations
        to external metadata storage.

        ```mlir
        #meta.id<"0x3A28213A">
        ```
    }];

    let parameters = (ins "std::uint64_t":$value);

    let assemblyFormat = "`<` params `>`";
}

#endif // VAST_DIALECT_META_IR_METAATTRIBUTES

// Copyright (c) 2021-present, Trail of Bits, Inc.

#ifndef VAST_TYPEQUALIFIERSINTERFACES
#define VAST_TYPEQUALIFIERSINTERFACES

include "mlir/IR/OpBase.td"

def ConstQualifierInterface : AttrInterface< "ConstQualifierInterface" > {
    let description = [{ This is interface to access const qualifier. }];
    let cppNamespace = "::vast";

    let methods = [
        InterfaceMethod<"Returns true if type has const qualifier.",
            "bool", "hasConst", (ins), [{}],
            /*defaultImplementation=*/ [{
                return $_attr.getIsConst();
            }]
        >
    ];
}

def VolatileQualifierInterface : AttrInterface< "VolatileQualifierInterface" > {
    let description = [{ This is interface to access volatile qualifier. }];
    let cppNamespace = "::vast";

    let methods = [
        InterfaceMethod<"Returns true if type has volatile qualifier.",
            "bool", "hasVolatile", (ins), [{}],
            /*defaultImplementation=*/ [{
                return $_attr.getIsVolatile();
            }]
        >
    ];
}

def RestrictQualifierInterface : AttrInterface< "RestrictQualifierInterface" > {
    let description = [{ This is interface to access restrict qualifier. }];
    let cppNamespace = "::vast";

    let methods = [
        InterfaceMethod<"Returns true if tzpe has restrict qualifier.",
            "bool", "hasRestrict", (ins), [{}],
            /*defaultImplementation=*/ [{
                return $_attr.getIsRestrict();
            }]
        >
    ];
}

def UnsignedQualifierInterface : AttrInterface< "UnsignedQualifierInterface" > {
    let description = [{ This is interface to access unsigned qualifier. }];
    let cppNamespace = "::vast";

    let methods = [
        InterfaceMethod<"Returns true if type has unsigned qualifier.",
            "bool", "hasUnsigned", (ins), [{}],
            /*defaultImplementation=*/ [{
                return $_attr.getIsUnsigned();
            }]
        >
    ];
}

#endif  // VAST_TYPEQUALIFIERSINTERFACES

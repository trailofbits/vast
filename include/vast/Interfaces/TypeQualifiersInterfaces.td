// Copyright (c) 2021-present, Trail of Bits, Inc.

#ifndef VAST_INTERFACES_TYPE_QUALIFIERS_INTERFACES
#define VAST_INTERFACES_TYPE_QUALIFIERS_INTERFACES

include "mlir/IR/OpBase.td"
include "vast/Interfaces/Common.td"

def ConstQualifierInterface : VastAttrInterface< "ConstQualifierInterface" > {
    let description = [{ This is interface to access const qualifier. }];

    let methods = [
        InterfaceMethod<"Returns true if type has const qualifier.",
            "bool", "hasConst", (ins), [{}],
            /*defaultImplementation=*/ [{
                return $_attr.getIsConst();
            }]
        >
    ];
}

def VolatileQualifierInterface : VastAttrInterface< "VolatileQualifierInterface" > {
    let description = [{ This is interface to access volatile qualifier. }];

    let methods = [
        InterfaceMethod<"Returns true if type has volatile qualifier.",
            "bool", "hasVolatile", (ins), [{}],
            /*defaultImplementation=*/ [{
                return $_attr.getIsVolatile();
            }]
        >
    ];
}

def RestrictQualifierInterface : VastAttrInterface< "RestrictQualifierInterface" > {
    let description = [{ This is interface to access restrict qualifier. }];

    let methods = [
        InterfaceMethod<"Returns true if tzpe has restrict qualifier.",
            "bool", "hasRestrict", (ins), [{}],
            /*defaultImplementation=*/ [{
                return $_attr.getIsRestrict();
            }]
        >
    ];
}

def UnsignedQualifierInterface : VastAttrInterface< "UnsignedQualifierInterface" > {
    let description = [{ This is interface to access unsigned qualifier. }];

    let methods = [
        InterfaceMethod<"Returns true if type has unsigned qualifier.",
            "bool", "hasUnsigned", (ins), [{}],
            /*defaultImplementation=*/ [{
                return $_attr.getIsUnsigned();
            }]
        >
    ];
}

#endif  // VAST_INTERFACES_TYPE_QUALIFIERS_INTERFACES

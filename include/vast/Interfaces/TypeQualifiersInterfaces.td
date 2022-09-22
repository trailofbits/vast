// Copyright (c) 2021-present, Trail of Bits, Inc.

#ifndef VAST_TYPEQUALIFIERSINTERFACES
#define VAST_TYPEQUALIFIERSINTERFACES

include "mlir/IR/OpBase.td"

class QualifierTypeInterface< string interface_name, string qualifier >
    : TypeInterface< interface_name >
{
    let cppNamespace = "::vast::hl";

    let description = [{ Interface for types supporting }] # qualifier # [{ qualifier. }];

    let methods = [
        InterfaceMethod< "Queries presence of " # qualifier # " qualifier", "bool", "has_" # qualifier >
    ];
}

def ConstQualifierInterface : QualifierTypeInterface<
    "ConstQualifierInterface", "const"
>;

def VolatileQualifierInterface : QualifierTypeInterface<
    "VolatileQualifierInterface", "volatile"
>;

def RestrictQualifierInterface : QualifierTypeInterface<
    "RestrictQualifierInterface", "restrict"
>;

#endif  // VAST_TYPEQUALIFIERSINTERFACES

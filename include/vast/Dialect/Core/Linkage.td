// Copyright (c) 2022-present, Trail of Bits, Inc.

#ifndef VAST_DIALECT_CORE_IR_LINKAGE
#define VAST_DIALECT_CORE_IR_LINKAGE

include "mlir/IR/EnumAttr.td"

// Linkage types inspired from Clang IR - remove ones present in upstream MLIR

// Linkage types. This is currently a replay of llvm/IR/GlobalValue.h, this is
// currently handy as part of forwarding appropriate linkage types for LLVM
// lowering, specially useful for C++ support.

// Unknown linkage
def Global_UnknownLinkage :
  I32EnumAttrCase<"UnknownLinkage", 0, "unknown">;
// Externally visible function
def Global_ExternalLinkage :
  I32EnumAttrCase<"ExternalLinkage", 1, "external">;
// Available for inspection, not emission.
def Global_AvailableExternallyLinkage :
  I32EnumAttrCase<"AvailableExternallyLinkage", 2, "available_externally">;
// Keep one copy of function when linking (inline)
def Global_LinkOnceAnyLinkage :
  I32EnumAttrCase<"LinkOnceAnyLinkage", 3, "linkonce">;
// Same, but only replaced by something equivalent.
def Global_LinkOnceODRLinkage :
  I32EnumAttrCase<"LinkOnceODRLinkage", 4, "linkonce_odr">;
// Keep one copy of named function when linking (weak)
def Global_WeakAnyLinkage :
  I32EnumAttrCase<"WeakAnyLinkage", 5, "weak">;
// Same, but only replaced by something equivalent.
def Global_WeakODRLinkage :
  I32EnumAttrCase<"WeakODRLinkage", 6, "weak_odr">;
// Special purpose, only applies to global arrays
def Global_AppendingLinkage :
    I32EnumAttrCase<"AppendingLinkage", 7, "appending">;
// Rename collisions when linking (static functions).
def Global_InternalLinkage :
  I32EnumAttrCase<"InternalLinkage", 8, "internal">;
// Like Internal, but omit from symbol table.
def Global_PrivateLinkage :
  I32EnumAttrCase<"PrivateLinkage", 9, "private">;
// ExternalWeak linkage description.
def Global_ExternalWeakLinkage :
  I32EnumAttrCase<"ExternalWeakLinkage", 10, "extern_weak">;
// Tentative definitions.
def Global_CommonLinkage :
  I32EnumAttrCase<"CommonLinkage", 11, "common">;

def Core_GlobalLinkageKind : Core_I32Enum<
    "GlobalLinkageKind", "Linkage type/kind",
    [
        Global_UnknownLinkage,
        Global_ExternalLinkage, Global_AvailableExternallyLinkage,
        Global_LinkOnceAnyLinkage, Global_LinkOnceODRLinkage,
        Global_WeakAnyLinkage, Global_WeakODRLinkage, Global_InternalLinkage,
        Global_PrivateLinkage, Global_ExternalWeakLinkage, Global_CommonLinkage,
        Global_AppendingLinkage
    ]>
{}

def Core_GlobalLinkageKindAttr : Core_I32EnumAttr<
  "global_linkage_kind", Core_GlobalLinkageKind
>;

#endif // VAST_DIALECT_CORE_IR_LINKAGE

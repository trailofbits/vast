// Copyright (c) 2022-present, Trail of Bits, Inc.

#ifndef VAST_DIALECT_CORE_UTILS
#define VAST_DIALECT_CORE_UTILS

def HasOneBlock   : CPred<"$_self.hasOneBlock()">;

// TODO(Heno): constraints on regions
def CondRegion  : Region< HasOneBlock, "condition region" >;
def CasesRegion : Region< HasOneBlock, "cases region" >;
def ValueRegion : Region< HasOneBlock, "value region" >;

#endif // VAST_DIALECT_CORE_UTILS


// Copyright (c) 2021-present, Trail of Bits, Inc.

#pragma once

#include <mlir/Support/LogicalResult.h>
#include <llvm/Support/raw_ostream.h>

#include <mutex>

namespace vast
{
    namespace hl
    {
        mlir::LogicalResult registerFromSourceParser();
    } // namespace hl

    inline void registerAllTranslations()
    {
        static std::once_flag once;
        std::call_once(once, [] {
            if (vast::hl::registerFromSourceParser().failed()) {
                llvm::errs() << "Registracion of FromSource pass failed.\n";
            }
        });
    }

} // namespace vast

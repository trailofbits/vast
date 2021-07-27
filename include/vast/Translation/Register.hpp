// Copyright (c) 2021-present, Trail of Bits, Inc.

#include "mlir/Support/LogicalResult.h"
#include <mutex>

namespace vast
{
    mlir::LogicalResult registerFromSourceParser();

    inline void registerAllTranslations()
    {
        static std::once_flag once;
        std::call_once(once, [] {
            vast::registerFromSourceParser();
        });
    }

} // namespace vast
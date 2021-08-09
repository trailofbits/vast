// Copyright (c) 2021-present, Trail of Bits, Inc.

#include "mlir/Support/LogicalResult.h"
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
            vast::hl::registerFromSourceParser();
        });
    }

} // namespace vast
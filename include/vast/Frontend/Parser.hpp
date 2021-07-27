// Copyright (c) 2021-present, Trail of Bits, Inc.

#include <mutex>

namespace vast
{
    inline void registerAllTranslations()
    {
        static std::once_flag once;
        std::call_once(once, [] {
            // vast::registerClangASTToMLIR();
        });
    }

} // namespace vast
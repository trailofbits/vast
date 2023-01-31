#pragma once

#include <memory>
#include <vast/CodeGen/ABIInfo.hpp>

namespace vast::cg
{
    /// This class organizes various target-specific codegeneration issues, like
    /// target-specific attributes, builtins and so on.
    /// Equivalent to LLVM's TargetCodeGenInfo.
    class target_info_t {
        using abi_info_ptr = std::unique_ptr< abi_info_t >;

        abi_info_ptr info = nullptr;

      public:
        target_info_t(abi_info_ptr info)
            : info(std::move(info)) {}

        /// Returns ABI info helper for the target.
        const abi_info_t &get_abi_info() const { return *info; }
    };

} // namespace vast::cg

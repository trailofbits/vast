#pragma once

#include <memory>
#include <vast/CodeGen/ABIInfo.hpp>

namespace vast::cg
{

    /// This class organizes various target-specific codegeneration issues, like
    /// target-specific attributes, builtins and so on.
    /// Equivalent to LLVM's TargetCodeGenInfo.
    struct target_info_t {
        using abi_info_ptr = std::unique_ptr< abi_info_t >;

        explicit target_info_t(abi_info_ptr info)
            : info(std::move(info))
        {}

        /// Returns ABI info helper for the target.
        const abi_info_t &get_abi_info() const { return *info; }

      private:
        abi_info_ptr info = nullptr;
    };


    struct aarch64_target_info : target_info_t {
        aarch64_target_info(const type_info_t &type_info, aarch64_abi_info::abi_kind kind)
            : target_info_t(std::make_unique< aarch64_abi_info >(type_info, kind))
        {}
    };

    struct x86_64_target_info : target_info_t {
        x86_64_target_info(const type_info_t &type_info, x86_avx_abi_level avx_level)
            : target_info_t(std::make_unique< x86_64_abi_info >(type_info, avx_level))
        {}
    };

} // namespace vast::cg

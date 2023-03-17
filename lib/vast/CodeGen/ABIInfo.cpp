// Copyright (c) 2023-present, Trail of Bits, Inc.

#include <vast/CodeGen/ABIInfo.hpp>
#include <vast/CodeGen/TargetInfo.hpp>

namespace vast::cg
{
    bool is_void_type(qual_type type) {
        if (const auto *builtin = type->getAs< clang::BuiltinType >()) {
            return builtin->getKind() == clang::BuiltinType::Void;
        }

        return false;
    }

    void default_abi_info::compute_info(function_info_t &fninfo) const {
        // Top level vast has unlimited arguments and return types. Lowering for ABI
        // specific concerns should happen during a lowering phase. Assume
        // everything is direct for now.
        auto process = [&] (const auto &type) {
            if (is_void_type(type))
                return abi_arg_info::get_ignore();
            else
                return abi_arg_info::get_direct();
        };

        for (auto &arg : fninfo.arguments()) {
            arg.info = process(arg.type);
        }

        fninfo.get_return_info() = process(fninfo.get_return_type());
    }

    abi_arg_info default_abi_info::classify_return_type(
        qual_type /* rty */, bool /* variadic */
    ) const {
        VAST_UNIMPLEMENTED;
    }

    abi_arg_info default_abi_info::classify_arg_type(
        qual_type /* rty */, bool /* variadic */, unsigned /* calling_convention */
    ) const {
        VAST_UNIMPLEMENTED;
    }

} // namespace vast::cg

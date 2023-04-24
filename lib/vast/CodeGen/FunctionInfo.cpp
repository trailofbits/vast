// Copyright (c) 2023-present, Trail of Bits, Inc.

#include "vast/CodeGen/FunctionInfo.hpp"

namespace vast::cg {

    function_info_t *function_info_t::create(
        calling_conv calling_convention,
        bool instance_method,
        bool chain_call,
        const ext_info &info,
        ext_parameter_info_span params,
        qual_type rty,
        qual_types_span arg_types,
        required_args required
    ) {
        VAST_ASSERT(params.empty() || params.size() == arg_types.size());
        VAST_ASSERT(!required.allows_optional_args()
            || required.get_num_required_args() <= arg_types.size()
        );

        void *buffer = operator new(totalSizeToAlloc< arg_info, ext_param_info >(
            arg_types.size() + 1, params.size())
        );

        // FIXME constructor
        function_info_t *fninfo = new (buffer) function_info_t();
        fninfo->calling_convention = calling_convention;
        fninfo->effective_calling_convention = calling_convention;
        fninfo->ast_calling_convention = info.getCC();
        fninfo->instance_method = instance_method;
        fninfo->chain_call = chain_call;
        fninfo->cmse_nonsecure_call = info.getCmseNSCall();
        fninfo->no_return = info.getNoReturn();
        fninfo->returns_retained = info.getProducesResult();
        fninfo->no_caller_saved_regs = info.getNoCallerSavedRegs();
        fninfo->has_reg_parm = info.getHasRegParm();
        fninfo->reg_parm = info.getRegParm();
        fninfo->no_cf_check = info.getNoCfCheck();
        fninfo->required = required;
        // fninfo->ArgStruct = nullptr;
        fninfo->arg_struct_align = 0;
        fninfo->num_args = unsigned(arg_types.size());
        fninfo->has_ext_parameter_infos = !params.empty();

        fninfo->get_args_buffer()[0].type = rty;
        for (unsigned i = 0; i < arg_types.size(); ++i) {
            fninfo->get_args_buffer()[i + 1].type = arg_types[i];
        }

        for (unsigned i = 0; i < params.size(); ++i) {
            fninfo->get_ext_param_infos_buffer()[i] = params[i];
        }

        return fninfo;
    }

} // namespace vast::cg

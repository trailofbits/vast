// Copyright (c) 2024-present, Trail of Bits, Inc.

#include "vast/Conversion/Passes.hpp"

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <clang/Basic/Builtins.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
VAST_UNRELAX_WARNINGS

#include "PassesDetails.hpp"
#include "vast/Conversion/Common/Block.hpp"
#include "vast/Conversion/Common/Passes.hpp"
#include "vast/Conversion/Common/Patterns.hpp"
#include "vast/Conversion/Common/Rewriter.hpp"

#include "vast/Util/Attribute.hpp"
#include "vast/Util/Common.hpp"
#include "vast/Util/DialectConversion.hpp"
#include "vast/Util/TypeList.hpp"

#include "vast/Dialect/Builtin/Dialect.hpp"
#include "vast/Dialect/Builtin/Ops.hpp"
#include "vast/Dialect/HighLevel/HighLevelDialect.hpp"

#include "../PassesDetails.hpp"

#include <memory>

namespace vast::conv {

    struct identity_visitor
    {
        template< typename /*target*/, typename... args_t >
        logical_result visit(operation op, args_t... args) {
            return mlir::success();
        }
    };

    struct rewriter_visitor
    {
        explicit rewriter_visitor(conversion_rewriter &rw)
            : rewriter(rw)
        {}

        conversion_rewriter &rewriter;

        template< typename target, typename... args_t >
        auto visit(operation op, args_t... args) {
            rewriter.replaceOpWithNewOp< target >(op, args...);
            return mlir::success();
        }
    };

    template< typename visitor_t >
    auto visit_builtin_op(operation op, auto operands, visitor_t &&visitor) {
        auto caller = mlir::dyn_cast< mlir::CallOpInterface >(op);
        if (!caller) {
            return mlir::failure();
        }
        auto callee = caller.resolveCallable();

        auto attr = util::get_attr< hl::BuiltinAttr >(callee);
        if (!attr) {
            return mlir::failure();
        }

        auto id = mlir::cast< hl::BuiltinAttr >(attr).getID();
        VAST_CHECK(id != 0, "Attempting to visit builtin expr that is not builtin (id is 0).");

        switch (id) {
            // case clang::Builtin::BIceil:
            // case clang::Builtin::BIceilf:
            // case clang::Builtin::BIceill:
            // case clang::Builtin::BI__builtin_ceil:
            // case clang::Builtin::BI__builtin_ceilf:
            // case clang::Builtin::BI__builtin_ceilf16:
            // case clang::Builtin::BI__builtin_ceill:
            // case clang::Builtin::BI__builtin_ceilf128:

            // case clang::Builtin::BIcopysign:
            // case clang::Builtin::BIcopysignf:
            // case clang::Builtin::BIcopysignl:
            // case clang::Builtin::BI__builtin_copysign:
            // case clang::Builtin::BI__builtin_copysignf:
            // case clang::Builtin::BI__builtin_copysignf16:
            // case clang::Builtin::BI__builtin_copysignl:
            // case clang::Builtin::BI__builtin_copysignf128:

            // case clang::Builtin::BIcos:
            // case clang::Builtin::BIcosf:
            // case clang::Builtin::BIcosl:
            // case clang::Builtin::BI__builtin_cos:
            // case clang::Builtin::BI__builtin_cosf:
            // case clang::Builtin::BI__builtin_cosf16:
            // case clang::Builtin::BI__builtin_cosl:
            // case clang::Builtin::BI__builtin_cosf128:

            // case clang::Builtin::BIexp:
            // case clang::Builtin::BIexpf:
            // case clang::Builtin::BIexpl:
            // case clang::Builtin::BI__builtin_exp:
            // case clang::Builtin::BI__builtin_expf:
            // case clang::Builtin::BI__builtin_expf16:
            // case clang::Builtin::BI__builtin_expl:
            // case clang::Builtin::BI__builtin_expf128:

            // case clang::Builtin::BIexp2:
            // case clang::Builtin::BIexp2f:
            // case clang::Builtin::BIexp2l:
            // case clang::Builtin::BI__builtin_exp2:
            // case clang::Builtin::BI__builtin_exp2f:
            // case clang::Builtin::BI__builtin_exp2f16:
            // case clang::Builtin::BI__builtin_exp2l:
            // case clang::Builtin::BI__builtin_exp2f128:

            // case clang::Builtin::BIfabs:
            // case clang::Builtin::BIfabsf:
            // case clang::Builtin::BIfabsl:
            // case clang::Builtin::BI__builtin_fabs:
            // case clang::Builtin::BI__builtin_fabsf:
            // case clang::Builtin::BI__builtin_fabsf16:
            // case clang::Builtin::BI__builtin_fabsl:
            // case clang::Builtin::BI__builtin_fabsf128:

            // case clang::Builtin::BIfloor:
            // case clang::Builtin::BIfloorf:
            // case clang::Builtin::BIfloorl:
            // case clang::Builtin::BI__builtin_floor:
            // case clang::Builtin::BI__builtin_floorf:
            // case clang::Builtin::BI__builtin_floorf16:
            // case clang::Builtin::BI__builtin_floorl:
            // case clang::Builtin::BI__builtin_floorf128:

            // case clang::Builtin::BIfma:
            // case clang::Builtin::BIfmaf:
            // case clang::Builtin::BIfmal:
            // case clang::Builtin::BI__builtin_fma:
            // case clang::Builtin::BI__builtin_fmaf:
            // case clang::Builtin::BI__builtin_fmaf16:
            // case clang::Builtin::BI__builtin_fmal:
            // case clang::Builtin::BI__builtin_fmaf128:

            // case clang::Builtin::BIfmax:
            // case clang::Builtin::BIfmaxf:
            // case clang::Builtin::BIfmaxl:
            // case clang::Builtin::BI__builtin_fmax:
            // case clang::Builtin::BI__builtin_fmaxf:
            // case clang::Builtin::BI__builtin_fmaxf16:
            // case clang::Builtin::BI__builtin_fmaxl:
            // case clang::Builtin::BI__builtin_fmaxf128:

            // case clang::Builtin::BIfmin:
            // case clang::Builtin::BIfminf:
            // case clang::Builtin::BIfminl:
            // case clang::Builtin::BI__builtin_fmin:
            // case clang::Builtin::BI__builtin_fminf:
            // case clang::Builtin::BI__builtin_fminf16:
            // case clang::Builtin::BI__builtin_fminl:
            // case clang::Builtin::BI__builtin_fminf128:

            //// fmod() is a special-case. It maps to the frem instruction rather than an
            //// LLVM intrinsic.
            // case clang::Builtin::BIfmod:
            // case clang::Builtin::BIfmodf:
            // case clang::Builtin::BIfmodl:
            // case clang::Builtin::BI__builtin_fmod:
            // case clang::Builtin::BI__builtin_fmodf:
            // case clang::Builtin::BI__builtin_fmodf16:
            // case clang::Builtin::BI__builtin_fmodl:
            // case clang::Builtin::BI__builtin_fmodf128:

            // case clang::Builtin::BIlog:
            // case clang::Builtin::BIlogf:
            // case clang::Builtin::BIlogl:
            // case clang::Builtin::BI__builtin_log:
            // case clang::Builtin::BI__builtin_logf:
            // case clang::Builtin::BI__builtin_logf16:
            // case clang::Builtin::BI__builtin_logl:
            // case clang::Builtin::BI__builtin_logf128:

            // case clang::Builtin::BIlog10:
            // case clang::Builtin::BIlog10f:
            // case clang::Builtin::BIlog10l:
            // case clang::Builtin::BI__builtin_log10:
            // case clang::Builtin::BI__builtin_log10f:
            // case clang::Builtin::BI__builtin_log10f16:
            // case clang::Builtin::BI__builtin_log10l:
            // case clang::Builtin::BI__builtin_log10f128:

            // case clang::Builtin::BIlog2:
            // case clang::Builtin::BIlog2f:
            // case clang::Builtin::BIlog2l:
            // case clang::Builtin::BI__builtin_log2:
            // case clang::Builtin::BI__builtin_log2f:
            // case clang::Builtin::BI__builtin_log2f16:
            // case clang::Builtin::BI__builtin_log2l:
            // case clang::Builtin::BI__builtin_log2f128:

            // case clang::Builtin::BInearbyint:
            // case clang::Builtin::BInearbyintf:
            // case clang::Builtin::BInearbyintl:
            // case clang::Builtin::BI__builtin_nearbyint:
            // case clang::Builtin::BI__builtin_nearbyintf:
            // case clang::Builtin::BI__builtin_nearbyintl:
            // case clang::Builtin::BI__builtin_nearbyintf128:

            // case clang::Builtin::BIpow:
            // case clang::Builtin::BIpowf:
            // case clang::Builtin::BIpowl:
            // case clang::Builtin::BI__builtin_pow:
            // case clang::Builtin::BI__builtin_powf:
            // case clang::Builtin::BI__builtin_powf16:
            // case clang::Builtin::BI__builtin_powl:
            // case clang::Builtin::BI__builtin_powf128:

            // case clang::Builtin::BIrint:
            // case clang::Builtin::BIrintf:
            // case clang::Builtin::BIrintl:
            // case clang::Builtin::BI__builtin_rint:
            // case clang::Builtin::BI__builtin_rintf:
            // case clang::Builtin::BI__builtin_rintf16:
            // case clang::Builtin::BI__builtin_rintl:
            // case clang::Builtin::BI__builtin_rintf128:

            // case clang::Builtin::BIround:
            // case clang::Builtin::BIroundf:
            // case clang::Builtin::BIroundl:
            // case clang::Builtin::BI__builtin_round:
            // case clang::Builtin::BI__builtin_roundf:
            // case clang::Builtin::BI__builtin_roundf16:
            // case clang::Builtin::BI__builtin_roundl:
            // case clang::Builtin::BI__builtin_roundf128:

            // case clang::Builtin::BIsin:
            // case clang::Builtin::BIsinf:
            // case clang::Builtin::BIsinl:
            // case clang::Builtin::BI__builtin_sin:
            // case clang::Builtin::BI__builtin_sinf:
            // case clang::Builtin::BI__builtin_sinf16:
            // case clang::Builtin::BI__builtin_sinl:
            // case clang::Builtin::BI__builtin_sinf128:

            // case clang::Builtin::BIsqrt:
            // case clang::Builtin::BIsqrtf:
            // case clang::Builtin::BIsqrtl:
            // case clang::Builtin::BI__builtin_sqrt:
            // case clang::Builtin::BI__builtin_sqrtf:
            // case clang::Builtin::BI__builtin_sqrtf16:
            // case clang::Builtin::BI__builtin_sqrtl:
            // case clang::Builtin::BI__builtin_sqrtf128:

            // case clang::Builtin::BItrunc:
            // case clang::Builtin::BItruncf:
            // case clang::Builtin::BItruncl:
            // case clang::Builtin::BI__builtin_trunc:
            // case clang::Builtin::BI__builtin_truncf:
            // case clang::Builtin::BI__builtin_truncf16:
            // case clang::Builtin::BI__builtin_truncl:
            // case clang::Builtin::BI__builtin_truncf128:

            // case clang::Builtin::BIlround:
            // case clang::Builtin::BIlroundf:
            // case clang::Builtin::BIlroundl:
            // case clang::Builtin::BI__builtin_lround:
            // case clang::Builtin::BI__builtin_lroundf:
            // case clang::Builtin::BI__builtin_lroundl:
            // case clang::Builtin::BI__builtin_lroundf128:

            // case clang::Builtin::BIllround:
            // case clang::Builtin::BIllroundf:
            // case clang::Builtin::BIllroundl:
            // case clang::Builtin::BI__builtin_llround:
            // case clang::Builtin::BI__builtin_llroundf:
            // case clang::Builtin::BI__builtin_llroundl:
            // case clang::Builtin::BI__builtin_llroundf128:

            // case clang::Builtin::BIlrint:
            // case clang::Builtin::BIlrintf:
            // case clang::Builtin::BIlrintl:
            // case clang::Builtin::BI__builtin_lrint:
            // case clang::Builtin::BI__builtin_lrintf:
            // case clang::Builtin::BI__builtin_lrintl:
            // case clang::Builtin::BI__builtin_lrintf128:

            // case clang::Builtin::BIllrint:
            // case clang::Builtin::BIllrintf:
            // case clang::Builtin::BIllrintl:
            // case clang::Builtin::BI__builtin_llrint:
            // case clang::Builtin::BI__builtin_llrintf:
            // case clang::Builtin::BI__builtin_llrintl:
            // case clang::Builtin::BI__builtin_llrintf128:
            // case clang::Builtin::BIprintf:

            // C stdarg builtins.
            case clang::Builtin::BI__builtin_stdarg_start:
            case clang::Builtin::BI__builtin_va_start:
            case clang::Builtin::BI__va_start:
                return visitor.template visit< hlbi::VAStartOp >(
                    op, op->getResultTypes(), operands
                );
            case clang::Builtin::BI__builtin_va_end:
                return visitor.template visit< hlbi::VAEndOp >(
                    op, op->getResultTypes(), operands
                );
            case clang::Builtin::BI__builtin_va_copy:
                return visitor.template visit< hlbi::VACopyOp >(
                    op, op->getResultTypes(), operands
                );

            // case clang::Builtin::BI__builtin_expect:
            // case clang::Builtin::BI__builtin_expect_with_probability:

            // case clang::Builtin::BI__builtin_unpredictable:

            // case clang::Builtin::BImove:
            // case clang::Builtin::BImove_if_noexcept:
            // case clang::Builtin::BIforward:
            // case clang::Builtin::BIas_const:
            // case clang::Builtin::BI__GetExceptionInfo:

            // case clang::Builtin::BI__fastfail:

            // case clang::Builtin::BI__builtin_coro_id:
            // case clang::Builtin::BI__builtin_coro_promise:
            // case clang::Builtin::BI__builtin_coro_resume:
            // case clang::Builtin::BI__builtin_coro_noop:
            // case clang::Builtin::BI__builtin_coro_destroy:
            // case clang::Builtin::BI__builtin_coro_done:
            // case clang::Builtin::BI__builtin_coro_alloc:
            // case clang::Builtin::BI__builtin_coro_begin:
            // case clang::Builtin::BI__builtin_coro_end:
            // case clang::Builtin::BI__builtin_coro_suspend:
            // case clang::Builtin::BI__builtin_coro_align:

            // case clang::Builtin::BI__builtin_coro_frame:
            // case clang::Builtin::BI__builtin_coro_free:
            // case clang::Builtin::BI__builtin_coro_size:
            // case clang::Builtin::BI__builtin_dynamic_object_size:
            // case clang::Builtin::BI__builtin_object_size:

            // case clang::Builtin::BI__builtin_unreachable:

            // case clang::Builtin::BImemcpy:
            // case clang::Builtin::BI__builtin_memcpy:
            // case clang::Builtin::BImempcpy:
            // case clang::Builtin::BI__builtin_mempcpy:

            // case clang::Builtin::BI__builtin_clrsb:
            // case clang::Builtin::BI__builtin_clrsbl:
            // case clang::Builtin::BI__builtin_clrsbll:

            // case clang::Builtin::BI__builtin_ctzs:
            // case clang::Builtin::BI__builtin_ctz:
            // case clang::Builtin::BI__builtin_ctzl:
            // case clang::Builtin::BI__builtin_ctzll:

            // case clang::Builtin::BI__builtin_clzs:
            // case clang::Builtin::BI__builtin_clz:
            // case clang::Builtin::BI__builtin_clzl:
            // case clang::Builtin::BI__builtin_clzll:

            // case clang::Builtin::BI__builtin_ffs:
            // case clang::Builtin::BI__builtin_ffsl:
            // case clang::Builtin::BI__builtin_ffsll:

            // case clang::Builtin::BI__builtin_parity:
            // case clang::Builtin::BI__builtin_parityl:
            // case clang::Builtin::BI__builtin_parityll:

            // case clang::Builtin::BI__popcnt16:
            // case clang::Builtin::BI__popcnt:
            // case clang::Builtin::BI__popcnt64:
            // case clang::Builtin::BI__builtin_popcount:
            // case clang::Builtin::BI__builtin_popcountl:
            // case clang::Builtin::BI__builtin_popcountll:
            //
            case clang::Builtin::BI__builtin_debugtrap:
                return visitor.template visit< hlbi::DebugTrapOp >(op, op->getResultTypes());
            case clang::Builtin::BI__builtin_trap:
                return visitor.template visit< hlbi::TrapOp >(op, op->getResultTypes());

            default:
                return mlir::failure();
        }
    }

    template< typename call_op >
    struct convert_builtin_operation : operation_conversion_pattern< call_op >
    {
        using base = operation_conversion_pattern< call_op >;
        using base::base;

        using adaptor_t = call_op::Adaptor;

        logical_result matchAndRewrite(call_op op, adaptor_t ops, conversion_rewriter &cw) const override {
            return visit_builtin_op(op.getOperation(), ops.getOperands(), rewriter_visitor(cw));
        }

        static void legalize(conversion_target &trg) {
            trg.addDynamicallyLegalOp< call_op >([](operation op) -> bool {
                return visit_builtin_op(op, op->getOperands(), identity_visitor()).failed();
            });
        }
    };

    struct HLToHLBIPass : ModuleConversionPassMixin< HLToHLBIPass, HLToHLBIBase >
    {
        using base     = ModuleConversionPassMixin< HLToHLBIPass, HLToHLBIBase >;
        using config_t = typename base::config_t;

        static conversion_target create_conversion_target(mcontext_t &context) {
            conversion_target target(context);
            target.addLegalDialect< hlbi::HLBuiltinDialect >();
            return target;
        }

        static void populate_conversions(config_t &config) {
            populate_conversions_base<
                util::type_list<
                    convert_builtin_operation< hl::CallOp >
                >
            >(config);
        }
    };
} // namespace vast::conv

std::unique_ptr< mlir::Pass > vast::createHLToHLBI() {
    return std::make_unique< vast::conv::HLToHLBIPass >();
}

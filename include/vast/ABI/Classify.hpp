// Copyright (c) 2022-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <llvm/Support/Alignment.h>
VAST_UNRELAX_WARNINGS

#include "vast/ABI/ABI.hpp"

#include "vast/Dialect/Core/CoreOps.hpp"
#include "vast/Dialect/HighLevel/HighLevelUtils.hpp"

namespace vast::abi {

    // Stateful - one object should be use per one function classification.
    // TODO(abi): For now this serves as x86_64, later core parts will be extracted.
    // TODO(codestyle): See if `arg_` and `ret_` can be unified - in clang they are separately
    //                  but it may be just a bad codestyle.
    template< typename FnInfo, typename TypeInfo >
    struct classifier_base
    {
        using self_t      = classifier_base< FnInfo, TypeInfo >;

        using func_info   = FnInfo;
        using type_info_t = TypeInfo;

        using ir_type  = typename func_info::type;
        using ir_types = typename func_info::types;

        func_info info;

        // Not a `const &` because `iN` and `fN` methods are not required to be `const` due to some
        // internals, plus this class is supossed to be lightweight.
        type_info_t type_info;

        static constexpr std::size_t max_gpr = 6;
        static constexpr std::size_t max_sse = 8;

        std::size_t needed_int = 0;
        std::size_t needed_sse = 0;

        classifier_base(func_info info, const type_info_t &type_info)
            : info(std::move(info)), type_info(type_info) {}

        auto size(ir_type t) { return type_info.size(t); }
        auto align(ir_type t) { return type_info.align(t); }

        // [ start, end )
        bool bits_contain_no_user_data(ir_type t, std::size_t start, std::size_t end, operation from) {
            if (type_info.size(t) <= start) {
                return true;
            }

            if (type_info.is_record(t) || type_info.is_array(t)) {
                // TODO(abi): CXXRecordDecl.
                auto fields = type_info.fields(t, from);

                std::size_t current = 0;
                auto field_range = fields | std::views::take_while([&](auto){ return current < end; });

                auto handle_field = [&](auto field) {
                    bool result = bits_contain_no_user_data(field, current, end - start, from);
                    current += type_info.size(field);
                    return result;
                };

                return std::ranges::all_of(field_range, handle_field);
            }

            return false;
        }

        ir_type int_type_at_offset(
            ir_type t, std::size_t offset, ir_type root, std::size_t root_offset, operation from
        ) {
            auto is_int_type    = [&](std::size_t trg_size) {
                return type_info.is_scalar_integer(t) && size(t) == trg_size;
            };

            if (offset == 0) {
                if ((type_info.is_pointer(t) && type_info.pointer_size() == 64) || is_int_type(64))
                    return t;

                if (is_int_type(8) || is_int_type(16) || is_int_type(32)) {
                    // TODO(abi): Here should be check if `BitsContainNoUserData` - however
                    //            for now it should be safe to always pretend to it being `false`?
                    if (bits_contain_no_user_data(root, offset + size(t), root_offset + 64, from))
                        return t;
                }
            }

            // We need to extract a field on current offset.
            // TODO(abi): This is done differently than clang, since they seem to be using
            //            underflow? on offset?
            if (type_info.is_struct(t) && size(t) > 64) {
                auto field_info = type_info.field_containing_offset(t, offset, from);
                VAST_ASSERT(field_info);
                auto [field, field_start] = *field_info;
                return int_type_at_offset(field, offset - field_start, root, root_offset, from);
            }

            if (type_info.is_array(t)) {
                auto [_, element_type] = type_info.array_info(t);
                auto element_size = size(element_type);
                auto element_offset = offset / element_size * element_size;
                return int_type_at_offset(element_type, offset - element_offset, root, root_offset, from);
            }

            VAST_CHECK(size(root) != 0, "Unexpected empty field? Type: {0}", t);

            auto final_size = std::min< std::size_t >(size(root) - (root_offset * 8), 64);
            // TODO(abi): Issue #422
            // This is definitely not expected right now.
            VAST_CHECK(final_size != 0, "ABI classification is trying to create i0.");
            return type_info.iN(final_size);
        }

        auto get_is_sse_type() {
            return [&](auto t) { return type_info.is_scalar_float(t); };
        }

        ir_type sse_target_type_at_offset(
            ir_type t, std::size_t offset, ir_type root, std::size_t root_offset, operation from
        ) {
            auto t0 = type_info.type_at_offset(t, offset, get_is_sse_type(), from);
            if (!t0 || size(t0) == 64) {
                return type_info.fN(64);
            }

            auto source_size = size(root);
            auto t0_size     = size(t0) - root_offset;

            auto is_16b_fp = [&](auto type) {
                return type_info.is_scalar_float(type) && size(type) == 16;
            };

            // Fetch second type if applicable.
            auto t1 = [&]() -> ir_type {
                auto nested = type_info.type_at_offset(
                    t, offset + t0_size, get_is_sse_type(), from
                );

                if (t0_size > source_size && nested) {
                    return nested;
                }

                // We know `nested` was either not retrieved or that the sizes are not
                // good enough.

                // Check case of half/bfloat + float.
                if (size(t0) == 16 && source_size > 4 * 8) {
                    // `+4` comes from alignement
                    return type_info.type_at_offset(t, offset + 32, get_is_sse_type(), from);
                }
                return {};
            }();

            // If we cannot get second type return first. It seems that `i8` would work
            // as well.
            if (!t1) {
                return t0;
            }

            // TODO: There is a bunch of conditions regarding size of `t0/t1` that
            //       return `llvm::FixedVector` types, which we currently cannot really
            //       work with anyway.
            VAST_CHECK(!(type_info.is_scalar_float(t0) && type_info.is_scalar_float(t1)),
                       "Not yet supported");
            VAST_CHECK(!(is_16b_fp(t0) || is_16b_fp(t1)), "Not yet supported");

            // Default case returns double.
            return type_info.fN(64);
        }

        // Enum for classification algorithm.
        enum class Class : uint32_t {
            Integer = 0,
            SSE,
            SSEUp,
            X87,
            X87Up,
            ComplexX87,
            Memory,
            NoClass
        };
        using classification_t = std::tuple< Class, Class >;

        static std::string to_string(Class c) {
            switch (c) {
                case Class::Integer:
                    return "Integer";
                case Class::SSE:
                    return "SSE";
                case Class::SSEUp:
                    return "SSEUp";
                case Class::X87:
                    return "X87";
                case Class::X87Up:
                    return "X87Up";
                case Class::ComplexX87:
                    return "ComplexX87";
                case Class::Memory:
                    return "Memory";
                case Class::NoClass:
                    return "NoClass";
            }
        }

        static std::string to_string(classification_t c) {
            auto [lo, hi] = c;
            return "[ " + to_string(lo) + ", " + to_string(hi) + " ]";
        }

        static Class join(Class a, Class b) {
            if (a == b) {
                return a;
            }

            if (a == Class::NoClass) {
                return b;
            }
            if (b == Class::NoClass) {
                return a;
            }

            if (a == Class::Memory || b == Class::Memory) {
                return Class::Memory;
            }

            if (a == Class::Integer || b == Class::Integer) {
                return Class::Integer;
            }

            auto use_mem = [&](auto x) {
                return x == Class::X87 || x == Class::X87Up || x == Class::ComplexX87;
            };

            if (use_mem(a) || use_mem(b)) {
                return Class::Memory;
            }

            return Class::SSE;
        }

        static classification_t join(classification_t a, classification_t b) {
            auto [a1, a2] = a;
            auto [b1, b2] = b;
            return { join(a1, b1), join(a2, b2) };
        }

        classification_t get_class(ir_type t, std::size_t &offset, operation from) {
            auto mk_classification = [&](auto c) -> classification_t {
                if (offset < 8 * 8) {
                    return { c, Class::NoClass };
                }
                return { Class::NoClass, c };
            };

            // First we handle all "builtin" types.

            if (type_info.is_void(t)) {
                return { Class::NoClass, Class::NoClass };
            }

            if (type_info.is_scalar_integer(t)) {
                // _Bool, char, short, int, long, long long
                if (size(t) <= 64) {
                    return mk_classification(Class::Integer);
                }
                // __int128
                return { Class::Integer, Class::Integer };
            }

            if (type_info.represents_pointer(t)) {
                return { Class::Integer, Class::NoClass };
            }

            // Float, Float16, Double, BFloat16
            if (type_info.is_scalar_float(t)) {
                // float, double, _Decimal32, _Decimal64, __m64
                if (size(t) <= 64) {
                    return mk_classification(Class::SSE);
                }
                // __float128, _Decimal128, __m128
                return { Class::SSE, { Class::SSEUp } };

                // TODO(abi): __m256
                // TODO(abi): long double
            }

            // TODO(abi): complex
            // TODO(abi): complex long double

            return get_aggregate_class(t, offset, from);
        }

        classification_t get_aggregate_class(ir_type t, std::size_t &offset, operation from) {
            if (size(t) > 8 * 64 || type_info.has_unaligned_field(t)) {
                return { Class::Memory, {} };
            }
            // TODO(abi): C++ perks.

            auto fields             = type_info.fields(t, from);
            classification_t result = { Class::NoClass, Class::NoClass };

            auto field_offset = offset;
            for (auto field_type : fields) {
                auto field_class  = classify(field_type, field_offset, from);
                field_offset     += size(field_type);
                result            = join(result, field_class);
            }

            offset += size(t);
            return post_merge(t, result);
        }

        classification_t post_merge(ir_type t, classification_t c) {
            auto [lo, hi] = c;
            if (lo == Class::Memory || hi == Class::Memory) {
                return { Class::Memory, Class::Memory };
            }

            // Introduced in some revision.
            if (hi == Class::X87Up && lo != Class::X87) {
                VAST_ASSERT(false);
                lo = Class::Memory;
            }

            if (size(t) > 128 && (lo != Class::SSE || hi != Class::SSEUp)) {
                lo = Class::Memory;
            }

            if (hi == Class::SSEUp && lo != Class::SSE) {
                return { lo, Class::SSE };
            }
            return { lo, hi };
        }

        auto classify(ir_type raw, std::size_t &offset, operation from) {
            auto t = type_info.prepare(raw);
            return get_class(t, offset, from);
        }

        auto classify(ir_type raw, operation from) {
            std::size_t offset = 0;
            return classify(raw, offset, from);
        }

        using half_class_result = std::variant< arg_info, ir_type, std::monostate >;

        static inline std::string to_string(const half_class_result &a) {
            if (auto arg = std::get_if< arg_info >(&a)) {
                return arg->to_string();
            }
            if (auto t = std::get_if< ir_type >(&a)) {
                return "type";
            }
            return "monostate";
        }

        // Both parts are passed as argument.
        half_class_result return_lo(ir_type t, classification_t c, operation from) {
            auto [lo, hi] = c;
            // Check are taken from clang, to not diverge from their implementation for now.
            switch (lo) {
                case Class::NoClass: {
                    if (hi == Class::NoClass) {
                        return { arg_info::make< ignore >(t) };
                    }
                    // Missing lo part.
                    VAST_ASSERT(hi == Class::SSE || hi == Class::Integer || hi == Class::X87Up);
                    [[fallthrough]];
                }

                case Class::SSEUp:
                case Class::X87Up:
                    VAST_UNREACHABLE("Wrong class");

                case Class::Memory:
                    // TODO(abi): Inject type.
                    return { arg_info::make< indirect >(type_info.as_pointer(t)) };

                case Class::Integer: {
                    auto target_type = int_type_at_offset(t, 0, t, 0, from);
                    // TODO(abi): get integer type for the slice.
                    if (type_info.is_scalar_integer(t) && type_info.can_be_promoted(t)) {
                        return { arg_info::make< extend >(target_type) };
                    }
                    return { target_type };
                }

                case Class::SSE:
                    return { t };
                default:
                    VAST_UNREACHABLE("Wrong class");
            }
        }

        half_class_result return_hi(ir_type t, classification_t c, operation from) {
            auto [lo, hi] = c;
            switch (hi) {
                case Class::Memory:
                case Class::X87:
                    VAST_UNREACHABLE("Wrong class");
                case Class::ComplexX87:
                case Class::NoClass:
                    return { std::monostate{} };
                case Class::Integer: {
                    auto target_type = int_type_at_offset(t, 64, t, 64, from);
                    if (lo == Class::NoClass) {
                        return { arg_info::make< direct >(target_type) };
                    }
                    return { target_type };
                }
                case Class::SSE:
                case Class::SSEUp:
                case Class::X87Up:
                    VAST_UNREACHABLE("Wrong class");
            }
        }

        // TODO(abi): Implement. Requires information about alignment and size of alloca.
        ir_types combine_half_types(ir_type lo, ir_type hi) {
            VAST_CHECK(size(hi) / 8 != 0, "{0}", hi);
            auto hi_start = llvm::alignTo(size(lo) / 8, align(hi));
            VAST_CHECK(hi_start != 0 && hi_start / 8 <= 8, "{0} {1} {2}", lo, hi, hi_start);

            // `hi` needs to start at later offset - we need to add explicit padding
            // to the `lo` type.
            auto adjusted_lo = [&]() -> ir_type {
                if (hi_start == 8) {
                    return lo;
                }
                if (type_info.is_scalar_integer(lo) || type_info.is_pointer(lo)) {
                    return type_info.iN(64);
                }
                // TODO(abi): float, half -> promote to double
                if (type_info.is_scalar_float(lo) && size(lo) < 32) {
                    return type_info.fN(64);
                }
                VAST_UNREACHABLE("Cannot combine half types for {0}, {1}.", lo, hi);
            }();

            return { adjusted_lo, hi };
        }

        arg_info resolve_classification(ir_type t, half_class_result low, half_class_result high) {
            // If either returned a result it should be used.
            // TODO(abi): Should `high` be allowed to return `arg_info`?
            if (auto out = get_if< arg_info >(&low)) {
                return std::move(*out);
            }
            if (auto out = get_if< arg_info >(&high)) {
                return std::move(*out);
            }

            if (holds_alternative< std::monostate >(high)) {
                auto coerced_type = get_if< ir_type >(&low);
                VAST_ASSERT(coerced_type);
                // TODO(abi): Pass in `coerced_type`.
                return arg_info::make< direct >(*coerced_type);
            }

            // Both returned types, we need to combine them.
            auto lo_type = get_if< ir_type >(&low);
            auto hi_type = get_if< ir_type >(&high);
            VAST_ASSERT(lo_type && hi_type);

            auto res_type = combine_half_types(*lo_type, *hi_type);
            return arg_info::make< direct >(res_type);
        }

        arg_info classify_return(ir_type t, operation from) {
            if (type_info.is_void(t)) {
                return arg_info::make< ignore >(t);
            }

            if (auto record = type_info.is_record(t)) {
                if (!type_info.can_be_passed_in_regs(t)) {
                    return arg_info::make< indirect >(type_info.as_pointer(t));
                }
            }

            // Algorithm based on AMD64-ABI
            auto c = classify(t, from);

            auto low  = return_lo(t, c, from);
            auto high = return_hi(t, c, from);

            return resolve_classification(t, std::move(low), std::move(high));
        }

        // Integer, SSE
        using reg_usage = std::tuple< std::size_t, std::size_t >;
        // TODO: This is pretty convoluted.
        using arg_class = std::tuple< half_class_result, reg_usage >;

        half_class_result arg_lo(ir_type t, classification_t c, operation from) {
            auto [lo, hi] = c;

            switch (lo) {
                case Class::NoClass: {
                    if (hi == Class::NoClass) {
                        return { arg_info::make< ignore >(t) };
                    }
                    VAST_ASSERT(hi == Class::SSE || hi == Class::Integer || hi == Class::X87Up);
                    return { std::monostate{} };
                }
                case Class::Memory:
                case Class::X87:
                case Class::ComplexX87: {
                    // TODO(abi): Some C++
                    // TODO(abi): This is more nuanced, see `getIndirectResult` in clang.
                    return { arg_info::make< indirect >(type_info.as_pointer(t)) };
                }

                case Class::SSEUp:
                case Class::X87Up: {
                    // This should actually not happen.
                    VAST_UNREACHABLE("This shouldn't happen");
                }

                case Class::Integer: {
                    ++needed_int;
                    auto target_type = int_type_at_offset(t, 0, t, 0, from);
                    // TODO(abi): Or enum.
                    if (hi == Class::NoClass && type_info.is_scalar_integer(target_type)) {
                        // TODO(abi): If enum, treat as underlying type.
                        if (type_info.can_be_promoted(target_type)) {
                            return { arg_info::make< extend >(target_type) };
                        }
                    }

                    return { target_type };
                }

                case Class::SSE: {
                    ++needed_sse;
                    auto target_type = sse_target_type_at_offset(t, 0, t, 0, from);
                    return { target_type };
                }
            }
        }

        half_class_result arg_hi(ir_type t, classification_t c, operation from) {
            auto [lo, hi] = c;
            switch (hi) {
                case Class::Memory:
                case Class::X87:
                case Class::ComplexX87:
                    VAST_UNREACHABLE("Invalid classification for arg_hi: {0}", to_string(hi));

                case Class::NoClass:
                    return { std::monostate{} };

                case Class::Integer: {
                    ++needed_int;
                    auto target_type = int_type_at_offset(t, 64, t, 64, from);
                    if (lo == Class::NoClass) {
                        return { arg_info::make< direct >(target_type) };
                    }
                    return { target_type };
                }

                case Class::X87Up: {
                    // Should follow the same rules as SSE
                    VAST_TODO("arg_hi::X87Up");
                }
                case Class::SSE: {
                    ++needed_sse;
                    auto target_type = sse_target_type_at_offset(t, 64, t, 64, from);
                    if (lo == Class::NoClass) {
                        return { arg_info::make< direct >(target_type) };
                    }
                    return { target_type };
                }
                case Class::SSEUp: {
                    VAST_TODO("arg_hi::SSEUp");
                }
            }
        }

        arg_info classify_arg(ir_type t, operation from) {
            auto c    = classify(t, from);
            auto low  = arg_lo(t, c, from);
            auto high = arg_hi(t, c, from);
            return resolve_classification(t, std::move(low), std::move(high));
        }

        self_t &compute_abi(operation from) {
            info.add_return(classify_return(
                type_info.prepare(info.return_type()), from
            ));
            for (auto arg : info.fn_type().getInputs()) {
                info.add_arg(classify_arg(type_info.prepare(arg), from));
            }
            return *this;
        }

        func_info take() { return std::move(info); }
    };

    template< typename FnInfo, typename TypeInfo >
    classifier_base(FnInfo, TypeInfo) -> classifier_base< FnInfo, TypeInfo >;

} // namespace vast::abi

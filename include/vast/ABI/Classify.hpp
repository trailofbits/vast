// Copyright (c) 2022-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <llvm/Support/Alignment.h>
VAST_UNRELAX_WARNINGS

#include "vast/ABI/ABI.hpp"

#include "vast/Dialect/HighLevel/HighLevelUtils.hpp"

namespace vast::abi {
    template< typename H, typename... Tail >
    auto maybe_strip(mlir::Type t) {
        auto casted = [&]() -> mlir::Type {
            auto c = t.dyn_cast< H >();
            if (c) {
                return c.getElementType();
            }
            return t;
        }();

        if constexpr (sizeof...(Tail) == 0) {
            return casted;
        } else {
            return maybe_strip< Tail... >(casted);
        }
    }

    struct TypeConfig
    {
        static bool is_void(mlir::Type t) { return t.isa< mlir::NoneType >(); }

        static bool is_aggregate(mlir::Type t) {
            // TODO(abi): `SubElementTypeInterface` is not good enough, since for
            //             example hl::PointerType implements it.
            // TODO(abi): Figure how to better handle this than manual listing.
            return t.isa< hl::RecordType >() || t.isa< hl::ElaboratedType >()
                || t.isa< hl::ArrayType >();
        }

        static bool is_scalar(mlir::Type t) {
            // TODO(abi): Also complex is an option in ABI.
            return !is_aggregate(t);
        }

        static bool is_complex(mlir::Type t) { VAST_UNREACHABLE(""); }

        static bool is_record(mlir::Type t) {
            if (t.isa< hl::RecordType >()) {
                return true;
            }
            if (auto et = t.dyn_cast< hl::ElaboratedType >()) {
                return is_record(et.getElementType());
            }
            return false;
        }

        static bool is_struct(mlir::Type t) {
            // TODO(abi): Are these equivalent?
            return is_record(t);
        }

        static bool is_array(mlir::Type t) {
            return maybe_strip< hl::ElaboratedType >(t).isa< hl::ArrayType >();
        }

        // Must be invoked with `t` that is array-like.
        static auto array_info(mlir_type t) -> std::tuple< std::optional< std::size_t >, mlir_type > {
            auto array_type = mlir::dyn_cast< hl::ArrayType >(t);
            VAST_ASSERT(array_type);
            return { array_type.getSize(), array_type.getElementType() };
        }

        static bool can_be_passed_in_regs(mlir::Type t) {
            // TODO(abi): Seems like in C nothing can prevent this.
            return true;
        }

        static bool is_scalar_integer(mlir::Type t) { return t.isa< mlir::IntegerType >(); }

        // TODO(abi): Implement.
        static bool has_unaligned_field(mlir::Type t) { return false; }

        static bool is_scalar_float(mlir::Type t) { return t.isa< mlir::FloatType >(); }

        static bool is_pointer(mlir::Type t) { return t.isa< hl::PointerType >(); }

        // Pointers, references etc
        static bool represents_pointer(mlir_type t) { return is_pointer(t); }

        static mlir_type as_pointer(mlir_type t) {
            return hl::PointerType::get(t.getContext(), t);
        }

        static hl::StructDeclOp get_struct_def(hl::RecordType t, mlir::ModuleOp m) {
            for (auto &op : *m.getBody()) {
                if (auto decl = mlir::dyn_cast< hl::StructDeclOp >(op)) {
                    if (decl.getName() == t.getName()) {
                        return decl;
                    }
                }
            }
            return {};
        }

        static bool can_be_promoted(mlir::Type t) { return false; }

        static mlir::Type prepare(mlir::Type t) { return maybe_strip< hl::LValueType >(t); }

        // TODO(abi): Will need to live in a different interface.
        static std::size_t pointer_size() { return 64; }

        static std::size_t size(const auto &dl, mlir::Type t) {
            return dl.getTypeSizeInBits(t);
        }

        // [ start, end )
        static bool bits_contain_no_user_data(
            mlir::Type t, std::size_t start, std::size_t end, const auto &classifier_ctx
        ) {
            const auto &[dl, op] = classifier_ctx;

            if (size(dl, t) <= start) {
                return true;
            }

            // TODO: Implement. For now we are saying the array is okay as in `C` this will be
            //       rarely broken.
            if (is_array(t)) {
                return true;
            }

            if (is_record(t)) {
                // TODO(abi): CXXRecordDecl.
                std::size_t current = 0;
                for (auto field : fields(t, op)) {
                    if (current >= end) {
                        break;
                    }
                    if (!bits_contain_no_user_data(field, current, end - start, classifier_ctx))
                    {
                        return false;
                    }

                    current += size(dl, t);
                }
                return true;
            }

            return false;
        }

        static mlir::Type iN(const auto &has_context, std::size_t s) {
            return mlir::IntegerType::get(has_context.getContext(), s);
        }

        static mlir_type fN(auto &mctx, std::size_t s) {
            if (s == 16) {
                return mlir::Float16Type::get(&mctx);
            }
            if (s == 32) {
                return mlir::Float32Type::get(&mctx);
            }
            if (s == 64) {
                return mlir::Float64Type::get(&mctx);
            }
            if (s == 80) {
                return mlir::Float80Type::get(&mctx);
            }
            if (s == 128) {
                return mlir::Float128Type::get(&mctx);
            }
            VAST_UNREACHABLE("Cannot create floating type of size: {0}", s);
        }

        static mlir::Type int_type_at_offset(
            mlir::Type t, std::size_t offset, mlir::Type root, std::size_t root_offset,
            const auto &classifier_ctx
        ) {
            const auto &[dl, _] = classifier_ctx;
            auto is_int_type    = [&](std::size_t trg_size) {
                const auto &[dl_, _] = classifier_ctx;
                return is_scalar_integer(t) && size(dl_, t) == trg_size;
            };

            if (offset == 0) {
                if ((is_pointer(t) && pointer_size() == 64) || is_int_type(64)) {
                    return t;
                }

                if (is_int_type(8) || is_int_type(16) || is_int_type(32)) {
                    // TODO(abi): Here should be check if `BitsContainNoUserData` - however
                    //            for now it should be safe to always pretend to it being `false`?
                    if (bits_contain_no_user_data(
                            root, offset + size(dl, t), root_offset + 64, classifier_ctx
                        ))
                    {
                        return t;
                    }
                }
            }

            // We need to extract a field on current offset.
            // TODO(abi): This is done differently than clang, since they seem to be using
            //            underflow? on offset?
            if (is_struct(t) && (size(dl, t) > 64)) {
                auto [field, field_start] = field_containing_offset(classifier_ctx, t, offset);
                VAST_ASSERT(field);
                return int_type_at_offset(
                    field, offset - field_start, root, root_offset, classifier_ctx
                );
            }

            if (is_array(t)) {
                auto [_, element_type] = array_info(t);
                auto element_size = size(dl, element_type);
                auto element_offset = root_offset / element_size * element_size;
                return int_type_at_offset(
                    element_type, offset - element_offset,
                    root, root_offset, classifier_ctx);
            }

            auto type_size = size(dl, root);
            VAST_CHECK(type_size != 0, "Unexpected empty field? Type: {0}", t);

            auto final_size = std::min< std::size_t >(type_size - (root_offset * 8), 64);
            // TODO(abi): Issue #422
            // This is definitely not expected right now.
            VAST_CHECK(final_size != 0, "ABI classification is trying to create i0.");
            return mlir::IntegerType::get(t.getContext(), final_size);
        }

        static auto get_is_sse_type() {
            return [](auto t) { return is_scalar_float(t); };
        }

        // TODO: In `clang` int and sse variants have separate implementations (there is
        //       a lot of difference) but in the core they are doing very similar things.
        //       It should be possible to make a shared helper that takes away a lot of
        //       shared complexity.
        static mlir_type sse_target_type_at_offset(
            mlir_type t, std::size_t offset, mlir_type root, std::size_t root_offset,
            const auto &classifier_ctx
        ) {
            const auto &[dl, _] = classifier_ctx;
            auto &mctx          = *t.getContext();

            auto t0 = type_at_offset(t, offset, classifier_ctx, get_is_sse_type());
            if (!t0 || size(dl, t0) == 64) {
                return fN(mctx, 64);
            }

            auto source_size = size(dl, root);
            auto t0_size     = size(dl, t0) - root_offset;

            auto is_16b_fp = [&, dl = dl](auto type) {
                return is_scalar_float(type) && size(dl, type) == 16;
            };

            // Fetch second type if applicable.
            auto t1 = [&]() -> mlir_type {
                auto nested =
                    type_at_offset(t, offset + t0_size, classifier_ctx, get_is_sse_type());
                if (t0_size > source_size && nested) {
                    return nested;
                }

                // We know `nested` was either not retrieved or that the sizes are not
                // good enough.

                // Check case of half/bfloat + float.
                if (size(dl, t0) == 16 && source_size > 4) {
                    // `+4` comes from alignement
                    return type_at_offset(t, offset + 4, classifier_ctx, get_is_sse_type());
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
            VAST_CHECK(!(is_scalar_float(t0) && is_scalar_float(t1)), "Not yet supported");
            VAST_CHECK(!(is_16b_fp(t0) || is_16b_fp(t1)), "Not yet supported");

            // Default case returns double.
            return fN(mctx, 64);
        }

        static mlir_type type_at_offset(
            mlir_type t, std::size_t offset, const auto &classifier_ctx, const auto &accept
        ) {
            if (offset == 0 && accept(t)) {
                return t;
            }

            if (is_struct(t)) {
                auto [field, field_start] = field_containing_offset(classifier_ctx, t, offset);
                return type_at_offset(field, field_start, classifier_ctx, accept);
            }

            // TODO: Array support.
            VAST_CHECK(!is_array(t), "Floats in array type not yet supported!");

            return {};
        }

        static auto field_containing_offset(
            const auto &classifier_ctx, mlir::Type t, std::size_t offset
        ) -> std::tuple< mlir::Type, std::size_t > {
            const auto &[dl, op] = classifier_ctx;

            auto curr = 0;
            for (auto field : fields(t, op)) {
                if (curr + size(dl, field) > offset) {
                    return { field, curr };
                }
                curr += size(dl, field);
            }
            VAST_UNREACHABLE("Did not find field at offset {0} in {1}", offset, t);
        }

        static gap::generator< mlir_type > mock_array_fields(hl::ArrayType array_type) {
            auto size = array_type.getSize();
            VAST_CHECK(size, "Variable size array type not yet supported");

            auto et =array_type.getElementType();
            for (std::size_t i = 0; i < *size; ++i)
               co_yield et;
        }

        static auto fields(mlir_type type, auto func) {
            if (auto array_type = mlir::dyn_cast< hl::ArrayType >(type))
                return mock_array_fields(array_type);
            auto mod = func->template getParentOfType< vast_module >();
            return vast::hl::field_types(type, mod);
        }
    };

    // Stateful - one object should be use per one function classification.
    // TODO(abi): For now this serves as x86_64, later core parts will be extracted.
    // TODO(codestyle): See if `arg_` and `ret_` can be unified - in clang they are separately
    //                  but it may be just a bad codestyle.
    template< typename FnInfo, typename DL >
    struct classifier_base
    {
        using self_t      = classifier_base< FnInfo, DL >;
        using func_info   = FnInfo;
        using data_layout = DL;

        using type  = typename func_info::type;
        using types = typename func_info::types;

        func_info info;
        const data_layout &dl;

        static constexpr std::size_t max_gpr = 6;
        static constexpr std::size_t max_sse = 8;

        std::size_t needed_int = 0;
        std::size_t needed_sse = 0;

        classifier_base(func_info info, const data_layout &dl)
            : info(std::move(info)), dl(dl) {}

        auto size(mlir::Type t) { return dl.getTypeSizeInBits(t); }

        auto align(type) {
            // TODO(abi): Implement into data layout of vast stack.
            return 0;
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

        classification_t get_class(mlir::Type t, std::size_t &offset) {
            auto mk_classification = [&](auto c) -> classification_t {
                if (offset < 8 * 8) {
                    return { c, Class::NoClass };
                }
                return { Class::NoClass, c };
            };

            // First we handle all "builtin" types.

            if (TypeConfig::is_void(t)) {
                return { Class::NoClass, Class::NoClass };
            }

            if (TypeConfig::is_scalar_integer(t)) {
                // _Bool, char, short, int, long, long long
                if (size(t) <= 64) {
                    return mk_classification(Class::Integer);
                }
                // __int128
                return { Class::Integer, Class::Integer };
            }

            if (TypeConfig::represents_pointer(t)) {
                return { Class::Integer, Class::NoClass };
            }

            // Float, Float16, Double, BFloat16
            if (TypeConfig::is_scalar_float(t)) {
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

            return get_aggregate_class(t, offset);
        }

        // TODO(abi): Refactor.
        auto mk_classifier_ctx() const { return std::make_tuple(dl, info.raw_fn); }

        classification_t get_aggregate_class(mlir::Type t, std::size_t &offset) {
            if (size(t) > 8 * 64 || TypeConfig::has_unaligned_field(t)) {
                return { Class::Memory, {} };
            }
            // TODO(abi): C++ perks.

            auto fields             = TypeConfig::fields(t, info.raw_fn);
            classification_t result = { Class::NoClass, Class::NoClass };

            auto field_offset = offset;
            for (auto field_type : fields) {
                auto field_class  = classify(field_type, field_offset);
                field_offset     += size(field_type);
                result            = join(result, field_class);
            }

            offset += size(t);
            return post_merge(t, result);
        }

        classification_t post_merge(mlir::Type t, classification_t c) {
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

        auto classify(mlir::Type raw, std::size_t &offset) {
            auto t = TypeConfig::prepare(raw);
            return get_class(t, offset);
        }

        auto classify(mlir::Type raw) {
            std::size_t offset = 0;
            return classify(raw, offset);
        }

        using half_class_result = std::variant< arg_info, type, std::monostate >;

        static inline std::string to_string(const half_class_result &a) {
            if (auto arg = std::get_if< arg_info >(&a)) {
                return arg->to_string();
            }
            if (auto t = std::get_if< type >(&a)) {
                return "type";
            }
            return "monostate";
        }

        // Both parts are passed as argument.
        half_class_result return_lo(type t, classification_t c) {
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
                    return { arg_info::make< indirect >(TypeConfig::as_pointer(t)) };

                case Class::Integer: {
                    auto target_type =
                        TypeConfig::int_type_at_offset(t, 0, t, 0, mk_classifier_ctx());
                    // TODO(abi): get integer type for the slice.
                    if (TypeConfig::is_scalar_integer(t) && TypeConfig::can_be_promoted(t)) {
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

        half_class_result return_hi(type t, classification_t c) {
            auto [lo, hi] = c;
            switch (hi) {
                case Class::Memory:
                case Class::X87:
                    VAST_UNREACHABLE("Wrong class");
                case Class::ComplexX87:
                case Class::NoClass:
                    return { std::monostate{} };
                case Class::Integer: {
                    auto target_type =
                        TypeConfig::int_type_at_offset(t, 8, t, 8, mk_classifier_ctx());
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
        types combine_half_types(type lo, type hi) {
            VAST_CHECK(size(hi) / 8 != 0, "{0}", hi);
            auto hi_start = llvm::alignTo(size(lo) / 8, size(hi) / 8);
            VAST_CHECK(hi_start != 0 && hi_start / 8 <= 8, "{0} {1} {2}", lo, hi, hi_start);

            // `hi` needs to start at later offset - we need to add explicit padding
            // to the `lo` type.
            auto adjusted_lo = [&]() -> type {
                if (hi_start == 8) {
                    return lo;
                }
                // TODO(abi): float, half -> promote to double
                if (TypeConfig::is_scalar_integer(lo) || TypeConfig::is_pointer(lo)) {
                    return TypeConfig::iN(lo, 64);
                }
                if (TypeConfig::is_scalar_float(lo) && size(lo) < 32) {
                    return TypeConfig::fN(*lo.getContext(), 64);
                }
                VAST_UNREACHABLE("Cannot combine half types for {0}, {1}.", lo, hi);
            }();

            return { adjusted_lo, hi };
        }

        arg_info resolve_classification(type t, half_class_result low, half_class_result high) {
            // If either returned a result it should be used.
            // TODO(abi): Should `high` be allowed to return `arg_info`?
            if (auto out = get_if< arg_info >(&low)) {
                return std::move(*out);
            }
            if (auto out = get_if< arg_info >(&high)) {
                return std::move(*out);
            }

            if (holds_alternative< std::monostate >(high)) {
                auto coerced_type = get_if< type >(&low);
                VAST_ASSERT(coerced_type);
                // TODO(abi): Pass in `coerced_type`.
                return arg_info::make< direct >(*coerced_type);
            }

            // Both returned types, we need to combine them.
            auto lo_type = get_if< type >(&low);
            auto hi_type = get_if< type >(&high);
            VAST_ASSERT(lo_type && hi_type);

            auto res_type = combine_half_types(*lo_type, *hi_type);
            return arg_info::make< direct >(res_type);
        }

        arg_info classify_return(type t) {
            if (TypeConfig::is_void(t)) {
                return arg_info::make< ignore >(t);
            }

            if (auto record = TypeConfig::is_record(t)) {
                if (!TypeConfig::can_be_passed_in_regs(t)) {
                    return arg_info::make< indirect >(TypeConfig::as_pointer(t));
                }
            }

            // Algorithm based on AMD64-ABI
            auto c = classify(t);

            auto low  = return_lo(t, c);
            auto high = return_hi(t, c);

            return resolve_classification(t, std::move(low), std::move(high));
        }

        // Integer, SSE
        using reg_usage = std::tuple< std::size_t, std::size_t >;
        // TODO: This is pretty convoluted.
        using arg_class = std::tuple< half_class_result, reg_usage >;

        half_class_result arg_lo(type t, classification_t c) {
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
                    return { arg_info::make< indirect >(TypeConfig::as_pointer(t)) };
                }

                case Class::SSEUp:
                case Class::X87Up: {
                    // This should actually not happen.
                    VAST_UNREACHABLE("This shouldn't happen");
                }

                case Class::Integer: {
                    ++needed_int;
                    auto target_type =
                        TypeConfig::int_type_at_offset(t, 0, t, 0, mk_classifier_ctx());
                    // TODO(abi): Or enum.
                    if (hi == Class::NoClass && TypeConfig::is_scalar_integer(target_type)) {
                        // TODO(abi): If enum, treat as underlying type.
                        if (TypeConfig::can_be_promoted(target_type)) {
                            return { arg_info::make< extend >(target_type) };
                        }
                    }

                    return { target_type };
                }

                case Class::SSE: {
                    ++needed_sse;
                    auto target_type =
                        TypeConfig::sse_target_type_at_offset(t, 0, t, 0, mk_classifier_ctx());
                    return { target_type };
                }
            }
        }

        half_class_result arg_hi(type t, classification_t c) {
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
                    auto target_type =
                        TypeConfig::int_type_at_offset(t, 8, t, 8, mk_classifier_ctx());
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
                    auto target_type =
                        TypeConfig::sse_target_type_at_offset(t, 8, t, 8, mk_classifier_ctx());
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

        arg_info classify_arg(type t) {
            auto c    = classify(t);
            auto low  = arg_lo(t, c);
            auto high = arg_hi(t, c);
            return resolve_classification(t, std::move(low), std::move(high));
        }

        self_t &compute_abi() {
            info.add_return(classify_return(TypeConfig::prepare(info.return_type())));
            for (auto arg : info.fn_type().getInputs()) {
                info.add_arg(classify_arg(TypeConfig::prepare(arg)));
            }
            return *this;
        }

        func_info take() { return std::move(info); }
    };

} // namespace vast::abi

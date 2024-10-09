// Copyright (c) 2022-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <llvm/Support/Alignment.h>
VAST_UNRELAX_WARNINGS

#include "vast/ABI/ABI.hpp"

#include "vast/Dialect/HighLevel/HighLevelUtils.hpp"

namespace vast::abi {

    struct mlir_type_info {
        using data_layout_t = mlir::DataLayout;

      protected:
        mcontext_t &mctx;
        const data_layout_t &dl;

      public:
        explicit mlir_type_info(mcontext_t &mctx, const data_layout_t &dl)
            : mctx(mctx), dl(dl)
        {}

        static bool is_void(mlir_type t);
        static bool is_complex(mlir_type t);

        static bool is_scalar_integer(mlir_type t);
        static bool is_scalar_float(mlir_type t);

        static bool is_scalar(mlir_type t);
        static bool is_aggregate(mlir_type t);
        static bool is_record(mlir_type t);
        static bool is_struct(mlir_type t);
        static bool is_array(mlir_type t);

        static bool can_be_passed_in_regs(mlir_type t);
        static bool has_unaligned_field(mlir_type t);

        static bool represents_pointer(mlir_type t);
        static bool is_pointer(mlir_type t);

        static bool can_be_promoted(mlir_type t);
        static mlir_type prepare(mlir_type t);

        // TODO: Move into different abstraction, pointer size is dependent on underlying
        //       cpu architecture not the IR.
        static std::size_t pointer_size();

        std::size_t size(mlir_type t);
        std::size_t align(mlir_type t);

        mlir_type iN(std::size_t s);
        mlir_type fN(std::size_t s);

        static mlir_type as_pointer(mlir_type t);

        mlir_type type_at_offset(mlir_type t, std::size_t offset, const auto &accept, operation from);

        auto field_containing_offset(mlir_type t, std::size_t offset, operation from)
            -> std::optional< std::tuple< mlir_type, std::size_t > >;

        gap::generator< mlir_type > mock_array_fields(hl::ArrayType array_type);

        gap::generator< mlir_type > fields(mlir_type type, operation from);

        using array_info_t = std::tuple< std::optional< std::size_t >, mlir_type >;
        array_info_t array_info(mlir_type t);
    };

    bool mlir_type_info::is_void(mlir_type t) {
        return mlir::isa< mlir::NoneType >(t);
    }

    bool mlir_type_info::is_scalar(mlir_type t) {
        // TODO(abi): Also complex is an option in ABI.
        return !is_aggregate(t);
    }

    bool mlir_type_info::is_complex(mlir_type t) {
        VAST_UNREACHABLE("Missing support for complex types!");
    }

    // TODO: Once Issue #621 is implemented, update this.
    bool mlir_type_info::is_aggregate(mlir_type t) {
        return mlir::isa< hl::RecordType, hl::ElaboratedType, hl::ArrayType >(t);
    }

    bool mlir_type_info::is_record(mlir_type t) {
        return mlir::isa< hl::RecordType >(hl::strip_elaborated(t));
    }

    bool mlir_type_info::is_struct(mlir_type t) {
        // TODO: For now these seem equivalent, but eventually once we introduce classes
        //       and other C++ features they may diverge.
        return is_record(t);
    }

    bool mlir_type_info::is_array(mlir_type t) {
        return mlir::isa< hl::ArrayType >(hl::strip_elaborated(t));
    }

    // Must be invoked with `t` that is array-like.
    auto mlir_type_info::array_info(mlir_type t) -> array_info_t {
        auto array_type = mlir::dyn_cast< hl::ArrayType >(t);
        VAST_ASSERT(array_type);
        return { array_type.getSize(), array_type.getElementType() };
    }

    bool mlir_type_info::can_be_passed_in_regs(mlir_type t) {
        // TODO(abi): Seems like in C nothing can prevent this.
        return true;
    }

    bool mlir_type_info::is_scalar_integer(mlir_type t) {
        return mlir::isa< mlir::IntegerType >(t);
    }

    // TODO(abi): Implement.
    bool mlir_type_info::has_unaligned_field(mlir_type t) {
        return false;
    }

    bool mlir_type_info::is_scalar_float(mlir_type t) {
        return mlir::isa< mlir::FloatType >(t);
    }

    bool mlir_type_info::is_pointer(mlir_type t) {
        return mlir::isa< hl::PointerType >(t);
    }

    // Pointers, references etc
    bool mlir_type_info::represents_pointer(mlir_type t) {
        return is_pointer(t);
    }

    mlir_type mlir_type_info::as_pointer(mlir_type t) {
        return hl::PointerType::get(t.getContext(), t);
    }

    bool mlir_type_info::can_be_promoted(mlir_type t) {
        return false;
    }

    mlir_type mlir_type_info::prepare(mlir_type t) {
        return hl::strip_value_category(t);
    }

    // TODO(abi): Will need to live in a different interface.
    std::size_t mlir_type_info::pointer_size() {
        return 64;
    }

    std::size_t mlir_type_info::size(mlir_type t) {
        return dl.getTypeSizeInBits(t);
    }

    std::size_t mlir_type_info::align(mlir_type t) {
        return dl.getTypeABIAlignment(t);
    }

    mlir_type mlir_type_info::iN(std::size_t s) {
        return mlir::IntegerType::get(&mctx, s);
    }

    mlir_type mlir_type_info::fN(std::size_t s) {
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

    gap::generator< mlir_type > mlir_type_info::fields(mlir_type type, operation from) {
        if (auto array_type = mlir::dyn_cast< hl::ArrayType >(type))
            return mock_array_fields(array_type);
        if (auto record_type = mlir::dyn_cast< hl::RecordType >(type))
            return field_types(record_type, from);
        VAST_UNREACHABLE("Unsupported type: {0}", type);
    }

    mlir_type mlir_type_info::type_at_offset(mlir_type t, std::size_t offset, const auto &accept, operation from) {
        if (offset == 0 && accept(t)) {
            return t;
        }

        if (is_struct(t)) {
            auto field_info = field_containing_offset(t, offset, from);
            if (!field_info)
                return {};
            auto [field, field_start] = *field_info;
            return type_at_offset(field, field_start, accept, from);
        }

        if (is_array(t)) {
            auto [_, element_type] = array_info(t);
            const auto element_size = size(element_type);
            const auto element_offset = offset / element_size * element_size;
            return type_at_offset(element_type, offset - element_offset, accept, from);
        }
        return {};
    }

    auto mlir_type_info::field_containing_offset(mlir_type t, std::size_t offset, operation from)
        -> std::optional< std::tuple< mlir_type, std::size_t > >
    {
        auto curr = 0;
        for (auto field : fields(t, from)) {
            if (curr + size(field) > offset) {
                return { std::make_tuple(field, curr) };
            }
            curr += size(field);
        }
        return {};
    }

    gap::generator< mlir_type > mlir_type_info::mock_array_fields(hl::ArrayType array_type) {
        auto size = array_type.getSize();
        VAST_CHECK(size, "Variable size array type not yet supported");

        auto et =array_type.getElementType();
        for (std::size_t i = 0; i < *size; ++i)
           co_yield et;
    }


} // namespace vast::abi

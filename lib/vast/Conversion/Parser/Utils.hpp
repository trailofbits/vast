#pragma once

#include "vast/Util/Common.hpp"

#include "vast/Dialect/Parser/Types.hpp"

namespace vast::pr {

    enum class data_type { data, nodata, maybedata };

    static inline mlir_type to_mlir_type(data_type type, mcontext_t *mctx) {
        switch (type) {
            case data_type::data: return pr::DataType::get(mctx);
            case data_type::nodata: return pr::NoDataType::get(mctx);
            case data_type::maybedata: return pr::MaybeDataType::get(mctx);
        }
    }

    template< typename... Ts >
    auto is_one_of(mlir_type ty) { return (mlir::isa< Ts >(ty) || ...); }

    static inline bool is_parser_type(mlir_type ty) {
        return is_one_of< pr::DataType, pr::NoDataType, pr::MaybeDataType >(ty);
    }

    static inline mlir_type join(mlir_type lhs, mlir_type rhs) {
        VAST_ASSERT(!lhs || is_parser_type(lhs));
        VAST_ASSERT(!rhs || is_parser_type(rhs));

        if (!lhs)
            return rhs;
        return lhs == rhs ? lhs : pr::MaybeDataType::get(lhs.getContext());
    }

    static inline mlir_type meet(mlir_type lhs, mlir_type rhs) {
        VAST_ASSERT(!lhs || is_parser_type(lhs));
        VAST_ASSERT(!rhs || is_parser_type(rhs));

        if (lhs == rhs)
            return lhs;
        if (mlir::isa< pr::MaybeDataType >(lhs))
            return rhs;
        if (mlir::isa< pr::MaybeDataType >(rhs))
            return lhs;
        return mlir_type{};
    }

    static inline mlir_type top_type(value_range values) {
        mlir_type ty;
        for (auto val : values) {
            ty = join(ty, val.getType());
        }
        return ty;
    }

    static inline mlir_type bottom_type(value_range values) {
        mlir_type ty;
        for (auto val : values) {
            ty = meet(ty, val.getType());
        }
        return ty;
    }

} // namespace vast::pr

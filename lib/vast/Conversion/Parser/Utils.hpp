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

} // namespace vast::pr

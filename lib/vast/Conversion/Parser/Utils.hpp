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

    static bool is_nodata(mlir_type type) { return mlir::isa< pr::NoDataType >(type); }

    static bool is_nodata(mlir_value value) { return is_nodata(value.getType()); }

    static bool is_nodata(mlir::ValueRange values) {
        for (auto value : values) {
            if (!is_nodata(value)) {
                return false;
            }
        }
        return true;
    }

    static bool is_data(mlir_type type) { return mlir::isa< pr::DataType >(type); }

    static bool is_data(mlir_value value) { return is_data(value.getType()); }

    static bool is_maybedata(mlir_type type) { return mlir::isa< pr::MaybeDataType >(type); }

    static bool is_maybedata(mlir_value value) { return is_maybedata(value.getType()); }

    static bool is_noparse_region(mlir::Region *region);

    static bool is_noparse_op(mlir::Operation &op) {
        if (mlir::isa< pr::NoParse >(op)) {
            return true;
        }

        if (mlir::isa< hl::NullStmt >(op)) {
            return true;
        }

        if (mlir::isa< hl::BreakOp >(op)) {
            return true;
        }

        if (mlir::isa< hl::ContinueOp >(op)) {
            return true;
        }

        if (auto yield = mlir::dyn_cast< hl::CondYieldOp >(op)) {
            if (is_nodata(yield.getResult())) {
                return true;
            }
        }

        if (auto yield = mlir::dyn_cast< hl::ValueYieldOp >(op)) {
            if (is_nodata(yield.getResult())) {
                return true;
            }
        }

        if (auto ret = mlir::dyn_cast< hl::ReturnOp >(op)) {
            if (is_nodata(ret.getResult())) {
                return true;
            }
        }

        if (auto call = mlir::dyn_cast< hl::CallOp >(op)) {
            return is_nodata(call.getArgOperands()) && is_nodata(call.getResults());
        }

        if (auto d = mlir::dyn_cast< hl::DefaultOp >(op)) {
            return is_noparse_region(&d.getBody());
        }

        if (auto c = mlir::dyn_cast< hl::CaseOp >(op)) {
            return is_noparse_region(&c.getBody()) && is_noparse_region(&c.getLhs());
        }

        return false;
    }

    static bool is_noparse_region(mlir::Region *region) {
        if (region->empty()) {
            return true;
        }

        for (auto &block : *region) {
            for (auto &op : block) {
                if (!is_noparse_op(op)) {
                    return false;
                }
            }
        }

        return true;
    }

} // namespace vast::pr

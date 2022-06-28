// Copyright (c) 2022-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <mlir/IR/SymbolTable.h>
#include <llvm/Support/raw_ostream.h>
VAST_UNRELAX_WARNINGS

#include "vast/Interfaces/SymbolInterface.hpp"

namespace vast::util
{
    using vast_symbol_interface = vast::VastSymbolOpInterface;
    using mlir_symbol_interface = mlir::SymbolOpInterface;

    using string_ref     = llvm::StringRef;

    // TODO(heno): rework to coroutines eventually
    void symbols(mlir::Operation *op, auto yield) {
        op->walk([&] (mlir::Operation *child) {
            if (auto symbol = mlir::dyn_cast< vast_symbol_interface >(child)) {
                yield(symbol);
            }
            else if (auto symbol = mlir::dyn_cast< mlir_symbol_interface >(child)) {
                yield(symbol);
            }
        });
    }

    static inline auto symbol_name(vast_symbol_interface value) { return value.getSymbolName(); }

    static inline auto symbol_name(mlir_symbol_interface value) { return value.getName(); }

    void yield_symbol_users(vast_symbol_interface op, auto scope, auto yield) {
        for (auto user : op->getUsers()) {
            yield(user);
        }
    };

    void yield_symbol_users(mlir_symbol_interface op, auto scope, auto yield) {
        if (auto users = op.getSymbolUses(scope)) {
            for (auto use : users.getValue()) {
                yield(use.getUser());
            }
        }
    };

    void yield_users(string_ref symbol, auto scope, auto yield) {
        auto filter_symbols = [&](auto op) {
            if (util::symbol_name(op) == symbol) {
                yield_symbol_users(op, scope, yield);
            }
        };

        util::symbols(scope, filter_symbols);
    }

    std::string show_location(auto &value) {
        auto loc = value.getLoc();
        std::string buff;
        llvm::raw_string_ostream ss(buff);
        if (auto file_loc = loc.template dyn_cast< mlir::FileLineColLoc >()) {
            ss << " : " << file_loc.getFilename().getValue() << ":" << file_loc.getLine()
                         << ":" << file_loc.getColumn();
        } else {
            ss << " : " << loc;
        }

        return ss.str();
    }

    std::string show_symbol_value(auto &value) {
        std::string buff;
        llvm::raw_string_ostream ss(buff);
        ss << value->getName() << " : " << symbol_name(value) << " " << show_location(value);
        return ss.str();
    }

} // namespace vast::util

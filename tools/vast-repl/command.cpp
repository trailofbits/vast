// Copyright (c) 2022-present, Trail of Bits, Inc.

#include "vast/repl/command.hpp"
#include "vast/repl/common.hpp"

namespace vast::repl::cmd {

    void check_source(const state_t &state) {
        if (!state.source.has_value()) {
            VAST_UNREACHABLE("error: missing source");
        }
    }

    const std::string &get_source(const state_t &state) {
        check_source(state);
        return state.source.value();
    }

    void check_and_emit_module(state_t &state) {
        if (!state.mod) {
            const auto &source = get_source(state);
            state.mod = codegen::emit_module(source, &state.ctx);
        }
    }

    //
    // exit command
    //
    void exit::run(state_t &state) const {
        state.exit = true;
    }

    //
    // help command
    //
    void help::run(state_t&) const {
        VAST_UNIMPLEMENTED;
    };

    //
    // load command
    //
    void load::run(state_t &state) const {
        auto source  = get_param< source_param >(params);
        state.source = codegen::get_source(source.path);
    };

    //
    // show command
    //
    void show_source(const state_t &state) {
        llvm::outs() << get_source(state) << "\n";
    }

    void show_ast(const state_t &state) {
        auto unit = codegen::ast_from_source(get_source(state));
        unit->getASTContext().getTranslationUnitDecl()->dump(llvm::outs());
        llvm::outs() << "\n";
    }

    void show_module(state_t &state) {
        check_and_emit_module(state);
        llvm::outs() << state.mod.get() << "\n";
    }

    void show_symbols(state_t &state) {
        check_and_emit_module(state);

        util::symbols(state.mod.get(), [&] (auto symbol) {
            llvm::outs() << util::show_symbol_value(symbol) << "\n";
        });
    }

    void show::run(state_t &state) const {
        auto what = get_param< kind_param >(params);
        switch (what) {
            case show_kind::source:  return show_source(state);
            case show_kind::ast:     return show_ast(state);
            case show_kind::module:  return show_module(state);
            case show_kind::symbols: return show_symbols(state);
        }
    };

    //
    // meta command
    //
    void meta::add(state_t &state) const {
        using ::vast::meta::add_identifier;

        auto name_param = get_param< symbol_param >(params);
        util::symbols(state.mod.get(), [&] (auto symbol) {
            if (util::symbol_name(symbol) == name_param.value) {
                auto id = get_param< identifier_param >(params);
                add_identifier(symbol, id.value);
                llvm::outs() << symbol << "\n";
            }
        });
    }

    void meta::get(state_t &state) const {
        using ::vast::meta::get_with_identifier;
        auto id = get_param< identifier_param >(params);
        for (auto op : get_with_identifier(state.mod.get(), id.value)) {
            llvm::outs() << *op << "\n";
        }
    }

    void meta::run(state_t &state) const {
        check_and_emit_module(state);

        auto action  = get_param< action_param >(params);
        switch (action) {
            case meta_action::add: add(state); break;
            case meta_action::get: get(state); break;
        }
    };

} // namespace vast::repl::cmd

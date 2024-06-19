// Copyright (c) 2022-present, Trail of Bits, Inc.

#include "vast/repl/command.hpp"

#include "vast/Conversion/Passes.hpp"
#include "vast/Tower/Tower.hpp"
#include "vast/repl/common.hpp"
#include <optional>

namespace vast::repl {
namespace cmd {

    void check_source(const state_t &state) {
        if (!state.source.has_value()) {
            throw_error("error: missing source");
        }
    }

    maybe_memory_buffer get_source_buffer(const state_t &state) {
        check_source(state);

        // Open the file using MemoryBuffer
        maybe_memory_buffer file_buffer = llvm::MemoryBuffer::getFile(state.source->c_str());

        // Check if the file is opened successfully
        if (auto errorCode = file_buffer.getError()) {
            throw_error("error: missing source {}", errorCode.message());
        }

        return file_buffer;
    }

    void check_and_emit_module(state_t &state) {
        if (!state.tower) {
            check_source(state);
            auto mod    = codegen::emit_module(state.source.value(), state.ctx);
            auto [t, _] = tw::default_tower::get(state.ctx, std::move(mod));
            state.tower = std::move(t);
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
        throw_error("Not implemented!");
    };

    //
    // load command
    //
    void load::run(state_t &state) const {
        state.source = get_param< source_param >(params).path;
    };

    //
    // show command
    //
    void show_source(const state_t &state) {
        auto buff = get_source_buffer(state);
        llvm::outs() << buff.get()->getBuffer() << "\n";
    }

    void show_ast(const state_t &state) {
        auto buff = get_source_buffer(state);
        auto unit = codegen::ast_from_source(buff.get()->getBuffer());
        unit->getASTContext().getTranslationUnitDecl()->dump(llvm::outs());
        llvm::outs() << "\n";
    }

    void show_module(state_t &state) {
        check_and_emit_module(state);
        llvm::outs() << state.tower->top().mod << "\n";
    }

    void show_symbols(state_t &state) {
        check_and_emit_module(state);

        util::symbols(state.tower->top().mod, [&] (auto symbol) {
            llvm::outs() << util::show_symbol_value(symbol) << "\n";
        });
    }

    void show_pipelines(state_t &state) {
        if (state.pipelines.empty()) {
            llvm::outs() << "no pipelines\n";
        }

        for (const auto &[name, _] : state.pipelines) {
            llvm::outs() << name << "\n";
        }
    }

    void show::run(state_t &state) const {
        auto what = get_param< kind_param >(params);
        switch (what) {
            case show_kind::source:  return show_source(state);
            case show_kind::ast:     return show_ast(state);
            case show_kind::module:  return show_module(state);
            case show_kind::symbols: return show_symbols(state);
            case show_kind::pipelines: return show_pipelines(state);
        }
    };

    //
    // meta command
    //
    void meta::add(state_t &state) const {
        using ::vast::meta::add_identifier;

        auto name_param = get_param< symbol_param >(params);
        util::symbols(state.tower->top().mod, [&] (auto symbol) {
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
        for (auto op : get_with_identifier(state.tower->top().mod, id.value)) {
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
    }

    //
    // raise command
    //
    void raise::run(state_t &state) const {
        check_and_emit_module(state);

        std::string pipeline = get_param< pipeline_param >(params).value;
        llvm::SmallVector< llvm::StringRef, 2 > passes;
        llvm::StringRef(pipeline).split(passes, ',');

        mlir::PassManager pm(&state.ctx);
        auto th = state.tower->top();
        for (auto pass : passes) {
            if (mlir::failed(mlir::parsePassPipeline(pass, pm))) {
                throw_error("failed to parse pass pipeline");
            }
            th = state.tower->apply(th, pm);
        }
    }

    //
    // sticky command
    //
    void sticky::run(state_t &state) const {
        auto cmd = get_param< command_param >(params);
        add_sticky_command(cmd.value, state);
    }

    void add_sticky_command(string_ref cmd, state_t &state) {
        auto tokens = parse_tokens(cmd);
        state.sticked.push_back(parse_command(tokens));
    }

} // namespace cmd

    command_ptr parse_command(std::span< command_token > tokens) {
        return match< cmd::command_list >(tokens);
    }

} // namespace vast::repl

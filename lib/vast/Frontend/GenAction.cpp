// Copyright (c) 2023-present, Trail of Bits, Inc.

#include "vast/Frontend/GenAction.hpp"

#include "vast/Frontend/Common.hpp"

namespace vast::cc {

    vast_gen_action::vast_gen_action(output_type act, MContext *montext)
        : mcontext(montext ? montext : new MContext), action(act)
    {}

    OwningModuleRef vast_gen_action::load_module(llvm::MemoryBufferRef /* mref */) {
        throw compiler_error("load_module not implemented");
    }

    void vast_gen_action::ExecuteAction() {
        throw compiler_error("ExecuteAction not implemented");
    }

    auto vast_gen_action::CreateASTConsumer(compiler_instance &/* ci */, llvm::StringRef /* input */)
        -> std::unique_ptr< clang::ASTConsumer >
    {
        throw compiler_error("CreateASTConsumer not implemented");
    }

    void vast_gen_action::EndSourceFileAction() {
        throw compiler_error("EndSourceFileAction not implemented");
    }

    void emit_assembly_action::anchor() {}

    emit_assembly_action::emit_assembly_action(MContext *mcontex)
        : vast_gen_action(output_type::emit_assembly, mcontex)
    {}

    void emit_llvm_action::anchor() {}

    emit_llvm_action::emit_llvm_action(MContext *mcontex)
        : vast_gen_action(output_type::emit_llvm, mcontex)
    {}

    void emit_obj_action::anchor() {}

    emit_obj_action::emit_obj_action(MContext *mcontex)
        : vast_gen_action(output_type::emit_obj, mcontex)
    {}

} // namespace vast::cc

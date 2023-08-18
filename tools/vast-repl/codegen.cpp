// Copyright (c) 2022-present, Trail of Bits, Inc.

#include "vast/repl/codegen.hpp"
#include "vast/repl/common.hpp"

#include "vast/repl/state.hpp"

#include "vast/CodeGen/CodeGen.hpp"

#include <fstream>

namespace vast::repl::codegen {

    std::string slurp(std::ifstream& in) {
        std::ostringstream sstr;
        sstr << in.rdbuf();
        return sstr.str();
    }

    std::unique_ptr< clang::ASTUnit > ast_from_source(const std::string &source) {
        return clang::tooling::buildASTFromCode(source);
    }

    std::string get_source(std::filesystem::path source) {
        std::ifstream in(source);
        return slurp(in);
    }

    owning_module_ref emit_module(const std::string &source, mcontext_t *mctx) {
        auto unit = codegen::ast_from_source(source);
        auto &actx = unit->getASTContext();
        vast::cg::CodeGenContext cgctx(*mctx, actx);
        vast::cg::DefaultCodeGen codegen(cgctx);
        codegen.emit_module(actx.getTranslationUnitDecl());
        return std::move(cgctx.mod);
    }

} // namespace vast::repl::codegen

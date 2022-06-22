// Copyright (c) 2022-present, Trail of Bits, Inc.

#include "vast/repl/codegen.hpp"

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

} // namespace vast::repl::codegen

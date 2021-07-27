// Copyright (c) 2021-present, Trail of Bits, Inc.

#include <cstdio>
#include <fstream>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Translation.h>
#include <mlir/Support/LogicalResult.h>

#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/MemoryBuffer.h>

#include "cppast/libclang_parser.hpp"

#include <iostream>
#include <filesystem>
#include <memory>

namespace vast
{
    std::unique_ptr<cppast::cpp_file> parse(const std::string path)
    {
        cppast::cpp_entity_index idx;

        cppast::stderr_diagnostic_logger logger;
        cppast::libclang_parser parser(type_safe::ref(logger));

        cppast::libclang_compile_config cfg;
        auto file = parser.parse(idx, path, cfg);
        return parser.error() ? nullptr : std::move(file);
    }

    static mlir::OwningModuleRef from_source_parser(const llvm::MemoryBuffer *input, mlir::MLIRContext *ctx)
    {

        auto path = std::filesystem::temp_directory_path();
        path += "vast-input.tmp";

        {
            std::ofstream filestream(path);
            filestream << input->getBuffer().str();
        }

        auto ast = parse(path);
        std::cout << ast->name() << std::endl;
        // TODO AST to MLIR

        return {};
    }

    mlir::LogicalResult registerFromSourceParser()
    {
        mlir::TranslateToMLIRRegistration from_source( "from-source",
            [] (llvm::SourceMgr &mgr, mlir::MLIRContext *ctx) -> mlir::OwningModuleRef {
                assert(mgr.getNumBuffers() == 1 && "expected single input buffer");
                auto buffer = mgr.getMemoryBuffer(mgr.getMainFileID());
                return from_source_parser(buffer, ctx);
            }
        );

        return mlir::success();
    }
} // namespace vast
// Copyright (c) 2022-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <clang/Basic/DiagnosticOptions.h>
#include <clang/Frontend/TextDiagnosticBuffer.h>
#include <clang/Frontend/TextDiagnosticPrinter.h>
#include <clang/Tooling/Tooling.h>
#include <llvm/ADT/IntrusiveRefCntPtr.h>
VAST_UNRELAX_WARNINGS

#include "vast/Frontend/Common.hpp"

namespace vast::cc
{
    using diagnostics_engine = clang::DiagnosticsEngine;

    //
    // diagnostics options
    //
    struct diagnostics_options : llvm_cnt_ptr< clang::DiagnosticOptions > {
        using base = llvm_cnt_ptr< clang::DiagnosticOptions >;

        diagnostics_options(argv_t argv)
            : base(clang::CreateAndPopulateDiagOpts(argv)) {}
    };

    //
    // diagnostics buffer
    //
    struct diagnostics_buffer {
        using base = clang::TextDiagnosticBuffer;

        explicit diagnostics_buffer() : buffer(new base()) {}

        base *get() { return buffer.get(); }

        void flush(diagnostics_engine &engine) {
            buffer->FlushDiagnostics(engine);
        }

        std::unique_ptr< base > buffer;
    };

    //
    // diagnostics
    //
    struct diagnostics {
        using ids = llvm_cnt_ptr< clang::DiagnosticIDs >;

        static ids make_ids() { return new clang::DiagnosticIDs(); }

        diagnostics(llvm::ArrayRef<const char *> argv)
            : opts(argv), buffer(), engine(make_ids(), opts, buffer.get(), false)
        {
            // FIXME: deal with DiagnosticSerializationFile

            clang::ProcessWarningOptions(engine, *opts, /* ReportDiags */false);
        }

        void flush() { buffer.flush(engine); }

        diagnostics_options opts;
        diagnostics_buffer buffer;
        diagnostics_engine engine;
    };

} // namespace vast::cc

// Copyright (c) 2023-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <mlir/Pass/PassInstrumentation.h>
#include <mlir/Pass/Pass.h>
VAST_UNRELAX_WARNINGS

#include "vast/Util/Common.hpp"

namespace vast::util {

    struct with_snapshots : mlir::PassInstrumentation
    {
        using output_stream_ptr = std::shared_ptr< llvm::raw_pwrite_stream >;

        explicit with_snapshots(string_ref file_prefix) : file_prefix(file_prefix) {}

        // We return `shared_ptr` in case we may want to keep the stream open
        // for longer in some derived class. Should not make a difference,
        // snapshoting will be expensive anyway.
        virtual output_stream_ptr make_output_stream(mlir::Pass *pass);

        virtual bool should_snapshot(mlir::Pass *pass) const = 0;

        void runAfterPass(mlir::Pass *pass, operation op) override;

        std::string file_prefix;
    };


    struct snapshot_after_passes : with_snapshots {
        using base     = with_snapshots;
        using passes_t = std::vector< string_ref >;

        template< typename ... args_t >
        explicit snapshot_after_passes(const passes_t &snapshot_at, args_t && ... args)
            : base(std::forward< args_t >(args)...), snapshot_at(snapshot_at)
        {}

        bool should_snapshot(mlir::Pass *pass) const override {
            return std::ranges::count(snapshot_at, pass->getArgument());
        }

        passes_t snapshot_at;
    };


    struct snapshot_all : with_snapshots {
        using base = with_snapshots;
        using base::base;

        bool should_snapshot(mlir::Pass *) const override { return true; }
    };

} // namespace vast::util

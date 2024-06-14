// Copyright (c) 2022-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <clang/AST/GlobalDecl.h>
#include <clang/Basic/TargetInfo.h>
#include <mlir/InitAllDialects.h>
VAST_UNRELAX_WARNINGS

#include "vast/Util/Common.hpp"
#include "vast/Util/Triple.hpp"
#include "vast/Util/DataLayout.hpp"

#include "vast/CodeGen/CodeGenVisitorBase.hpp"
#include "vast/CodeGen/CodeGenBuilder.hpp"
#include "vast/CodeGen/ScopeContext.hpp"
#include "vast/CodeGen/CodeGenMetaGenerator.hpp"

#include "vast/CodeGen/CodeGenFunction.hpp"

#include "vast/Dialect/Dialects.hpp"
#include "vast/Dialect/Core/CoreAttributes.hpp"

#include "vast/CodeGen/DefaultSymbolGenerator.hpp"

namespace vast::cg {

    struct module_generator : generator_base
    {
        module_generator(codegen_builder &bld, scoped_visitor_view visitor)
            : generator_base(bld, visitor)
        {}

        void emit(clang::DeclGroupRef decls);
        void emit(clang::Decl *decl);

        void finalize();
        void emit_data_layout();
    };

} // namespace vast::cg

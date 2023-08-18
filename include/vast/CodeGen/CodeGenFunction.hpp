// Copyright (c) 2022-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Common.hpp"
#include "vast/Dialect/HighLevel/HighLevelOps.hpp"

namespace vast::cg {

    struct global_emition {
        enum class kind { definition, declaration };

        bool for_vtable = false;
        bool defer = false;
        bool thunk = false;

        kind emition_kind;
    };

    static constexpr global_emition emit_definition = {
        .emition_kind = global_emition::kind::definition
    };

    static constexpr global_emition deferred_emit_definition = {
        .defer        = true,
        .emition_kind = global_emition::kind::definition
    };

    static constexpr global_emition emit_declaration = {
        .emition_kind = global_emition::kind::declaration
    };

    constexpr bool is_for_definition(global_emition emit) {
        return emit.emition_kind == global_emition::kind::definition;
    }

    constexpr bool is_for_declaration(global_emition emit) {
        return emit.emition_kind == global_emition::kind::declaration;
    }

    using vast_function = vast::hl::FuncOp;

} // namespace vast::cg

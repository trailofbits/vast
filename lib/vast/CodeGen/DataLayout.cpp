// Copyright (c) 2021-present, Trail of Bits, Inc.

#include "vast/CodeGen/DataLayout.hpp"

namespace vast::hl
{
    void emit_data_layout(mcontext_t &ctx, owning_module_ref &mod, const dl::DataLayoutBlueprint &dl) {
        std::vector< mlir::DataLayoutEntryInterface > entries;
        for (const auto &[_, e] : dl.entries) {
            entries.push_back(e.wrap(ctx));
        }

        mod.get()->setAttr(
            mlir::DLTIDialect::kDataLayoutAttrName, mlir::DataLayoutSpecAttr::get(&ctx, entries)
        );
    }

} // namespace vast::hl

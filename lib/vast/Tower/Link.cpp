// Copyright (c) 2024-present, Trail of Bits, Inc.

#include "vast/Tower/Link.hpp"

#include <ranges>

namespace vast::tw {

    op_mapping reverse_mapping(const op_mapping &from) {
        op_mapping out;
        for (const auto &[root, ops] : from) {
            for (auto op : ops) {
                out[op].push_back(root);
            }
        }
        return out;
    }

    void dbg(const op_mapping &mapping, auto &outs) {
        outs << "Mapping:\n";
        for (const auto &[from, to] : mapping) {
            outs << ".." << *from << "\n";
            for (auto t : to) {
                outs << "   -> " << *t << "\n";
            }
        }
    }

    using loc_to_op_t = llvm::DenseMap< loc_t, operations >;

    void dbg(const loc_to_op_t &mapping, auto &outs) {
        outs << "loc_to_op\n";
        for (const auto &[loc, to] : mapping) {
            outs << ".." << loc << "\n";
            for (auto t : to) {
                outs << "   -> " << *t << "\n";
            }
        }
    }

    loc_to_op_t gather_map(location_info_t &li, operation op) {
        loc_to_op_t out;
        auto collect = [&](operation op) { out[li.self(op)].push_back(op); };
        op->walk(collect);
        return out;
    }

    struct continuos_mapping_builder
    {
        std::vector< loc_to_op_t > details;
        const unit_link_vector &raw_link;
        location_info_t &li;

        continuos_mapping_builder(const unit_link_vector &raw_link, location_info_t &li)
            : raw_link(raw_link), li(li) {
            for (const auto &l : raw_link) {
                details.push_back(gather_map(li, l->from().mod));
            }
            details.push_back(gather_map(li, raw_link.back()->to().mod));
        }

        op_mapping compute_parent_mapping() {
            op_mapping out;

            auto handle_level = [&](const auto &level) {
                for (auto &[op, todo] : out) {
                    operations parents;
                    for (auto current : todo) {
                        if (auto prev_ops = level.find(li.prev(current)); prev_ops != level.end())
                            parents.append_range(prev_ops->second);
                    }
                    todo = std::move(parents);
                }
            };

            std::reverse(details.begin(), details.end());
            // initialize
            for (auto [_, op] : details.front()) {
                out[op.front()] = { op };
            }

            for (const auto &level : details | std::views::drop(1))
                handle_level(level);

            return out;
        }
    };

    operations fat_link::children(operation op) { return down[op]; }

    operations fat_link::children(operations ops) {
        operations out;
        for (auto op : ops)
            out.append_range(children(op));
        return out;
    }

    operations fat_link::shared_children(operations) { VAST_UNIMPLEMENTED_MSG("nyi!"); }

    operations fat_link::parents(operation op) { return up[op]; }

    operations fat_link::parents(operations ops) {
        operations out;
        for (auto op : ops)
            out.append_range(parents(op));
        return out;
    }

    operations fat_link::shared_parents(operations) { VAST_UNIMPLEMENTED_MSG("nyi!"); }

    handle_t fat_link::from() const { return steps.front()->from(); }
    handle_t fat_link::to() const { return steps.back()->to(); }

    fat_link::fat_link(unit_link_vector steps_) : steps(std::move(steps_)) {
        VAST_CHECK(steps.size() >= 1, "Not enough steps=to build a link!");
        auto &li = steps.front()->li();
        auto bld = continuos_mapping_builder(steps, li);
        up       = bld.compute_parent_mapping();
        down     = reverse_mapping(up);
    }

} // namespace vast::tw

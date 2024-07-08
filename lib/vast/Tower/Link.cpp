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

        auto flag = mlir::OpPrintingFlags().skipRegions();
        auto render_op = [&](operation op) -> decltype(outs) & {
            outs << op << ": ";
            op->print(outs, flag);
            if (!op->getRegions().empty())
                outs << " ... regions ...";
            return outs;
        };

        for (const auto &[from, to] : mapping) {
            outs << "== ";
            render_op(from) << "\n";
            for (auto t : to) {
                outs << "   -> ";
                render_op(t) << "\n";
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

    std::vector< op_mapping > mk_link_mappings(const auto &links) {
        std::vector< op_mapping > out;
        for (const auto &l : links)
            out.emplace_back(l->parents_to_children());
        return out;
    }

    // `transition` handles the lookup between levels - we need this as we want to generalise
    // (`levels` can be arbitrary mapping, not just `op -> { op }`).
    op_mapping build_map(op_mapping init, const auto &levels, auto transition) {
        auto handle_level = [&](const auto &level) {
            auto handle_element = [&](operation op) -> operations {
                auto prev_ops = level.find(transition(op));
                if (prev_ops == level.end())
                    return {};
                return prev_ops->second;
            };

            for (auto &[op, todo] : init) {
                operations parents;
                for (auto current : todo) {
                    parents.append_range(handle_element(current));
                }
                todo = std::move(parents);
            }
        };

        for (const auto &level : levels | std::views::drop(1))
            handle_level(level);

        return init;
    }

    op_mapping build_map(const link_vector &links) {
        auto transition = [](operation op) { return op; };
        auto init = links.front()->parents_to_children();
        return build_map(init, mk_link_mappings(links), transition);
    }

    op_mapping build_map(std::vector< loc_to_op_t > links, location_info_t &li) {
        auto transition = [&](operation op) { return li.prev(op); };

        std::reverse(links.begin(), links.end());

        // initialize
        op_mapping init;
        for (auto [_, op] : links.front()) {
            init[op.front()] = { op };
        }
        return build_map(init, links, transition);
    }

    op_mapping build_map(handle_t from, handle_t to, location_info_t &li) {
        return build_map({gather_map(li, from.mod), gather_map(li, to.mod)}, li);
    }

    /* conversion_step::link_interface API */

    operations conversion_step::children(operation) { VAST_UNIMPLEMENTED; }
    operations conversion_step::children(operations) { VAST_UNIMPLEMENTED; }

    operations conversion_step::parents(operation) { VAST_UNIMPLEMENTED;  }
    operations conversion_step::parents(operations) { VAST_UNIMPLEMENTED; }

    op_mapping conversion_step::parents_to_children() {
        return reverse_mapping(children_to_parents());
    }

    op_mapping conversion_step::children_to_parents() {
        return build_map(from(), to(), _location_info);
    }

    handle_t conversion_step::from() const { return _from; }
    handle_t conversion_step::to() const { return _to; }

    /* fat_link */

    fat_link::fat_link(link_vector links)
        : links(std::move(links)),
          down(build_map(this->links)),
          up(reverse_mapping(up))
      {}

    /* fat_link::link_interface API */

    operations fat_link::children(operation op) {
        return down[op];
    }

    operations fat_link::children(operations ops) {
        operations out;
        for (auto op : ops)
            out.append_range(children(op));
        return out;
    }

    operations fat_link::parents(operation op) { return up[op]; }

    operations fat_link::parents(operations ops) {
        operations out;
        for (auto op : ops)
            out.append_range(parents(op));
        return out;
    }

    op_mapping fat_link::parents_to_children() { return down; }
    op_mapping fat_link::children_to_parents() { return up; }

    handle_t fat_link::from() const { return links.front()->from(); }
    handle_t fat_link::to() const { return links.back()->to(); }

} // namespace vast::tw

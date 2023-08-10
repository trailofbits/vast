// Copyright (c) 2023-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
VAST_UNRELAX_WARNINGS

namespace vast::conv::abi
{
    /* Handles aggregate type reconstruction. */

    // TODO(conv:abi): Parametrize by materializer, which will resolve what dialect
    //                is the target
    //                 * hl
    //                 * ll-vars
    //                 * llvm
    template< typename pattern, typename op_t >
    struct field_allocator
    {
        using self_t = field_allocator< pattern, op_t >;

        struct state_t
        {
            std::size_t arg_idx = 0;
            std::size_t offset = 0;

            const pattern &parent;
            op_t abi_op;

            state_t(const pattern &parent, op_t abi_op)
                : parent(parent), abi_op(abi_op)
            {}

            state_t( const state_t &) = delete;
            state_t( state_t &&) = delete;

            state_t &operator=(state_t) = delete;

            auto arg() { return abi_op.getOperands()[arg_idx]; };
            auto bw(auto w) { return parent.bw(w); };

            bool fits(mlir_type type)
            {
                return bw(arg()) >= offset + bw(type);
            };

            void advance()
            {
                ++arg_idx;
                offset = 0;
            }

            // TODO(conv:abi): Possibly add some debug prints to help us
            //                 debug some weird corner cases in the future?
            mlir::Value allocate(mlir_type type, auto &rewriter)
            {
                auto start = offset;
                offset += bw(type);
                if (bw(type) == bw(arg()))
                    return arg();

                return rewriter.template create< ll::Extract >(
                        abi_op.getLoc(), type, arg(),
                        start, offset);
            };
        };

        state_t &state;
        vast_module mod;
        std::vector< mlir::Value > partials;

        field_allocator(state_t &state, vast_module mod)
            : state(state),
              mod(mod)
        {}

        static state_t mk_state(const pattern &parent, op_t abi_op)
        {
            return state_t(parent, abi_op);
        }

        bool needs_nesting(mlir_type type) const
        {
            return contains_subtype< hl::RecordType >(type);
        }

        mlir::Value run_on(mlir_type root_type, auto &rewriter)
        {
            auto handle_type = [&](mlir_type field_type) -> mlir::Value
            {
                if (needs_nesting(field_type))
                    return self_t(state, mod).run_on(field_type, rewriter);

                if (!state.fits(field_type))
                    state.advance();
                return state.allocate(field_type, rewriter);
            };

            for (auto field_type : vast::hl::field_types(root_type, mod))
                partials.push_back(handle_type(field_type));

            // Make the thing;
            return make_aggregate(root_type, partials, rewriter);
        }

        mlir::Value make_aggregate(mlir_type type, const auto &partials,
                                   auto &rewriter)
        {
            return rewriter.template create< hl::InitListExpr >(
                    state.abi_op.getLoc(), type, partials).getResult(0);
        }
    };

    /* Aggregate type deconstruction. */

    template< typename pattern, typename op_t >
    struct aggregate_deconstructor
    {
        using self_t = aggregate_deconstructor< pattern, op_t >;

        struct state_t
        {
            std::size_t arg_idx = 0;
            std::size_t offset = 0;

            const pattern &parent;
            op_t abi_op;

            std::vector< mlir::Value > current;

            state_t(const pattern &parent, op_t abi_op)
                : parent(parent), abi_op(abi_op)
            {}

            state_t( const state_t &) = delete;
            state_t( state_t &&) = delete;

            state_t &operator=(state_t) = delete;

            auto dst() { return abi_op.getResults().getTypes()[arg_idx]; };

            auto bw(auto w) { return parent.bw(w); };

            bool fits(mlir_type type)
            {
                return bw(dst()) >= offset + bw(type);
            };

            void advance()
            {
                ++arg_idx;
                offset = 0;
            }

            mlir::Value construct(auto &rewriter)
            {
                // `std::vector` is guaranteed to be empty after move.
                auto out = rewriter.template create< ll::Concat >(
                        abi_op.getLoc(),
                        dst(), std::move(current));
                advance();
                current.clear();
                return out;
            }

            // Returns value only one destination is saturated.
            auto allocate(mlir_type type, auto &rewriter, auto val)
                -> std::optional< mlir::Value >
            {
                auto start = offset;

                if (fits(type))
                {
                    offset += bw(type);
                    current.push_back(val);
                    if (start + bw(type) == bw(dst()))
                        return { construct(rewriter) };
                    return {};
                }

                // We need to do the split
                auto breakpoint = bw(dst()) - start;

                auto mk_int_type = [&](auto size)
                {
                    return mlir::IntegerType::get(
                           abi_op.getContext(),
                           size,
                           mlir::IntegerType::Signless);
                };

                auto prefix = rewriter.template create< ll::Extract >(
                        abi_op.getLoc(), mk_int_type(breakpoint), val,
                        0, breakpoint);

                current.push_back(prefix);
                auto to_yield = construct(rewriter);

                auto suffix_size = (start + bw(type)) - bw(dst());
                auto suffix = rewriter.template create< ll::Extract >(
                        abi_op.getLoc(), mk_int_type(suffix_size), val,
                        breakpoint, bw(type));

                offset = suffix_size;
                current.push_back(suffix);

                return { to_yield };
            }
        };

        state_t &state;
        vast_module mod;
        std::vector< mlir::Value > partials;

        aggregate_deconstructor(state_t &state, vast_module mod)
            : state(state),
              mod(mod)
        {}

        static state_t mk_state(const pattern &parent, op_t abi_op)
        {
            return state_t(parent, abi_op);
        }

        bool needs_nesting(mlir_type type) const
        {
            return contains_subtype< hl::RecordType >(type);
        }

        void run_on(operation root, auto &rewriter)
        {
            auto handle_field = [&](auto gep)
            {
                auto field_type = gep.getType();
                if (needs_nesting(field_type))
                    return self_t(state, mod).run_on(gep.getOperation(), rewriter);

                if (auto val = state.allocate(field_type, rewriter, gep))
                    partials.push_back(*val);
            };

            for (auto field_gep : hl::traverse_record(root, rewriter))
                handle_field(field_gep);
        }

        auto run(operation root, auto &rewriter) &&
        {
            run_on(root, rewriter);
            // Now construct the rest into a value
            auto val = state.construct(rewriter);
            partials.push_back(val);

            return std::move(partials);
        }
    };

    template< typename pattern_t, typename abi_op_t, typename rewriter_t >
    auto deconstruct_aggregate(const pattern_t &pattern, abi_op_t op,
                               mlir::Operation *value, rewriter_t &rewriter)
    {
        using deconstructor_t = aggregate_deconstructor< pattern_t, abi_op_t >;
        auto state = deconstructor_t::mk_state(pattern, op);

        auto module_op = op->template getParentOfType< vast_module >();
        VAST_ASSERT(module_op);
        return deconstructor_t(state, module_op).run(value, rewriter);
    }

    // TODO(conv:abi): This is currently probably too restrained - figure out
    //                 if we need to constraint type, and whether it actually
    //                 needs to an argument (or we can just extract it from `op`).
    template< typename pattern_t, typename abi_op_t, typename rewriter_t >
    auto reconstruct_aggregate(const pattern_t &pattern, abi_op_t op,
                               hl::RecordType record_type, rewriter_t &rewriter)
    {
        using reconstructor_t = field_allocator< pattern_t, abi_op_t >;
        auto state = reconstructor_t::mk_state(pattern, op);

        auto module_op = op->template getParentOfType< vast_module >();
        VAST_ASSERT(module_op);
        return reconstructor_t(state, module_op).run_on(record_type, rewriter);
    }

} // namespace vast::conv::abi

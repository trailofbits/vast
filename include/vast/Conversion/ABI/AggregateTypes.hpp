// Copyright (c) 2023-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
VAST_UNRELAX_WARNINGS

namespace vast::conv::abi
{
    /* Handles aggregate type reconstruction. */

    struct aggregate_construction_base
    {
      protected:
        template< typename pattern, typename op_t >
        struct state_t
        {
            // Current argument or result we work with.
            std::size_t arg_idx = 0;
            // Current offset into the type - how many bits are already used.
            std::size_t offset = 0;

            // Pattern that constructed this object - in case some
            // data from broader context is needed.
            const pattern &parent;
            // Current operation conversion is rewriting.
            op_t abi_op;

            state_t(const pattern &parent, op_t abi_op)
                : parent(parent), abi_op(abi_op)
            {}

            state_t( const state_t &) = delete;
            state_t( state_t &&) = delete;

            state_t &operator=(state_t) = delete;

            auto bw(auto w) { return parent.bw(w); };

            void advance()
            {
                ++arg_idx;
                offset = 0;
            }


            std::size_t align_paddding_size(mlir_type type)
            {
                auto align = parent.dl.getTypeABIAlignment(type) * 8;
                if (offset % align == 0)
                    return 0;
                return align - (offset % align);
            }
        };

        // TODO(conv:abi): Issue #423 - figure out how to make this not adhoc.
        //                 `SubElement` interface won't work, for pointers or arrays
        //                 we do not care.
        bool needs_nesting(mlir_type type) const
        {
            return contains_subtype< hl::RecordType >(type);
        }

        // TODO(conv:abi): Figure out how to instead use some generic helper.
        gap::generator< ll::StructGEPOp > field_ptrs(
            operation root, auto loc, auto &bld
        ) const {
            auto module_op = root->getParentOfType< vast_module >();
            VAST_ASSERT(module_op);
            VAST_ASSERT(root->getNumResults() == 1);

            auto trg_type = [&] {
                auto root_type = root->getResultTypes()[0];
                if (auto ptr_type = mlir::dyn_cast< hl::PointerType >(root_type))
                    return ptr_type.getElementType();
                return root_type;
            }();

            auto def = hl::definition_of(trg_type, module_op);
            std::size_t idx = 0;
            for (const auto &[name, type] : def.getFieldsInfo()) {
                auto ptr_type = hl::PointerType::get(root->getContext(), type);
                auto gep = bld.template create< ll::StructGEPOp >(
                    loc, ptr_type, root->getResult(0), bld.getI32IntegerAttr(idx),
                    bld.getStringAttr(name));
                ++idx;
                co_yield gep;
            }
        }
    };

    // TODO(conv:abi): Parametrize by materializer, which will resolve what dialect
    //                is the target
    //                 * hl
    //                 * ll-vars
    //                 * llvm
    template< typename pattern, typename op_t >
    struct aggregate_reconstructor : aggregate_construction_base
    {
      protected:
        using self_t = aggregate_reconstructor< pattern, op_t >;

        struct state_t : aggregate_construction_base::state_t< pattern, op_t >
        {
            using base = aggregate_construction_base::state_t< pattern, op_t >;
            using base::base;

            using base::bw;
            using base::advance;

            auto arg()
            {
                return this->abi_op.getOperands()[this->arg_idx];
            };


            bool fits(mlir_type type)
            {
                return this->bw(arg()) >= this->offset + bw(type);
            };


            // TODO(conv:abi): Possibly add some debug prints to help us
            //                 debug some weird corner cases in the future?
            mlir::Value allocate(mlir_type type, auto &rewriter)
            {
                auto start = this->offset;
                this->offset += bw(type);
                if (bw(type) == bw(arg()))
                    return arg();

                return rewriter.template create< ll::Extract >(
                        this->abi_op.getLoc(), type, arg(),
                        start, this->offset);
            };

            void adjust_by_align(mlir_type type)
            {
                this->offset += this->align_paddding_size(type);
            }
        };

        state_t &state;
        vast_module mod;
        std::vector< mlir::Value > partials;


        mlir::Value run_on(mlir_type root_type, auto &rewriter)
        {
            auto handle_type = [&](mlir_type field_type) -> mlir::Value
            {
                if (needs_nesting(field_type))
                    return self_t(state, mod).run_on(field_type, rewriter);

                state.adjust_by_align(field_type);

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

      public:

        aggregate_reconstructor(state_t &state, vast_module mod)
            : state(state),
              mod(mod)
        {}

        static state_t mk_state(const pattern &parent, op_t abi_op)
        {
            return state_t(parent, abi_op);
        }

        mlir::Value run(mlir_type root, auto &rewriter)
        {
            return run_on(root, rewriter);
        }
    };

    /* Aggregate type deconstruction. */

    template< typename pattern, typename op_t >
    struct aggregate_deconstructor : aggregate_construction_base
    {
      protected:
        using self_t = aggregate_deconstructor< pattern, op_t >;

        struct state_t : aggregate_construction_base::state_t< pattern, op_t >
        {
            using base = aggregate_construction_base::state_t< pattern, op_t >;
            using base::base;

            using base::bw;
            using base::advance;

            std::vector< mlir::Value > current;

            auto dst()
            {
                VAST_CHECK(
                    this->arg_idx < this->abi_op.getResults().getTypes().size(),
                    "Trying to access {0} index in result types of {1}",
                    this->arg_idx, this->abi_op);
                return this->abi_op.getResults().getTypes()[this->arg_idx];
            }

            bool fits(mlir_type type)
            {
                return bw(dst()) >= this->offset + bw(type);
            };

            mlir::Value construct(auto &rewriter)
            {
                // `std::vector` is guaranteed to be empty after move.
                auto out = rewriter.template create< ll::Concat >(
                    this->abi_op.getLoc(),
                    dst(), std::move(current));

                advance();
                current.clear();
                return out;
            }

            // Returns value only one destination is saturated.
            auto allocate(mlir_type type, auto &rewriter, auto val)
                -> std::optional< mlir::Value >
            {
                auto start = this->offset;

                if (fits(type))
                {
                    this->offset += bw(type);
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
                           this->abi_op.getContext(),
                           size,
                           mlir::IntegerType::Signless);
                };

                auto prefix = rewriter.template create< ll::Extract >(
                    this->abi_op.getLoc(), mk_int_type(breakpoint), val,
                    0, breakpoint);

                current.push_back(prefix);

                auto suffix_size = (start + bw(type)) - bw(dst());
                auto to_yield = construct(rewriter);

                auto suffix = rewriter.template create< ll::Extract >(
                    this->abi_op.getLoc(), mk_int_type(suffix_size), val,
                    breakpoint, bw(type));

                this->offset = suffix_size;
                current.push_back(suffix);

                return { to_yield };
            }

            void adjust_by_align(auto &rewriter, auto loc, mlir_type type)
            {
                auto pad_by = this->align_paddding_size(type);
                if (pad_by == 0)
                    return;

                auto i_type = mlir::IntegerType::get(
                    this->abi_op.getContext(), pad_by, mlir::IntegerType::Signless);
                auto val = llvm::APSInt(pad_by, false);

                current.push_back(rewriter.template create< hl::ConstantOp >(loc, i_type, val));
            }
        };

        state_t &state;
        vast_module mod;
        std::vector< mlir::Value > partials;

        bool needs_nesting(mlir_type type) const
        {
            return contains_subtype< hl::RecordType >(type);
        }

        void run_on(operation root, auto &rewriter)
        {
            auto handle_field = [&](auto gep)
            {
                auto field_type = gep.getType();
                auto ptr_type = mlir::dyn_cast< hl::PointerType >(field_type);
                VAST_ASSERT(ptr_type);
                auto element_type = ptr_type.getElementType();

                if (needs_nesting(element_type))
                    return self_t(state, mod).run_on(gep.getOperation(), rewriter);

                auto rvalue = rewriter.template create< ll::Load >(
                    gep.getLoc(), element_type, gep) ;

                state.adjust_by_align(rewriter, gep.getLoc(), rvalue.getType());
                if (auto val = state.allocate(element_type, rewriter, rvalue))
                    partials.push_back(*val);
            };

            auto loc = root->getLoc();
            for (auto field_gep : this->field_ptrs(root, loc, rewriter))
                handle_field(field_gep);
        }

      public:
        aggregate_deconstructor(state_t &state, vast_module mod)
            : state(state),
              mod(mod)
        {}

        auto run(operation root, auto &rewriter) &&
        {
            run_on(root, rewriter);
            // Now construct the rest into a value
            if (!this->state.current.empty()) {
                auto val = state.construct(rewriter);
                partials.push_back(val);
            }

            return std::move(partials);
        }

        static state_t mk_state(const pattern &parent, op_t abi_op)
        {
            return state_t(parent, abi_op);
        }
    };

    // Top-level hooks to perform conversion from `abi.` to executable dialect.
    // Structures are reconstructed/deconstructed in such way that data flow
    // can be followed without memory operations.

    // Nested attributes are flattened together into one value.
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
    // From one value, attributes are extracted and structures are reconstructed
    // (including nested structures).
    template< typename pattern_t, typename abi_op_t, typename rewriter_t >
    auto reconstruct_aggregate(const pattern_t &pattern, abi_op_t op,
                               hl::RecordType record_type, rewriter_t &rewriter)
    {
        using reconstructor_t = aggregate_reconstructor< pattern_t, abi_op_t >;
        auto state = reconstructor_t::mk_state(pattern, op);

        auto module_op = op->template getParentOfType< vast_module >();
        VAST_ASSERT(module_op);
        return reconstructor_t(state, module_op).run(record_type, rewriter);
    }

} // namespace vast::conv::abi

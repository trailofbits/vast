// Copyright (c) 2022, Trail of Bits, Inc.

#pragma once

VAST_RELAX_WARNINGS
#include <mlir/IR/Types.h>
VAST_UNRELAX_WARNINGS

#include "vast/Util/Maybe.hpp"
#include "vast/Util/TypeConverter.hpp"

// TODO(lukas): Possibly move this out of Util?

namespace vast::util::tc
{
    // TODO(lukas): Implement.
    static inline bool is_variadic(mlir::FuncOp op)
    {
        return true;
    }

    static inline auto convert_fn_t(auto &tc, mlir::FuncOp op)
    -> std::tuple< mlir::TypeConverter::SignatureConversion, mlir::Type >
    {
        mlir::TypeConverter::SignatureConversion conversion(op.getNumArguments());
        auto target_type = tc.convertFunctionSignature(
                op.getType(), is_variadic(op), conversion);
        return { std::move(conversion), target_type };
    }

    struct LLVMTypeConverter : mlir::LLVMTypeConverter, util::TCHelpers< LLVMTypeConverter >
    {
        using parent_t = mlir::LLVMTypeConverter;
        using helpers_t = util::TCHelpers< LLVMTypeConverter >;
        using maybe_type_t = typename util::TCHelpers< LLVMTypeConverter >::maybe_type_t;
        using self_t = LLVMTypeConverter;

        template< typename ... Args >
        LLVMTypeConverter(Args && ... args) : parent_t(std::forward< Args >(args) ... )
        {
            addConversion([&](hl::LValueType t) { return this->convert_lvalue_type(t); });
            addConversion([&](hl::PointerType t) { return this->convert_pointer_type(t); });
            addConversion([&](mlir::MemRefType t) { return this->convert_memref_type(t); });
            addConversion([&](mlir::UnrankedMemRefType t) {
                    return this->convert_memref_type(t);
            });
            // Overriding the inherited one to provide way to handle `hl.lvalue` in args.
            addConversion([&](mlir::FunctionType t) {
                    return this->convert_fn_t(t);
            });
            addConversion([&](mlir::NoneType t) {
                    return mlir::LLVM::LLVMVoidType::get(t.getContext());
            });

        }

        maybe_types_t do_conversion(mlir::Type t)
        {
            types_t out;
            if (mlir::succeeded(this->convertTypes(t, out)))
                return { std::move( out ) };
            return {};
        }

        auto make_ptr_type()
        {
            return [&](auto t)
            {
                VAST_ASSERT(!t.template isa< mlir::NoneType >());
                return mlir::LLVM::LLVMPointerType::get(t);
            };
        }

        maybe_type_t convert_lvalue_type(hl::LValueType t)
        {
            return Maybe(t.getElementType()).and_then(helpers_t::convert_type_to_type())
                                            .unwrap()
                                            .and_then(self_t::make_ptr_type())
                                            .take_wrapped< maybe_type_t >();
        }

        maybe_type_t convert_pointer_type(hl::PointerType t)
        {
            return Maybe(t.getElementType()).and_then(convert_type_to_type())
                                            .unwrap()
                                            .and_then(self_t::make_ptr_type())
                                            .take_wrapped< maybe_type_t >();
        }

        auto make_array(auto shape_)
        {
            return [shape = std::move(shape_)](auto t)
            {
                mlir::Type out = mlir::LLVM::LLVMArrayType::get(t, shape.back());
                for (int i = shape.size() - 2; i >= 0; --i)
                {
                    out = mlir::LLVM::LLVMArrayType::get(out, shape[i]);
                }
                return out;
            };
        }

        maybe_type_t convert_memref_type(mlir::MemRefType t)
        {
            return Maybe(t.getElementType()).and_then(helpers_t::convert_type_to_type())
                                            .unwrap()
                                            .and_then(make_array(t.getShape()))
                                            .take_wrapped< maybe_type_t >();
        }

        maybe_type_t convert_memref_type(mlir::UnrankedMemRefType t)
        {
            return {};
        }

        using signature_conversion_t = mlir::TypeConverter::SignatureConversion;
        using maybe_signature_conversion_t = std::optional< signature_conversion_t >;

        maybe_signature_conversion_t get_conversion_signature(mlir::FuncOp fn,
                                                              bool variadic)
        {
            signature_conversion_t conversion(fn.getNumArguments());
            for (auto arg : llvm::enumerate(fn.getType().getInputs()))
            {
                auto cty = convert_arg_t(arg.value());
                if (!cty)
                    return {};
                conversion.addInputs(arg.index(), *cty);
            }
            return { std::move(conversion) };
        }

        maybe_type_t convert_fn_t(mlir::FunctionType t)
        {
            auto a_res = this->on_types(t.getInputs(), &LLVMTypeConverter::convert_arg_t);
            auto r_res = this->on_types(t.getResults(), &LLVMTypeConverter::convert_ret_t);

            if (!a_res || !r_res)
                return {};

            // LLVM function can have only one return value;
            if (r_res->size() != 1)
                return {};
            // TODO(lukas): Not sure how to get info if the type is variadic or not here.
            return mlir::LLVM::LLVMFunctionType::get(r_res->front(), *a_res, false);
        }

        maybe_types_t on_types(auto range, auto fn)
        {
            types_t out;
            auto append = appender(out);

            for (auto t : range)
                if (auto cty = (this->*fn)(t))
                    append(std::move(*cty));
                else
                    return {};
            return { std::move(out) };
        }

        maybe_types_t convert_arg_t(mlir::Type t)
        {
            if (auto lvalue = t.dyn_cast< hl::LValueType >())
                return this->convert_type_to_types(lvalue.getElementType());
            return this->convert_type_to_types(t);
        }

        maybe_types_t convert_ret_t(mlir::Type t)
        {
            if (auto lvalue = t.dyn_cast< hl::LValueType >())
                return this->convert_type_to_types(lvalue.getElementType());
            return this->convert_type_to_types(t);
        }
    };

} // namespace vast::util

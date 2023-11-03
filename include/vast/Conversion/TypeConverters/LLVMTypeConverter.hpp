// Copyright (c) 2022, Trail of Bits, Inc.

#pragma once

VAST_RELAX_WARNINGS
#include <mlir/Conversion/LLVMCommon/TypeConverter.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMTypes.h>
#include <mlir/IR/Types.h>
VAST_UNRELAX_WARNINGS

#include "vast/Dialect/HighLevel/HighLevelTypes.hpp"
#include "vast/Dialect/HighLevel/HighLevelUtils.hpp"
#include "vast/Util/Maybe.hpp"

#include "vast/Conversion/TypeConverters/TypeConverter.hpp"

// TODO(lukas): Possibly move this out of Util?

namespace vast::conv::tc {

    namespace LLVM = mlir::LLVM;

    struct LLVMTypeConverter
        : mlir::LLVMTypeConverter
        , tc::mixins< LLVMTypeConverter >
    {
        using base      = mlir::LLVMTypeConverter;
        using helpers_t = tc::mixins< LLVMTypeConverter >;

        template< typename... Args >
        LLVMTypeConverter(Args &&...args) : base(std::forward< Args >(args)...) {
            addConversion([&](hl::LabelType t) { return t; });
            addConversion([&](hl::DecayedType t) { return this->convert_decayed(t); });
            addConversion([&](hl::LValueType t) { return this->convert_lvalue_type(t); });
            addConversion([&](hl::PointerType t) { return this->convert_pointer_type(t); });
            addConversion([&](mlir::MemRefType t) { return this->convert_memref_type(t); });
            addConversion([&](mlir::UnrankedMemRefType t) {
                return this->convert_memref_type(t);
            });
            // Overriding the inherited one to provide way to handle `hl.lvalue` in args.
            addConversion([&](core::FunctionType t) { return this->convert_fn_t(t); });
            addConversion([&](mlir::NoneType t) {
                return LLVM::LLVMVoidType::get(t.getContext());
            });
        }

        // Moving the converter caused bugs in the code - since the base class has no comments
        // on this, we defensively forbid any sorts of copies/moves. This should usually pose
        // no problem as one type converter per pass has long enough lifetime to be passed
        // as a reference.
        LLVMTypeConverter(const LLVMTypeConverter &) = delete;
        LLVMTypeConverter(LLVMTypeConverter &&)      = delete;

        LLVMTypeConverter &operator=(const LLVMTypeConverter &) = delete;
        LLVMTypeConverter &operator=(LLVMTypeConverter &&)      = delete;

        maybe_types_t do_conversion(mlir::Type t) {
            types_t out;
            if (mlir::succeeded(this->convertTypes(t, out))) {
                return { std::move(out) };
            }
            return {};
        }

        auto make_ptr_type() {
            return [&](auto t) {
                VAST_ASSERT(!t.template isa< mlir::NoneType >());
                return LLVM::LLVMPointerType::get(t);
            };
        }

        maybe_type_t convert_decayed(hl::DecayedType t) {
            VAST_UNREACHABLE(
                "We shouldn't encounter decayed this late in the pipeline, {0}", t
            );
            return {};
        }

        maybe_type_t convert_lvalue_type(hl::LValueType t) {
            return Maybe(t.getElementType())
                .and_then(helpers_t::convert_type_to_type())
                .unwrap()
                .and_then(make_ptr_type())
                .take_wrapped< maybe_type_t >();
        }

        maybe_type_t convert_pointer_type(hl::PointerType t) {
            return Maybe(t.getElementType())
                .and_then(convert_type_to_type())
                .unwrap()
                .and_then(make_ptr_type())
                .take_wrapped< maybe_type_t >();
        }

        auto make_array(auto shape) {
            return [shape = std::move(shape)](auto t) {
                auto out = LLVM::LLVMArrayType::get(t, shape.back());
                for (int i = shape.size() - 2; i >= 0; --i) {
                    out = LLVM::LLVMArrayType::get(out, shape[i]);
                }
                return out;
            };
        }

        maybe_type_t convert_memref_type(mlir::MemRefType t) {
            return Maybe(t.getElementType())
                .and_then(helpers_t::convert_type_to_type())
                .unwrap()
                .and_then(make_array(t.getShape()))
                .take_wrapped< maybe_type_t >();
        }

        maybe_type_t convert_memref_type(mlir::UnrankedMemRefType t) { return {}; }

        maybe_signature_conversion_t
        get_conversion_signature(mlir::FunctionOpInterface fn, bool variadic) {
            signature_conversion_t conversion(fn.getNumArguments());
            auto fn_type = fn.getFunctionType().dyn_cast< core::FunctionType >();
            VAST_ASSERT(fn_type);
            for (auto arg : llvm::enumerate(fn_type.getInputs())) {
                auto cty = convert_arg_t(arg.value());
                if (!cty) {
                    return {};
                }
                conversion.addInputs(arg.index(), *cty);
            }
            return { std::move(conversion) };
        }

        maybe_type_t convert_fn_t(core::FunctionType t) {
            auto a_res = this->on_types(t.getInputs(), &LLVMTypeConverter::convert_arg_t);
            auto r_res = this->on_types(t.getResults(), &LLVMTypeConverter::convert_ret_t);

            if (!a_res || !r_res) {
                return {};
            }

            // LLVM function can have only one return value;
            VAST_ASSERT(r_res->size() <= 1);

            if (r_res->empty()) {
                r_res->emplace_back(LLVM::LLVMVoidType::get(t.getContext()));
            }

            return LLVM::LLVMFunctionType::get(r_res->front(), *a_res, t.isVarArg());
        }

        maybe_types_t on_types(auto range, auto fn) {
            types_t out;
            auto append = appender(out);

            for (auto t : range) {
                if (auto cty = (this->*fn)(t)) {
                    append(std::move(*cty));
                } else {
                    return {};
                }
            }
            return { std::move(out) };
        }

        maybe_types_t convert_arg_t(mlir::Type t) {
            if (auto lvalue = t.dyn_cast< hl::LValueType >()) {
                return this->convert_type_to_types(lvalue.getElementType());
            }
            return this->convert_type_to_types(t);
        }

        maybe_types_t convert_ret_t(mlir::Type t) {
            if (auto lvalue = t.dyn_cast< hl::LValueType >()) {
                return this->convert_type_to_types(lvalue.getElementType());
            }
            return this->convert_type_to_types(t);
        }
    };

    template< typename self_t >
    struct LLVMStruct
    {
      private:
        self_t &self() { return static_cast< self_t & >(*this); }

      public:
        maybe_types_t convert_field_types(mlir_type t) {
            auto field_types = self().get_field_types(t);
            if (!field_types)
                return {};

            mlir::SmallVector< mlir_type, 4 > out;
            for (auto field_type : *field_types) {
                auto c = self().convert_type_to_type(field_type);
                VAST_ASSERT(c);
                out.push_back(*c);
            }
            return { std::move(out) };
        }

        template< typename op_t >
        auto convert_recordlike() {
            // We need this prototype to handle recursive types.
            return [&](op_t t, mlir::SmallVectorImpl< mlir_type > &out,
                       mlir::ArrayRef< mlir_type > stack) -> logical_result {
                auto core = mlir::LLVM::LLVMStructType::getIdentified(
                    t.getContext(), t.getName());
                // Last element is `t`.
                auto bt = stack.drop_back();

                if (core.isOpaque() && std::ranges::find(bt, t) == bt.end()) {
                    if (auto body = convert_field_types(t)) {
                        // Multithreading may cause some issues?
                        auto status = core.setBody(*body, false);
                        VAST_ASSERT(mlir::succeeded(status));
                    }
                }
                out.push_back(core);
                return mlir::success();
            };
        }
    };

    // Requires that the named types *always* map to llvm struct types.
    // TODO(lukas): What about type aliases.
    struct FullLLVMTypeConverter : LLVMTypeConverter,
                                   LLVMStruct< FullLLVMTypeConverter >
    {
        using base = LLVMTypeConverter;

        vast_module mod;

        template< typename... Args >
        FullLLVMTypeConverter(vast_module mod,
                              Args &&...args)
        : base(std::forward< Args >(args)...),
          mod(mod) {
            addConversion([&](hl::ElaboratedType t) { return convert_elaborated_type(t); });
            addConversion(convert_recordlike< hl::RecordType >());
        }

        auto get_field_types(mlir_type t) -> std::optional< gap::generator< mlir_type > > {
            if (!mlir::isa< hl::RecordType >(t))
                return {};
            auto def = hl::definition_of(t, mod);
            // Nothing found, leave the structure opaque.
            if (!def) {
                return {};
            }
            return { hl::field_types(*def) };
        }

        maybe_type_t convert_elaborated_type(hl::ElaboratedType t) {
            return this->convert_type_to_type(t.getElementType());
        }

        maybe_type_t convert_record_type(hl::RecordType t) {
            auto &mctx = this->getContext();
            auto name  = t.getName();
            auto raw   = LLVM::LLVMStructType::getIdentified(&mctx, name);
            if (!raw || raw.getName() != name) {
                return {};
            }
            return { raw };
        }
    };

} // namespace vast::conv::tc

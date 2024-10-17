// Copyright (c) 2022, Trail of Bits, Inc.

#pragma once

VAST_RELAX_WARNINGS
#include <mlir/Analysis/DataLayoutAnalysis.h>

#include <mlir/Conversion/LLVMCommon/TypeConverter.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMTypes.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/IR/Types.h>

#include <llvm/ADT/ScopeExit.h>
VAST_UNRELAX_WARNINGS

#include "vast/Dialect/Core/CoreOps.hpp"

#include "vast/Dialect/HighLevel/HighLevelTypes.hpp"
#include "vast/Dialect/HighLevel/HighLevelUtils.hpp"
#include "vast/Util/Maybe.hpp"

#include "vast/Conversion/TypeConverters/TypeConverter.hpp"

// TODO(lukas): Possibly move this out of Util?

namespace vast::conv::tc {

    namespace LLVM = mlir::LLVM;

    // Really basic draft of how union lowering works, it should however be able to
    // handle most of the usual basic use-cases.
    struct union_lowering
    {
        using self_t = union_lowering;

        const mlir::DataLayout &dl;
        hl::UnionDeclOp union_decl;

        std::vector< mlir_type > fields = {};

        union_lowering(const mlir::DataLayout &dl, hl::UnionDeclOp union_decl)
            : dl(dl), union_decl(union_decl) {}

        union_lowering(const union_lowering &)  = delete;
        union_lowering(const union_lowering &&) = delete;

        unsigned align(mlir_type t) { return dl.getTypeABIAlignment(t); }

        unsigned size(mlir_type t) { return dl.getTypeSize(t); }

        self_t &compute_lowering() {
            mlir_type result;
            for (auto field_type : union_decl.getFieldTypes()) {
                result = merge(result, handle_field(storage_type(field_type)));
            }

            // TODO(conv:hl-to-ll-geps): Set packed.
            // We need to reconstruct the actual union type, should be a method
            // of `UnionDeclOp` maybe?
            // append_padding_bytes(union_decl.getType(), result);
            fields.push_back(result);

            return *this;
        }

        // This maybe should be extracted outside.
        mlir_type storage_type(mlir_type type) {
            // convert for mem - basically do things like `i1 -> i8`.
            // bitfield has some extras
            return mlir::IntegerType::get(type.getContext(), size(type) * 8);
        }

        void append_padding_bytes(mlir_type union_type, mlir_type field_type) {
            VAST_TODO("Unions that need padding!");
            // fields.push_back(array of size difference);
        }

        mlir_type merge(mlir_type current, mlir_type next) {
            if (!current) {
                return next;
            }

            if (align(current) < align(next)) {
                return next;
            }

            if (align(current) == align(next) && size(next) > size(current)) {
                return next;
            }

            return current;
        }

        mlir_type handle_field(mlir_type field_type) {
            // TODO(conv:hl-to-ll-geps): Bitfields.
            // TODO(conv:hl-to-ll-geps): Something related to zero initialization.
            return field_type;
        }

        static gap::generator< mlir_type > final_fields(std::vector< mlir_type > fields) {
            for (auto ft : fields) {
                co_yield ft;
            }
        }
    };

    using lower_to_llvm_options = mlir::LowerToLLVMOptions;


    struct llvm_type_converter
        : mlir::LLVMTypeConverter
        , tc::mixins< llvm_type_converter >
    {
        using base = mlir::LLVMTypeConverter;

        auto ignore_none_materialization() {
            return [&](mlir::OpBuilder &bld, mlir::NoneType t, mlir::ValueRange inputs, mlir::Location loc) {
                return bld.create< mlir::LLVM::ZeroOp >(loc, *this->convert_type_to_type(t));
            };
        }

        auto ignore_void_materialization() {
            return [&](mlir::OpBuilder &bld, mlir::LLVM::LLVMVoidType t, mlir::ValueRange inputs, mlir::Location loc) {
                return bld.create< mlir::LLVM::ZeroOp >(loc, *this->convert_type_to_type(t));
            };
        }

        llvm_type_converter(mcontext_t *mctx, const mlir::DataLayoutAnalysis &dl, lower_to_llvm_options opts, operation op)
            : base(mctx, opts, &dl)
        {
            addConversion([&](hl::LabelType t) { return t; });
            addConversion([&](hl::LValueType t) { return this->convert_lvalue_type(t); });
            addConversion([&](hl::PointerType t) { return this->convert_pointer_type(t); });
            addConversion([&](hl::ArrayType t) { return this->convert_array_type(t); });
            addConversion([&](mlir::UnrankedMemRefType t) {
                return this->convert_memref_type(t);
            });
            // Overriding the inherited one to provide way to handle `hl.lvalue` in args.
            addConversion([&](core::FunctionType t) { return this->convert_fn_t(t); });
            addConversion([&](mlir::NoneType t) {
                return LLVM::LLVMVoidType::get(t.getContext());
            });

            addConversion([op, this](hl::RecordType t, mlir::SmallVectorImpl< mlir_type > &out) {
                auto core = mlir::LLVM::LLVMStructType::getIdentified(
                    t.getContext(), t.getName()
                );

                auto &stack = this->getCurrentThreadRecursiveStack();
                if (core.isOpaque() && !llvm::count(stack, t)) {
                    stack.push_back(t);
                    auto pop = llvm::make_scope_exit([&]{ stack.pop_back(); });
                    if (auto body = convert_field_types(op, t)) {
                        [[maybe_unused]] auto status = core.setBody(*body, false);
                        VAST_ASSERT(mlir::succeeded(status));
                    }
                }
                out.push_back(core);
                return mlir::success();
            });

            // Since higher level model void values as "real" values in the IR, it can happen
            // we have `return foo()` where `foo` returns `void`. Now during conversion, because
            // of type mismatches, materialization callbacks will be invoked. By default, this
            // means unrealised conversions, but because the value in IR stops existing (as codegen
            // does not work with void values and LLVM dialect constraints) this means verifier error.
            // Instead we just emit a dummy constant, so that verifier does not hit unhappy path
            // that leads to a crash. These should be removed by subsequent passes as a cleanup.
            addArgumentMaterialization(ignore_none_materialization());
            addSourceMaterialization(ignore_none_materialization());
            addTargetMaterialization(ignore_none_materialization());

            addArgumentMaterialization(ignore_void_materialization());
            addSourceMaterialization(ignore_void_materialization());
            addTargetMaterialization(ignore_void_materialization());
        }

        // Moving the converter caused bugs in the code - since the base class has no comments
        // on this, we defensively forbid any sorts of copies/moves. This should usually pose
        // no problem as one type converter per pass has long enough lifetime to be passed
        // as a reference.
        llvm_type_converter(const llvm_type_converter &) = delete;
        llvm_type_converter(llvm_type_converter &&)      = delete;

        llvm_type_converter &operator=(const llvm_type_converter &) = delete;
        llvm_type_converter &operator=(llvm_type_converter &&)      = delete;


        maybe_types_t do_conversion(mlir_type t) const {
            types_t out;
            if (mlir::succeeded(this->convertTypes(t, out))) {
                return { std::move(out) };
            }
            return {};
        }

        auto make_ptr_type() {
            return [&](auto t) {
                VAST_ASSERT(!mlir::isa< mlir::NoneType >(t));
                return LLVM::LLVMPointerType::get(&this->getContext(), 0);
            };
        }

        maybe_type_t convert_lvalue_type(hl::LValueType t) {
            return Maybe(t.getElementType())
                .and_then(convert_type_to_type())
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
            return [shape = std::move(shape)](mlir_type t) {
                VAST_CHECK(shape, "Was not able to retrieve size of array type!");
                return mlir::LLVM::LLVMArrayType::get(t, *shape);
            };
        }

        maybe_type_t convert_array_type(hl::ArrayType t) {
            return Maybe(t.getElementType())
                .and_then(convert_type_to_type())
                .unwrap()
                .and_then(make_array(t.getSize()))
                .take_wrapped< maybe_type_t >();
        };

        maybe_type_t convert_memref_type(mlir::UnrankedMemRefType t) { return {}; }

        maybe_signature_conversion_t
        get_conversion_signature(core::function_op_interface fn, bool variadic) {
            signature_conversion_t conversion(fn.getNumArguments());
            auto fn_type = mlir::dyn_cast< core::FunctionType >(fn.getFunctionType());
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
            auto a_res = this->on_types(t.getInputs(), &llvm_type_converter::convert_arg_t);
            auto r_res = this->on_types(t.getResults(), &llvm_type_converter::convert_ret_t);

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

        maybe_types_t convert_arg_t(mlir_type t) {
            if (auto lvalue = mlir::dyn_cast< hl::LValueType >(t)) {
                return this->convert_type_to_types(lvalue.getElementType());
            }
            return this->convert_type_to_types(t);
        }

        maybe_types_t convert_ret_t(mlir_type t) {
            if (auto lvalue = mlir::dyn_cast< hl::LValueType >(t)) {
                return this->convert_type_to_types(lvalue.getElementType());
            }
            return this->convert_type_to_types(t);
        }

        auto get_field_types(operation op, hl::RecordType t) -> std::optional< gap::generator< mlir_type > > {
            if (!mlir::isa< hl::RecordType >(t)) {
                return {};
            }

            auto ts = core::symbol_table::lookup< core::type_symbol >(op, t.getName());
            VAST_CHECK(ts, "Record type {0} not present in the symbol table.", t.getName());
            auto def = mlir::dyn_cast_if_present< core::aggregate_interface >(ts);

            // Nothing found, leave the structure opaque.
            if (!def) {
                return {};
            }

            if (auto union_decl = mlir::dyn_cast< hl::UnionDeclOp >(*def)) {
                auto dl     = this->getDataLayoutAnalysis()->getAtOrAbove(union_decl);
                auto fields = union_lowering{ dl, union_decl }.compute_lowering().fields;
                return { union_lowering::final_fields(std::move(fields)) };
            } else {
                return { def.getFieldTypes() };
            }
        }

        maybe_types_t convert_field_types(operation op, hl::RecordType t) {
            auto field_types = get_field_types(op, t);
            if (!field_types) {
                return {};
            }

            mlir::SmallVector< mlir_type, 4 > out;
            for (auto field_type : *field_types) {
                auto c = self().convert_type_to_type(field_type);
                VAST_ASSERT(c);
                out.push_back(*c);
            }
            return { std::move(out) };
        }
    };

} // namespace vast::conv::tc

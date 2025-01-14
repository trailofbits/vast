
#include "vast/Util/Warnings.hpp"

#include "vast/Conversion/Parser/Passes.hpp"

VAST_RELAX_WARNINGS
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/YAMLParser.h>
#include <llvm/Support/YAMLTraits.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
VAST_UNRELAX_WARNINGS

#include "PassesDetails.hpp"

#include "vast/Conversion/Common/Mixins.hpp"
#include "vast/Conversion/Common/Patterns.hpp"

#include "vast/Conversion/TypeConverters/TypeConvertingPattern.hpp"

#include "vast/Util/Common.hpp"
#include "vast/Util/Terminator.hpp"

#include "vast/Dialect/Parser/Ops.hpp"
#include "vast/Dialect/Parser/Types.hpp"

#include "vast/Conversion/Parser/Config.hpp"

#include <ranges>

namespace vast::conv {

    enum class data_type { data, nodata, maybedata };

    mlir_type to_mlir_type(data_type type, mcontext_t *mctx) {
        switch (type) {
            case data_type::data: return pr::DataType::get(mctx);
            case data_type::nodata: return pr::NoDataType::get(mctx);
            case data_type::maybedata: return pr::MaybeDataType::get(mctx);
        }
    }

    template< typename... Ts >
    auto is_one_of(mlir_type ty) { return (mlir::isa< Ts >(ty) || ...); }

    bool is_parser_type(mlir_type ty) {
        return is_one_of< pr::DataType, pr::NoDataType, pr::MaybeDataType >(ty);
    }

    enum class function_category { sink, source, parser, nonparser };

    struct function_model
    {
        data_type return_type;
        std::vector< data_type > arguments;
        function_category category;

        bool is_sink() const { return category == function_category::sink; }

        bool is_source() const { return category == function_category::source; }

        bool is_parser() const { return category == function_category::parser; }

        bool is_nonparser() const { return category == function_category::nonparser; }

        mlir_type get_return_type(mcontext_t *mctx) const {
            return to_mlir_type(return_type, mctx);
        }

        mlir_type get_argument_type(unsigned int idx, mcontext_t *mctx) const {
            return to_mlir_type(idx < arguments.size() ? arguments[idx] : arguments.back(), mctx);
        }

        std::vector< mlir_type > get_argument_types(mcontext_t *mctx) const {
            std::vector< mlir_type > out;
            out.reserve(arguments.size());
            for (auto arg : arguments) {
                out.push_back(to_mlir_type(arg, mctx));
            }
            return out;
        }
    };

    struct named_function_model {
        std::string name;
        function_model model;
    };

    using function_models = llvm::StringMap< function_model >;

} // namespace vast::conv

LLVM_YAML_IS_SEQUENCE_VECTOR(vast::conv::data_type);
LLVM_YAML_IS_SEQUENCE_VECTOR(vast::conv::named_function_model);

using llvm::yaml::IO;
using llvm::yaml::MappingTraits;
using llvm::yaml::ScalarEnumerationTraits;

template<>
struct ScalarEnumerationTraits< vast::conv::data_type >
{
    static void enumeration(IO &io, vast::conv::data_type &value) {
        io.enumCase(value, "data", vast::conv::data_type::data);
        io.enumCase(value, "nodata", vast::conv::data_type::nodata);
        io.enumCase(value, "maybedata", vast::conv::data_type::maybedata);
    }
};

template<>
struct ScalarEnumerationTraits< vast::conv::function_category >
{
    static void enumeration(IO &io, vast::conv::function_category &value) {
        io.enumCase(value, "sink", vast::conv::function_category::sink);
        io.enumCase(value, "source", vast::conv::function_category::source);
        io.enumCase(value, "parser", vast::conv::function_category::parser);
        io.enumCase(value, "nonparser", vast::conv::function_category::nonparser);
    }
};

template<>
struct MappingTraits< vast::conv::function_model >
{
    static void mapping(IO &io, vast::conv::function_model &model) {
        io.mapRequired("return_type", model.return_type);
        io.mapRequired("arguments", model.arguments);
        io.mapRequired("category", model.category);
    }
};

template <>
struct MappingTraits< ::vast::conv::named_function_model > {
    static void mapping(IO &io, ::vast::conv::named_function_model &model) {
        io.mapRequired("function", model.name);
        io.mapRequired("model", model.model);
    }
};

namespace vast::conv {

    struct parser_conversion_config : base_conversion_config
    {
        using base = base_conversion_config;

        parser_conversion_config(
            rewrite_pattern_set patterns, conversion_target target,
            const function_models &models
        )
            : base(std::move(patterns), std::move(target)), models(models)
        {}

        template< typename pattern >
        void add_pattern() {
            auto ctx = patterns.getContext();
            if constexpr (std::is_constructible_v< pattern, mcontext_t * >) {
                patterns.template add< pattern >(ctx);
            } else if constexpr (std::is_constructible_v< pattern, mcontext_t *, const function_models & >) {
                patterns.template add< pattern >(ctx, models);
            } else {
                static_assert(false, "pattern does not have a valid constructor");
            }
        }

        const function_models &models;
    };

    struct function_type_converter
        : tc::identity_type_converter
        , tc::mixins< function_type_converter >
    {
        mcontext_t &mctx;
        std::optional< function_model > model;

        using mixin_base = tc::mixins< function_type_converter >;
        using mixin_base::convert_type_to_type;

        explicit function_type_converter(mcontext_t &mctx, std::optional< function_model > model)
            : mctx(mctx), model(model)
        {
            addConversion([this](tc::core_function_type ty) -> maybe_type_t {
                return convert_type_to_type(ty);
            });
        }

        maybe_type_t convert_arg_type(mlir_type ty, unsigned long idx) const {
            return Maybe(model
                ? model->get_argument_type(idx, &mctx)
                : pr::MaybeDataType::get(&mctx)
            ).template take_wrapped< maybe_type_t >();
        }

        maybe_type_t convert_type_to_type(tc::core_function_type ty) const {
            auto sig = signature_conversion(ty.getInputs());
            if (!sig) {
                return std::nullopt;
            }

            auto rty = convert_types_to_types(ty.getResults());
            if (!rty) {
                return std::nullopt;
            }

            return tc::core_function_type::get(
                ty.getContext(), sig->getConvertedTypes(), *rty, ty.isVarArg()
            );
        }

        maybe_type_t convert_type_to_type(mlir_type ty) const {
            if (auto ft = mlir::dyn_cast< tc::core_function_type >(ty)) {
                return convert_type_to_type(ft);
            }
            return Maybe(model
                ? model->get_return_type(&mctx)
                : pr::MaybeDataType::get(&mctx)
            ).template take_wrapped< maybe_type_t >();
        }
    };

    namespace pattern {

        mlir_value cast_value(mlir_value val, mlir_type ty, conversion_rewriter &rewriter) {
            if (auto cast = mlir::dyn_cast< mlir::UnrealizedConversionCastOp >(val.getDefiningOp())) {
                if (cast->getOperand(0).getType() == ty) {
                    return cast.getOperand(0);
                }
            }

            return rewriter.create< mlir::UnrealizedConversionCastOp >(
                val.getLoc(), ty, val
            ).getResult(0);
        }

        std::vector< mlir_value > convert_value_types(
            mlir::ValueRange values, mlir::TypeRange types, auto &rewriter
        ) {
            std::vector< mlir_value > out;
            out.reserve(values.size());

            for (size_t i = 0; i < values.size(); ++i) {
                // use last type for variadic operations
                auto ty = i < types.size() ? types[i] : types.back();
                auto val = values[i];
                out.push_back(val.getType() == ty ? val : cast_value(val, ty, rewriter));
            }

            return out;
        }

        template< typename op_t >
        struct parser_conversion_pattern_base
            : mlir_pattern_mixin< operation_conversion_pattern< op_t > >
            , mlir::OpConversionPattern< op_t >
        {
            using base = mlir::OpConversionPattern< op_t >;

            parser_conversion_pattern_base(mcontext_t *mctx, const function_models &models)
                : base(mctx), models(models)
            {}

            static std::optional< function_model > get_model(
                const function_models &models, string_ref name
            ) {
                if (auto kv = models.find(name); kv != models.end()) {
                    return kv->second;
                }

                return std::nullopt;
            }

            std::optional< function_model > get_model(string_ref name) const {
                return get_model(models, name);
            }

            const function_models &models;
        };

        //
        // Parser operation conversion patterns
        //
        template< typename op_t >
        struct ToNoParse : one_to_one_conversion_pattern< op_t, pr::NoParse >
        {
            using base = one_to_one_conversion_pattern< op_t, pr::NoParse >;
            using base::base;

            using adaptor_t = typename op_t::Adaptor;

            logical_result matchAndRewrite(
                op_t op, adaptor_t adaptor, conversion_rewriter &rewriter
            ) const override {
                auto rty = to_mlir_type(data_type::nodata, rewriter.getContext());
                auto args = convert_value_types(adaptor.getOperands(), rty, rewriter);
                auto converted = rewriter.create< pr::NoParse >(op.getLoc(), rty, args);
                rewriter.replaceOpWithNewOp< mlir::UnrealizedConversionCastOp >(
                    op, op.getType(), converted->getResult(0)
                );
                return mlir::success();
            }

            static void legalize(parser_conversion_config &cfg) {
                cfg.target.addLegalOp< pr::NoParse >();
                cfg.target.addLegalOp< mlir::UnrealizedConversionCastOp >();
                cfg.target.addIllegalOp< op_t >();
            }
        };

        struct CallConversion : parser_conversion_pattern_base< hl::CallOp >
        {
            using op_t = hl::CallOp;
            using base = parser_conversion_pattern_base< op_t >;
            using base::base;

            using adaptor_t = typename op_t::Adaptor;

            logical_result matchAndRewrite(
                op_t op, adaptor_t adaptor, conversion_rewriter &rewriter
            ) const override {
                if (op.getCallee().empty()) {
                    return mlir::failure();
                }

                auto callee = op.getCallee();
                if (auto kv = models.find(callee); kv != models.end()) {
                    const auto &[_, model] = *kv;
                    auto modeled = create_op_from_model(model, op, adaptor, rewriter);
                    rewriter.replaceOpWithNewOp< mlir::UnrealizedConversionCastOp >(
                        op, op.getResultTypes(), modeled->getResult(0)
                    );

                    return mlir::success();
                }

                return mlir::failure();
            }

            operation create_op_from_model(
                function_model model, op_t op, adaptor_t adaptor, conversion_rewriter &rewriter
            ) const {
                auto rty = model.get_return_type(op.getContext());
                auto arg_tys = model.get_argument_types(op.getContext());
                auto args = convert_value_types(adaptor.getOperands(), arg_tys, rewriter);

                if (model.is_source()) {
                    return rewriter.create< pr::Source >(op.getLoc(), rty, args);
                }

                if (model.is_sink()) {
                    return rewriter.create< pr::Sink >(op.getLoc(), rty, args);
                }

                if (model.is_parser()) {
                    return rewriter.create< pr::Parse >(op.getLoc(), rty, args);
                }

                if (model.is_nonparser()) {
                    return rewriter.create< pr::NoParse >(op.getLoc(), rty, args);
                }

                VAST_UNREACHABLE("Unknown function category");
            }

            static void legalize(parser_conversion_config &cfg) {
                cfg.target.addLegalOp< pr::NoParse, pr::Parse, pr::Source, pr::Sink >();
                cfg.target.addLegalOp< mlir::UnrealizedConversionCastOp >();
                cfg.target.addDynamicallyLegalOp< op_t >([models = cfg.models](op_t op) {
                    return models.count(op.getCallee()) == 0;
                });
            }
        };

        struct ParamConversion
            : one_to_one_conversion_pattern< hl::ParmVarDeclOp, pr::Decl >
        {
            using op_t = hl::ParmVarDeclOp;
            using base = one_to_one_conversion_pattern< hl::ParmVarDeclOp, pr::Decl >;
            using base::base;

            using adaptor_t = typename op_t::Adaptor;

            logical_result matchAndRewrite(
                op_t op, adaptor_t adaptor, conversion_rewriter &rewriter
            ) const override {
                rewriter.replaceOpWithNewOp<  pr::Decl >(
                    op, op.getSymName(), op.getParam().getType()
                );

                return mlir::success();
            }
        };

        struct DeclRefConversion
            : one_to_one_conversion_pattern< hl::DeclRefOp, pr::Ref >
        {
            using op_t = hl::DeclRefOp;
            using base = one_to_one_conversion_pattern< op_t, pr::Ref >;
            using base::base;

            using adaptor_t = typename op_t::Adaptor;

            logical_result matchAndRewrite(
                op_t op, adaptor_t adaptor, conversion_rewriter &rewriter
            ) const override {
                auto rewrite = [&] (auto ty) {
                    ty = is_parser_type(ty) ? ty : pr::MaybeDataType::get(rewriter.getContext());
                    auto converted = rewriter.create< pr::Ref >(op.getLoc(), ty, op.getName());
                    rewriter.replaceOpWithNewOp< mlir::UnrealizedConversionCastOp >(
                        op, op.getType(), converted->getResult(0)
                    );
                    return mlir::success();
                };

                if (auto vs = core::symbol_table::lookup< core::var_symbol >(op, op.getName())) {
                    if (auto var = mlir::dyn_cast< hl::VarDeclOp >(vs)) {
                        return rewrite(var.getType());
                    } else if (auto par = mlir::dyn_cast< hl::ParmVarDeclOp >(vs)) {
                        return rewrite(par.getParam().getType());
                    }
                }

                return mlir::failure();
            }
        };

        struct ReturnConversion
            : parser_conversion_pattern_base< hl::ReturnOp >
        {
            using op_t = hl::ReturnOp;
            using base = parser_conversion_pattern_base< op_t >;
            using base::base;

            using adaptor_t = typename op_t::Adaptor;

            logical_result matchAndRewrite(
                op_t op, adaptor_t adaptor, conversion_rewriter &rewriter
            ) const override {
                auto func = op->getParentOfType< hl::FuncOp >();
                auto model = get_model(func.getSymName());

                auto rty = model
                    ? model->get_return_type(rewriter.getContext())
                    : pr::MaybeDataType::get(rewriter.getContext());

                rewriter.replaceOpWithNewOp< op_t >(
                    op, convert_value_types(adaptor.getOperands(), rty, rewriter)
                );

                return mlir::success();
            }

            static void legalize(parser_conversion_config &cfg) {
                cfg.target.addLegalOp< mlir::UnrealizedConversionCastOp >();
                cfg.target.addDynamicallyLegalOp< op_t >([](op_t op) {
                    for (auto ty : op.getResult().getType()) {
                        if (!is_parser_type(ty)) {
                            return false;
                        }
                    }
                    return true;
                });
            }
        };

        struct FuncConversion
            : parser_conversion_pattern_base< hl::FuncOp >
            , tc::op_type_conversion< hl::FuncOp, function_type_converter >
        {
            using op_t = hl::FuncOp;
            using base = parser_conversion_pattern_base< op_t >;
            using base::base;

            using adaptor_t = typename op_t::Adaptor;


            logical_result matchAndRewrite(
                op_t op, adaptor_t adaptor, conversion_rewriter &rewriter
            ) const override {
                auto tc = function_type_converter(*rewriter.getContext(), get_model(op.getSymName()));
                if (auto func_op = mlir::dyn_cast< core::function_op_interface >(op.getOperation())) {
                    return this->replace(func_op, rewriter, tc);
                }

                return mlir::failure();
            }

            static void legalize(parser_conversion_config &cfg) {
                cfg.target.addLegalOp< mlir::UnrealizedConversionCastOp >();
                cfg.target.addDynamicallyLegalOp< op_t >([models = cfg.models](op_t op) {
                    return function_type_converter(
                        *op.getContext(), get_model(models, op.getSymName())
                    ).isLegal(op.getFunctionType());
                });
            }
        };

        struct ExprConversion
            : parser_conversion_pattern_base< hl::ExprOp >
        {
            using op_t = hl::ExprOp;
            using base = parser_conversion_pattern_base< op_t >;
            using base::base;

            using adaptor_t = typename op_t::Adaptor;

            logical_result matchAndRewrite(
                op_t op, adaptor_t adaptor, conversion_rewriter &rewriter
            ) const override {
                auto body = op.getBody();
                if (!body) {
                    return mlir::failure();
                }

                auto yield = terminator< hl::ValueYieldOp >::get(*body);
                VAST_PATTERN_CHECK(yield, "Expected yield in: {0}", op);

                rewriter.inlineBlockBefore(body, op);
                rewriter.replaceOp(op, yield.op().getResult());
                rewriter.eraseOp(yield.op());

                return mlir::success();
            }

            static void legalize(parser_conversion_config &cfg) {
                cfg.target.addLegalOp< mlir::UnrealizedConversionCastOp >();
            }
        };

        struct VarDeclConversion
            : parser_conversion_pattern_base< hl::VarDeclOp >
        {
            using op_t = hl::VarDeclOp;
            using base = parser_conversion_pattern_base< op_t >;
            using base::base;

            using adaptor_t = typename op_t::Adaptor;

            logical_result matchAndRewrite(
                op_t op, adaptor_t adaptor, conversion_rewriter &rewriter
            ) const override {
                auto maybe = to_mlir_type(data_type::maybedata, rewriter.getContext());
                /* auto decl = */ rewriter.create< pr::Decl >(op.getLoc(), op.getSymName(), maybe);

                if (auto &init_region = op.getInitializer(); !init_region.empty()) {
                    VAST_PATTERN_CHECK(init_region.getBlocks().size() == 1, "Expected single block in: {0}", op);
                    auto &init_block  = init_region.back();
                    auto yield = terminator< hl::ValueYieldOp >::get(init_block);
                    VAST_PATTERN_CHECK(yield, "Expected yield in: {0}", op);

                    rewriter.inlineBlockBefore(&init_block, op);
                    rewriter.setInsertionPointAfter(op);
                    auto ref = rewriter.create< pr::Ref >(op.getLoc(), maybe, op.getSymName());
                    auto value = rewriter.create< mlir::UnrealizedConversionCastOp >(
                        yield.op().getLoc(), maybe, yield.op().getResult()
                    );
                    rewriter.create< pr::Assign >(yield.op().getLoc(), value.getResult(0), ref);
                    rewriter.eraseOp(yield.op());
                }

                rewriter.eraseOp(op);
                return mlir::success();
            }

            static void legalize(parser_conversion_config &cfg) {
                cfg.target.addLegalOp< pr::Decl, pr::Ref, pr::Assign >();
                cfg.target.addLegalOp< mlir::UnrealizedConversionCastOp >();
            }
        };

        using operation_conversions = util::type_list<
            ToNoParse< hl::ConstantOp >,
            ToNoParse< hl::ImplicitCastOp >,
            ToNoParse< hl::CmpOp >, ToNoParse< hl::FCmpOp >,
            // Integer arithmetic
            ToNoParse< hl::MulIOp >,
            ToNoParse< hl::DivSOp >, ToNoParse< hl::DivUOp >,
            ToNoParse< hl::RemSOp >, ToNoParse< hl::RemUOp >,
            // Floating point arithmetic
            ToNoParse< hl::AddFOp >, ToNoParse< hl::SubFOp >,
            ToNoParse< hl::MulFOp >, ToNoParse< hl::DivFOp >,
            ToNoParse< hl::RemFOp >,
            // Other operations
            ExprConversion,
            FuncConversion,
            ParamConversion,
            DeclRefConversion,
            VarDeclConversion,
            ReturnConversion,
            CallConversion
        >;

    } // namespace pattern

    struct HLToParserPass : ConversionPassMixin< HLToParserPass, HLToParserBase >
    {
        using base = ConversionPassMixin< HLToParserPass, HLToParserBase >;

        static conversion_target create_conversion_target(mcontext_t &mctx) {
            return conversion_target(mctx);
        }

        static void populate_conversions(parser_conversion_config &cfg) {
            base::populate_conversions< pattern::operation_conversions >(cfg);
        }

        void setup_pass() {
            load_and_parse(pr::parsers_config_path);

            if (!config.empty()) {
                load_and_parse(config);
            }
        }

        void load_and_parse(string_ref config) {
            auto file_or_err = llvm::MemoryBuffer::getFile(config);
            if (auto ec = file_or_err.getError()) {
                llvm::errs() << "Could not open config file: " << ec.message() << "\n";
                return;
            }

            std::vector< named_function_model > functions;

            llvm::yaml::Input yin(file_or_err.get()->getBuffer());
            yin >> functions;

            if (yin.error()) {
                llvm::errs() << "Error parsing config file: " << yin.error().message() << "\n";
                return;
            }

            for (auto &&named : functions) {
                models.insert_or_assign(std::move(named.name), std::move(named.model));
            }
        }

        parser_conversion_config make_config() {
            auto &ctx = getContext();
            return { rewrite_pattern_set(&ctx), create_conversion_target(ctx), models };
        }

        function_models models;
    };

} // namespace vast::conv

std::unique_ptr< mlir::Pass > vast::createHLToParserPass() {
    return std::make_unique< vast::conv::HLToParserPass >();
}

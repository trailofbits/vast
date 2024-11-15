
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
#include "vast/Util/Common.hpp"

#include "vast/Dialect/Parser/Ops.hpp"
#include "vast/Dialect/Parser/Types.hpp"

#include "vast/Conversion/Parser/Config.hpp"

#include <ranges>

namespace vast::conv {

    enum class data_type { data, nodata, anydata };

    mlir_type to_mlir_type(data_type type, mcontext_t *mctx) {
        switch (type) {
            case data_type::data: return pr::DataType::get(mctx);
            case data_type::nodata: return pr::NoDataType::get(mctx);
            case data_type::anydata: return pr::AnyDataType::get(mctx);
        }
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
        io.enumCase(value, "anydata", vast::conv::data_type::anydata);
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

    namespace pattern {
        template< typename op_t >
        struct parser_conversion_pattern_base
            : mlir_pattern_mixin< operation_conversion_pattern< op_t > >
            , mlir::OpConversionPattern< op_t >
        {
            using base = mlir::OpConversionPattern< op_t >;

            parser_conversion_pattern_base(mcontext_t *mctx, const function_models &models)
                : base(mctx), models(models)
            {}

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
                rewriter.replaceOpWithNewOp< pr::NoParse >(
                    op, op.getType(), adaptor.getOperands()
                );
                return mlir::success();
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
                std::vector< mlir_value > args;
                auto arg_tys = model.get_argument_types(op.getContext());

                if (arg_tys.size() < adaptor.getOperands().size()) {
                    for (auto i = arg_tys.size(); i < adaptor.getOperands().size(); ++i) {
                        arg_tys.push_back(arg_tys.back());
                    }
                }

                VAST_ASSERT(arg_tys.size() == adaptor.getOperands().size());

                for (auto [arg, ty] : llvm::zip(adaptor.getOperands(), arg_tys)) {
                    auto converted = [&] {
                        if (arg.getType() != ty) {
                            return rewriter.create< mlir::UnrealizedConversionCastOp >(
                                op.getLoc(), ty, arg
                            ).getResult(0);
                        } else {
                            return arg;
                        }
                    } ();

                    args.push_back(converted);
                }

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


            template< typename new_op_t >
            logical_result replace_with_new_op(
                op_t op, adaptor_t adaptor, conversion_rewriter &rewriter
            ) const {
                auto model = models.lookup(op.getCallee());

                rewriter.replaceOpWithNewOp< new_op_t >(
                    op, op.getResultTypes(), adaptor.getOperands()
                );
                return mlir::success();
            }

            static void legalize(parser_conversion_config &cfg) {
                cfg.target.addLegalOp< pr::NoParse, pr::Parse, pr::Source, pr::Sink >();
                cfg.target.addLegalOp< mlir::UnrealizedConversionCastOp >();
                cfg.target.addDynamicallyLegalOp< op_t >([models = cfg.models](op_t op) {
                    return models.count(op.getCallee()) == 0;
                });
            }
        };

        using operation_conversions = util::type_list<
            ToNoParse< hl::ConstantOp >,
            ToNoParse< hl::ImplicitCastOp >,
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

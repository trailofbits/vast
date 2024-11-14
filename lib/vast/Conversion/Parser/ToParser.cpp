
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

#include "vast/Util/Common.hpp"
#include "vast/Conversion/Common/Mixins.hpp"
#include "vast/Conversion/Common/Patterns.hpp"

#include "vast/Dialect/Parser/Ops.hpp"
#include "vast/Dialect/Parser/Types.hpp"

#include "vast/Conversion/Parser/Config.hpp"

namespace vast::conv
{
    using function_category_t = llvm::StringSet<>;

    struct function_categories
    {
        bool is_sink(string_ref name) const { return sinks.count(name); }
        bool is_source(string_ref name) const { return sources.count(name); }
        bool is_parser(string_ref name) const { return parsers.count(name); }
        bool is_nonparser(string_ref name) const { return nonparsers.count(name); }

        function_categories get_categories() const {
            return { sinks, sources, parsers, nonparsers };
        }

        const function_category_t &sinks;
        const function_category_t &sources;
        const function_category_t &parsers;
        const function_category_t &nonparsers;
    };

    struct parser_conversion_config
        : base_conversion_config
        , function_categories
    {
        using base = base_conversion_config;

        using function_categories::get_categories;

        parser_conversion_config(
              rewrite_pattern_set patterns
            , conversion_target target
            , function_categories categories
        )
            : base(std::move(patterns), std::move(target)), function_categories(categories)
        {}

        template< typename pattern >
        static constexpr bool requires_function_categories = std::is_constructible_v<
            pattern, mcontext_t *, function_categories
        >;

        template< typename pattern >
            requires requires_function_categories< pattern >
        void add_pattern() {
            patterns.template add< pattern >(patterns.getContext(), get_categories());
        }

        template< typename pattern >
        requires (!requires_function_categories< pattern >)
        void add_pattern() {
            patterns.template add< pattern >(patterns.getContext());
        }
    };

    namespace pattern
    {
        template< typename op_t >
        struct parser_conversion_pattern_base
            : mlir_pattern_mixin< operation_conversion_pattern< op_t > >
            , function_categories
            , mlir::OpConversionPattern< op_t >
        {
            using base = mlir::OpConversionPattern< op_t >;
            using base::base;

            parser_conversion_pattern_base(mcontext_t *mctx, function_categories categories)
                : function_categories(categories), base(mctx)
            {}
        };

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

        struct CallConversion
            : parser_conversion_pattern_base< hl::CallOp >
        {
            using op_t = hl::CallOp;
            using base = parser_conversion_pattern_base< op_t >;
            using base::base;

            using base::is_sink;
            using base::is_source;
            using base::is_parser;
            using base::is_nonparser;

            using adaptor_t = typename op_t::Adaptor;

            logical_result matchAndRewrite(
                op_t op, adaptor_t adaptor, conversion_rewriter &rewriter
            ) const override {
                if (op.getCallee().empty())
                    return mlir::failure();

                auto callee = op.getCallee();

                if (is_source(callee)) {
                    return replace_with_new_op< pr::Source >(op, adaptor, rewriter);
                } else if (is_sink(callee)) {
                    return replace_with_new_op< pr::Sink >(op, adaptor, rewriter);
                } else if (is_parser(callee)) {
                    return replace_with_new_op< pr::Parse >(op, adaptor, rewriter);
                } else if (is_nonparser(callee)) {
                    return replace_with_new_op< pr::NoParse >(op, adaptor, rewriter);
                } else {
                    return mlir::failure();
                }
            }

            template< typename new_op_t >
            logical_result replace_with_new_op(
                op_t op, adaptor_t adaptor, conversion_rewriter &rewriter
            ) const {
                rewriter.replaceOpWithNewOp< new_op_t >(op, op.getResultTypes(), adaptor.getOperands());
                return mlir::success();
            }

            static bool is_legal(op_t call, const function_categories &categories) {
                auto callee = call.getCallee();
                return !categories.is_source(callee)
                    && !categories.is_sink(callee)
                    && !categories.is_parser(callee)
                    && !categories.is_nonparser(callee);
            }

            static void legalize(parser_conversion_config &cfg) {
                auto cats = cfg.get_categories();
                cfg.target.addLegalOp< pr::NoParse, pr::Parse, pr::Source, pr::Sink >();
                cfg.target.addDynamicallyLegalOp< op_t >([cats](op_t op) { return is_legal(op, cats); });
            }
        };

        using all = util::type_list<
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
            base::populate_conversions< pattern::all >(cfg);
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

            std::unique_ptr< llvm::MemoryBuffer > &buff = *file_or_err;
            string_ref data = buff->getBuffer();

            llvm::SourceMgr smgr;
            smgr.AddNewSourceBuffer(std::move(buff), llvm::SMLoc());

            llvm::yaml::Stream yaml_stream(data, smgr);
            for (auto &doc : yaml_stream) {
                auto root = llvm::dyn_cast< llvm::yaml::MappingNode >(doc.getRoot());
                if (!root) {
                    llvm::errs() << "Invalid YAML format: root node is not a mapping\n";
                    return;
                }

                for (auto &kv : *root) {
                    auto key = llvm::dyn_cast< llvm::yaml::ScalarNode >(kv.getKey());
                    auto value = kv.getValue();

                    if (!key || !value) {
                        continue;
                    }

                    if (auto *function_list = llvm::dyn_cast< llvm::yaml::SequenceNode >(value)) {
                        auto category = key->getRawValue();
                        populate_function_set(category, function_list);
                    }
                }
            }
        }

        void populate_function_set(string_ref category, llvm::yaml::SequenceNode *function_list) {
            auto &target_set = get_category_set(category);

            for (auto &function : *function_list) {
                if (auto *function_name = llvm::dyn_cast< llvm::yaml::ScalarNode >(&function)) {
                    target_set.insert(function_name->getRawValue().str());
                }
            }
        }

        function_category_t & get_category_set(string_ref category) {
            if (category == "sink") {
                return sinks;
            } else if (category == "source") {
                return sources;
            } else if (category == "parse") {
                return parsers;
            } else if (category == "noparse") {
                return nonparsers;
            } else {
                VAST_UNREACHABLE("Unknown category: {0}", category);
            }
        }

        parser_conversion_config make_config() {
            auto &ctx = getContext();
            return {
                rewrite_pattern_set(&ctx),
                create_conversion_target(ctx),
                function_categories{
                    sinks, sources, parsers, nonparsers
                }
            };
        }

        function_category_t sources;
        function_category_t sinks;
        function_category_t parsers;
        function_category_t nonparsers;
    };

} // namespace vast::conv


std::unique_ptr< mlir::Pass > vast::createHLToParserPass() {
    return std::make_unique< vast::conv::HLToParserPass >();
}

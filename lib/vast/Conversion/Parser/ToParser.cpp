
#include "vast/Util/Warnings.hpp"

#include "vast/Conversion/Parser/Passes.hpp"

VAST_RELAX_WARNINGS
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/SourceMgr.h>
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
    namespace pattern
    {
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

        using all = util::type_list<
            ToNoParse< hl::ConstantOp >,
            ToNoParse< hl::ImplicitCastOp >
        >;

    } // namespace pattern


    struct HLToParserPass : ConversionPassMixin< HLToParserPass, HLToParserBase >
    {
        using base = ConversionPassMixin< HLToParserPass, HLToParserBase >;

        static conversion_target create_conversion_target(mcontext_t &mctx) {
            return conversion_target(mctx);
        }

        static void populate_conversions(auto &cfg) {
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

        llvm::StringSet<> & get_category_set(string_ref category) {
            if (category == "sink") {
                return sinks;
            } else if (category == "source") {
                return sources;
            } else if (category == "parse") {
                return parsers;
            } else if (category == "noparse") {
                return nonparsers;
            }

            VAST_UNREACHABLE("Unknown category: {0}", category);
        }

        llvm::StringSet<> sources;
        llvm::StringSet<> sinks;
        llvm::StringSet<> parsers;
        llvm::StringSet<> nonparsers;
    };

} // namespace vast::conv


std::unique_ptr< mlir::Pass > vast::createHLToParserPass() {
    return std::make_unique< vast::conv::HLToParserPass >();
}

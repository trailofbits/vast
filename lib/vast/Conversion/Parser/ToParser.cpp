
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
#include "Utils.hpp"

#include "vast/Conversion/Common/Mixins.hpp"
#include "vast/Conversion/Common/Patterns.hpp"

#include "vast/Conversion/TypeConverters/TypeConvertingPattern.hpp"

#include "vast/Util/Common.hpp"
#include "vast/Util/Terminator.hpp"

#include "vast/Dialect/Parser/Ops.hpp"
#include "vast/Dialect/Parser/Types.hpp"

#include "vast/Conversion/Parser/Config.hpp"

#include "vast/server/server.hpp"
#include "vast/server/types.hpp"

#include <ranges>

namespace vast::conv {


    enum class function_category { sink, source, parser, nonparser };

    struct function_model
    {
        pr::data_type return_type;
        std::vector< pr::data_type > arguments;
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

    struct location
    {
        std::string filePath;
        server::range range;
    };

    location get_location(file_loc_t loc) {
        return {
            .filePath = loc.getFilename().str(),
            .range = {
                .start = { loc.getLine(), loc.getColumn(), },
                .end = { loc.getLine(), loc.getColumn(), },
            },
        };
    }

    location get_location(name_loc_t loc) {
        return get_location(mlir::cast< file_loc_t >(loc.getChildLoc()));
    }

    std::optional< location > get_location(loc_t loc) {
        if (auto file_loc = mlir::dyn_cast< file_loc_t >(loc)) {
            return get_location(file_loc);
        } else if (auto name_loc = mlir::dyn_cast< name_loc_t >(loc)) {
            return get_location(name_loc);
        }

        return std::nullopt;
    }

    pr::data_type parse_type_name(const std::string &name) {
        if (name == "data") {
            return pr::data_type::data;
        } else if (name == "nodata") {
            return pr::data_type::nodata;
        } else {
            return pr::data_type::maybedata;
        }
    }

    function_category
    ask_user_for_category(vast::server::server_base &server, core::function_op_interface op) {
        auto loc = op.getLoc();
        auto sym = mlir::dyn_cast< core::SymbolOpInterface >(op.getOperation());
        VAST_ASSERT(sym);
        auto name = sym.getSymbolName().str();

        vast::server::input_request req{
            .type = {"nonparser", "sink", "source", "parser",},
            .text = "Please choose category for function `" + name + '`',
            .filePath = std::nullopt,
            .range = std::nullopt,
        };

        if (auto req_loc = get_location(loc)) {
            req.filePath = req_loc->filePath;
            req.range    = req_loc->range;
        }

        auto response = server.send_request(req);
        if (auto result = std::get_if< vast::server::input_request::response_type >(&response))
        {
            if (result->value == "nonparser") {
                return function_category::nonparser;
            } else if (result->value == "sink") {
                return function_category::sink;
            } else if (result->value == "source") {
                return function_category::source;
            } else if (result->value == "parser") {
                return function_category::parser;
            }
        }
        return function_category::nonparser;
    }

    pr::data_type ask_user_for_return_type(
        vast::server::server_base &server, core::function_op_interface op
    ) {
        auto loc = op.getLoc();
        auto sym = mlir::dyn_cast< core::SymbolOpInterface >(op.getOperation());
        VAST_ASSERT(sym);
        auto name = sym.getSymbolName().str();

        vast::server::input_request req{
            .type     = { "maybedata", "nodata", "data" },
            .text     = "Please choose return type for function `" + name + '`',
            .filePath = std::nullopt,
            .range    = std::nullopt,
        };

        if (auto req_loc = get_location(loc)) {
            req.filePath = req_loc->filePath;
            req.range    = req_loc->range;
        }

        auto response = server.send_request(req);
        if (auto result = std::get_if< vast::server::input_request::response_type >(&response))
        {
            return parse_type_name(result->value);
        }
        return pr::data_type::maybedata;
    }

    pr::data_type ask_user_for_argument_type(
        vast::server::server_base &server, core::function_op_interface op, unsigned int idx
    ) {
        auto num_body_args = op.getFunctionBody().getNumArguments();
        auto sym           = mlir::dyn_cast< core::SymbolOpInterface >(op.getOperation());
        VAST_ASSERT(sym);
        auto name = sym.getSymbolName().str();

        vast::server::input_request req{
            .type = { "maybedata", "nodata", "data" },
            .text = "Please choose a type for argument " + std::to_string(idx)
                + " of function `" + name + '`',
            .filePath = std::nullopt,
            .range    = std::nullopt,
        };

        if (idx < num_body_args) {
            auto arg = op.getArgument(idx);
            auto loc = arg.getLoc();
            if (auto req_loc = get_location(loc)) {
                req.filePath = req_loc->filePath;
                req.range    = req_loc->range;
            }
        }

        auto response = server.send_request(req);
        if (auto result = std::get_if< vast::server::input_request::response_type >(&response))
        {
            return parse_type_name(result->value);
        }
        return pr::data_type::maybedata;
    }

    function_model ask_user_for_function_model(
        vast::server::server_base &server, core::function_op_interface op
    ) {
        function_model model;
        model.return_type = ask_user_for_return_type(server, op);
        for (unsigned int i = 0; i < op.getNumArguments(); ++i) {
            model.arguments.push_back(ask_user_for_argument_type(server, op, i));
        }
        model.category = ask_user_for_category(server, op);
        return model;
    }

} // namespace vast::conv

LLVM_YAML_IS_SEQUENCE_VECTOR(vast::pr::data_type);
LLVM_YAML_IS_SEQUENCE_VECTOR(vast::conv::named_function_model);

using llvm::yaml::IO;
using llvm::yaml::MappingTraits;
using llvm::yaml::ScalarEnumerationTraits;

template<>
struct ScalarEnumerationTraits< vast::pr::data_type >
{
    static void enumeration(IO &io, vast::pr::data_type &value) {
        io.enumCase(value, "data", vast::pr::data_type::data);
        io.enumCase(value, "nodata", vast::pr::data_type::nodata);
        io.enumCase(value, "maybedata", vast::pr::data_type::maybedata);
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
            rewrite_pattern_set patterns, conversion_target target, function_models &models,
            vast::server::server_base *server
        )
            : base(std::move(patterns), std::move(target)), models(models), server(server) {}

        template< typename pattern >
        void add_pattern() {
            auto ctx = patterns.getContext();
            if constexpr (std::is_constructible_v< pattern, mcontext_t * >) {
                patterns.template add< pattern >(ctx);
            } else if constexpr (std::is_constructible_v<
                                     pattern, mcontext_t *, function_models &,
                                     vast::server::server_base * >)
            {
                patterns.template add< pattern >(ctx, models, server);
            } else {
                static_assert(false, "pattern does not have a valid constructor");
            }
        }

        function_models &models;
        vast::server::server_base *server;
    };

    using signature_conversion_t       = mlir::TypeConverter::SignatureConversion;
    using maybe_signature_conversion_t = std::optional< signature_conversion_t >;

    template< typename Range >
    bool all_of_parser_types(const Range &types) {
        return llvm::all_of(types, pr::is_parser_type);
    }

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
            addConversion([this](mlir_type ty) -> maybe_type_t {
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

            VAST_ASSERT(ty.getNumResults() == 1);
            auto rty = convert_type_to_type(ty.getResult(0));
            if (!rty) {
                return std::nullopt;
            }

            return tc::core_function_type::get(
                ty.getContext(), sig->getConvertedTypes(), *rty, ty.isVarArg()
            );
        }

        maybe_type_t convert_type_to_type(mlir_type ty) const {
            if (pr::is_parser_type(ty)) {
                return ty;
            }

            if (mlir::isa< hl::VoidType >(ty)) {
                return pr::NoDataType::get(&mctx);
            }

            if (auto ft = mlir::dyn_cast< tc::core_function_type >(ty)) {
                return convert_type_to_type(ft);
            }

            return Maybe(model
                ? model->get_return_type(&mctx)
                : pr::MaybeDataType::get(&mctx)
            ).template take_wrapped< maybe_type_t >();
        }

        maybe_signature_conversion_t
        get_conversion_signature(core::function_op_interface fn, bool variadic) {
            signature_conversion_t conversion(fn.getNumArguments());
            auto fty = mlir::dyn_cast< core::FunctionType >(fn.getFunctionType());
            VAST_ASSERT(fty);
            for (auto arg : llvm::enumerate(fty.getInputs())) {
                auto cty = convert_arg_type(arg.value(), arg.index());
                if (!cty) {
                    return {};
                }
                conversion.addInputs(arg.index(), *cty);
            }
            return { std::move(conversion) };
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
            value_range values, type_range types, auto &rewriter
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

        std::vector< mlir_value > realized_operand_values(value_range values, auto &rewriter) {
            std::vector< mlir_value > out;
            out.reserve(values.size());
            for (auto val : values) {
                out.push_back([&] () -> mlir_value {
                    if (auto cast = mlir::dyn_cast< mlir::UnrealizedConversionCastOp >(val.getDefiningOp())) {
                        if (pr::is_parser_type(cast.getOperand(0).getType())) {
                            return cast.getOperand(0);
                        }
                    }

                    if (!pr::is_parser_type(val.getType())) {
                        return rewriter.template create< mlir::UnrealizedConversionCastOp >(
                            val.getLoc(), pr::MaybeDataType::get(val.getContext()), val
                        ).getResult(0);
                    }

                    return val;
                }());
            }
            return out;
        }

        mlir_type join(mlir_type lhs, mlir_type rhs) {
            if (!lhs)
                return rhs;
            return lhs == rhs ? lhs : pr::MaybeDataType::get(lhs.getContext());
        }

        mlir_type top_type(value_range values) {
            mlir_type ty;
            for (auto val : values) {
                ty = join(ty, val.getType());
            }
            return ty;
        }

        template< typename op_t >
        struct parser_conversion_pattern_base
            : mlir_pattern_mixin< operation_conversion_pattern< op_t > >
            , mlir::OpConversionPattern< op_t >
        {
            using base = mlir::OpConversionPattern< op_t >;

            parser_conversion_pattern_base(
                mcontext_t *mctx, function_models &models, vast::server::server_base *server
            )
                : base(mctx), models(models), server(server) {}

            static std::optional< function_model > get_model(
                function_models &models, core::function_op_interface op,
                vast::server::server_base *server
            ) {
                auto sym = mlir::dyn_cast< core::SymbolOpInterface >(op.getOperation());
                VAST_ASSERT(sym);
                if (auto kv = models.find(sym.getSymbolName()); kv != models.end()) {
                    return kv->second;
                }

                if (server) {
                    auto model                  = ask_user_for_function_model(*server, op);
                    models[sym.getSymbolName()] = model;
                    return model;
                }

                return std::nullopt;
            }

            std::optional< function_model > get_model(core::function_op_interface op) const {
                return get_model(models, op, server);
            }

            function_models &models;
            vast::server::server_base *server;
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
                auto rty = to_mlir_type(pr::data_type::nodata, rewriter.getContext());
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

        template< typename op_t >
        struct ToMaybeParse : operation_conversion_pattern< op_t >
        {
            using base = operation_conversion_pattern< op_t >;
            using base::base;

            using adaptor_t = typename op_t::Adaptor;

            logical_result matchAndRewrite(
                op_t op, adaptor_t adaptor, conversion_rewriter &rewriter
            ) const override {
                auto args = realized_operand_values(adaptor.getOperands(), rewriter);
                auto rty = top_type(args);

                auto converted = [&] () -> operation {
                    auto matches_return_type = [rty] (auto val) { return val.getType() == rty; };
                    if (mlir::isa< pr::NoDataType >(rty) && llvm::all_of(args, matches_return_type))
                        return rewriter.create< pr::NoParse >(op.getLoc(), rty, args);
                    return rewriter.create< pr::MaybeParse >(op.getLoc(), rty, args);
                } ();

                VAST_ASSERT(op->getNumResults() == 1);
                rewriter.replaceOpWithNewOp< mlir::UnrealizedConversionCastOp >(
                    op, op->getOpResult(0).getType(), converted->getResult(0)
                );

                return mlir::success();
            }

            static void legalize(parser_conversion_config &cfg) {
                cfg.target.addLegalOp< pr::MaybeParse >();
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

                auto args = realized_operand_values(adaptor.getOperands(), rewriter);
                auto rty  = return_type(op, callee);
                auto converted = rewriter.create< hl::CallOp >(op.getLoc(), callee, rty, args);
                if (converted->getNumResults() == 1) {
                    rewriter.replaceOpWithNewOp< mlir::UnrealizedConversionCastOp >(
                        op, op.getResultTypes(), converted->getResult(0)
                    );
                } else {
                    rewriter.eraseOp(op);
                }

                return mlir::success();
            }

            mlir_type return_type(op_t op, string_ref callee) const {
                assert(op.getNumResults() == 1);
                auto rty = op.getResult(0).getType();
                if (isa< hl::VoidType >(rty)) {
                    return pr::NoDataType::get(op.getContext());
                }

                return pr::MaybeDataType::get(op.getContext());
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
                cfg.target.addDynamicallyLegalOp< hl::CallOp >([](hl::CallOp op) {
                    for (auto arg : op.getOperands()) {
                        if (!pr::is_parser_type(arg.getType())) {
                            return false;
                        }
                    }
                    return true;
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
                auto ty = adaptor.getParam().getType();
                rewriter.replaceOpWithNewOp< pr::Decl >(op, adaptor.getSymName(), ty);
                auto ref = rewriter.create< pr::Ref >(op.getLoc(), ty, adaptor.getSymName());
                rewriter.create< pr::Assign >(op.getLoc(), adaptor.getParam(), ref);
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
                    ty = pr::is_parser_type(ty) ? ty : pr::MaybeDataType::get(rewriter.getContext());
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

        template< typename SourceOp >
        struct ReturnConversion : parser_conversion_pattern_base< SourceOp >
        {
            using op_t = SourceOp;
            using base = parser_conversion_pattern_base< op_t >;
            using base::base;

            using base::get_model;

            using adaptor_t = typename op_t::Adaptor;

            logical_result matchAndRewrite(
                op_t op, adaptor_t adaptor, conversion_rewriter &rewriter
            ) const override {
                auto func  = op->template getParentOfType< hl::FuncOp >();
                auto tc    = function_type_converter(*rewriter.getContext(), get_model(func));
                auto rty   = tc.convert_types_to_types(op.getResult().getType());

                rewriter.replaceOpWithNewOp< op_t >(
                    op, convert_value_types(adaptor.getOperands(), *rty, rewriter)
                );

                return mlir::success();
            }

            static void legalize(parser_conversion_config &cfg) {
                cfg.target.addLegalOp< mlir::UnrealizedConversionCastOp >();
                cfg.target.addDynamicallyLegalOp< op_t >([](op_t op) {
                    return all_of_parser_types(op.getResult().getType());
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
                auto tc = function_type_converter(*rewriter.getContext(), get_model(op));
                auto fty               = op.getFunctionType();
                auto maybe_target_type = tc.convert_type_to_type(fty);
                auto maybe_signature   = tc.get_conversion_signature(op, fty.isVarArg());
                if (!maybe_target_type || !*maybe_target_type || !maybe_signature) {
                    VAST_PATTERN_FAIL("Failed to convert function type: {0}", fty);
                }

                auto signature = *maybe_signature;

                rewriter.modifyOpInPlace(op, [&]() {
                    op.setType(*maybe_target_type);
                    if (!op.getBlocks().empty()) {
                        tc::convert_region_types(op, signature);
                    }
                });

                return mlir::success();
            }

            static void legalize(parser_conversion_config &cfg) {
                cfg.target.addLegalOp< mlir::UnrealizedConversionCastOp >();
                cfg.target.addDynamicallyLegalOp< op_t >([](op_t op) {
                    auto fty = op.getFunctionType();
                    return all_of_parser_types(fty.getInputs())
                        && all_of_parser_types(fty.getResults());
                });
            }
        };

        struct AssignConversion
            : one_to_one_conversion_pattern< hl::AssignOp, pr::Assign >
        {
            using op_t = hl::AssignOp;
            using base = one_to_one_conversion_pattern< op_t, pr::Assign >;
            using base::base;

            using adaptor_t = typename op_t::Adaptor;

            logical_result matchAndRewrite(
                op_t op, adaptor_t adaptor, conversion_rewriter &rewriter
            ) const override {
                auto args = realized_operand_values(adaptor.getOperands(), rewriter);
                rewriter.create< pr::Assign >(op.getLoc(), std::vector< mlir_type >(), args);

                if (!op->getUsers().empty()) {
                    rewriter.replaceOp(op, args[0]);
                } else {
                    rewriter.eraseOp(op);
                }
                return mlir::success();
            }
        };

        struct CondYieldConversion
            : parser_conversion_pattern_base< hl::CondYieldOp >
        {
            using op_t = hl::CondYieldOp;
            using base = parser_conversion_pattern_base< op_t >;
            using base::base;

            using adaptor_t = typename op_t::Adaptor;

            logical_result matchAndRewrite(
                op_t op, adaptor_t adaptor, conversion_rewriter &rewriter
            ) const override {
                auto operand = adaptor.getResult();

                if (auto cast = mlir::dyn_cast< mlir::UnrealizedConversionCastOp >(operand.getDefiningOp())) {
                    if (pr::is_parser_type(cast.getOperand(0).getType())) {
                        rewriter.replaceOpWithNewOp< op_t >(op, cast.getOperand(0));
                        return mlir::success();
                    }
                }

                if (pr::is_parser_type(operand.getType())) {
                    rewriter.replaceOpWithNewOp< op_t >(op, operand);
                    return mlir::success();
                }

                auto cast = rewriter.create< mlir::UnrealizedConversionCastOp >(
                    op.getLoc(), pr::MaybeDataType::get(op.getContext()), operand
                );
                rewriter.replaceOpWithNewOp< op_t >(op, cast.getResult(0));
                return mlir::success();
            }

            static void legalize(parser_conversion_config &cfg) {
                cfg.target.addDynamicallyLegalOp< op_t >([](op_t op) {
                    return pr::is_parser_type(op.getResult().getType());
                });
            }
        };

        struct ValueYieldConversion : parser_conversion_pattern_base< hl::ValueYieldOp >
        {
            using op_t = hl::ValueYieldOp;
            using base = parser_conversion_pattern_base< op_t >;
            using base::base;

            using adaptor_t = typename op_t::Adaptor;

            logical_result matchAndRewrite(
                op_t op, adaptor_t adaptor, conversion_rewriter &rewriter
            ) const override {
                auto operand = adaptor.getResult();

                if (auto cast = mlir::dyn_cast< mlir::UnrealizedConversionCastOp >(
                        operand.getDefiningOp()
                    ))
                {
                    if (pr::is_parser_type(cast.getOperand(0).getType())) {
                        rewriter.replaceOpWithNewOp< op_t >(op, cast.getOperand(0));
                        return mlir::success();
                    }
                }

                if (pr::is_parser_type(operand.getType())) {
                    rewriter.replaceOpWithNewOp< op_t >(op, operand);
                    return mlir::success();
                }

                auto cast = rewriter.create< mlir::UnrealizedConversionCastOp >(
                    op.getLoc(), pr::MaybeDataType::get(op.getContext()), operand
                );
                rewriter.replaceOpWithNewOp< op_t >(op, cast.getResult(0));
                return mlir::success();
            }

            static void legalize(parser_conversion_config &cfg) {
                cfg.target.addDynamicallyLegalOp< op_t >([](op_t op) {
                    return pr::is_parser_type(op.getResult().getType());
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

        struct LazyConversion : parser_conversion_pattern_base< core::LazyOp >
        {
            using op_t = core::LazyOp;
            using base = parser_conversion_pattern_base< op_t >;
            using base::base;

            using adaptor_t = typename op_t::Adaptor;

            logical_result matchAndRewrite(
                op_t op, adaptor_t adaptor, conversion_rewriter &rewriter
            ) const override {
                auto &body = op.getLazy().front();
                if (body.empty()) {
                    return mlir::failure();
                }

                auto yield = terminator< hl::ValueYieldOp >::get(body);
                VAST_PATTERN_CHECK(yield, "Expected yield in: {0}", op);

                rewriter.inlineBlockBefore(&body, op);
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
                auto maybe = to_mlir_type(pr::data_type::maybedata, rewriter.getContext());
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
            // Cast operations
            ToMaybeParse< hl::ImplicitCastOp >, ToMaybeParse< hl::BuiltinBitCastOp >,
            ToMaybeParse< hl::CStyleCastOp >,
            // Comparison operations
            ToNoParse< hl::CmpOp >, ToNoParse< hl::FCmpOp >, ToMaybeParse< hl::Deref >,
            // Integer arithmetic
            ToMaybeParse< hl::AddIOp >, ToMaybeParse< hl::SubIOp >,
            ToMaybeParse< hl::PostIncOp >, ToMaybeParse< hl::PostDecOp >,
            ToMaybeParse< hl::PreIncOp >, ToMaybeParse< hl::PreDecOp >,
            // Logic operations
            ToMaybeParse< hl::BinXorOp >, ToMaybeParse< hl::BinAndOp >,
            ToMaybeParse< hl::BinOrOp >, ToMaybeParse< hl::LNotOp >,
            ToMaybeParse< core::BinLOrOp >, ToMaybeParse< core::BinLAndOp >,
            ToMaybeParse< hl::BinLOrOp >, ToMaybeParse< hl::BinLAndOp >,
            ToMaybeParse< hl::BinComma >,
            // Shift operations
            ToMaybeParse< hl::BinShlOp >, ToMaybeParse< hl::BinLShrOp >,
            ToMaybeParse< hl::BinAShrOp >,
            // Non-parsing integer arithmetic
            ToNoParse< hl::MulIOp >, ToNoParse< hl::DivSOp >, ToNoParse< hl::DivUOp >,
            ToNoParse< hl::RemSOp >, ToNoParse< hl::RemUOp >,
            // Floating point arithmetic
            ToNoParse< hl::AddFOp >, ToNoParse< hl::SubFOp >, ToNoParse< hl::MulFOp >,
            ToNoParse< hl::DivFOp >, ToNoParse< hl::RemFOp >, ToNoParse< hl::RealOp >,
            ToNoParse< hl::ImagOp >,
            // Arrays
            ToMaybeParse< hl::SubscriptOp >, ToMaybeParse< hl::AddressOf >,
            ToMaybeParse< hl::RecordMemberOp >,
            // Other operations
            ToMaybeParse< core::SelectOp >, AssignConversion, CondYieldConversion,
            ValueYieldConversion, ExprConversion, FuncConversion, ParamConversion,
            DeclRefConversion, VarDeclConversion, CallConversion, LazyConversion,
            // Return
            ReturnConversion< hl::ReturnOp >, ReturnConversion< core::ImplicitReturnOp > >;

    } // namespace pattern

    struct HLToParserPass : ConversionPassMixin< HLToParserPass, HLToParserBase >
    {
        using base = ConversionPassMixin< HLToParserPass, HLToParserBase >;

        struct server_handler
        {};

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

            if (!socket.empty()) {
                server = std::make_shared< vast::server::server< server_handler > >(
                    vast::server::sock_adapter::create_unix_socket(socket)
                );
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
            return { rewrite_pattern_set(&ctx), create_conversion_target(ctx), models,
                     server.get() };
        }

        function_models models;
        std::shared_ptr< vast::server::server< server_handler > > server;
    };

} // namespace vast::conv

std::unique_ptr< mlir::Pass > vast::createHLToParserPass() {
    return std::make_unique< vast::conv::HLToParserPass >();
}

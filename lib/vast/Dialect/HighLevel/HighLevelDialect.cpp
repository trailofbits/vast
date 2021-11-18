// Copyright (c) 2021-present, Trail of Bits, Inc.

#include "vast/Dialect/HighLevel/HighLevelDialect.hpp"
#include "vast/Dialect/HighLevel/HighLevelAttributes.hpp"
#include "vast/Dialect/HighLevel/HighLevelTypes.hpp"
#include "vast/Dialect/HighLevel/HighLevelOps.hpp"

#include "vast/Util/Functions.hpp"

#include <mlir/IR/TypeSupport.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/DialectImplementation.h>

#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/SMLoc.h>
#include <optional>
#include <type_traits>
#include <vector>


namespace vast::hl
{
    void HighLevelDialect::initialize()
    {
        registerTypes();
        registerAttributes();

        addOperations<
            #define GET_OP_LIST
            #include "vast/Dialect/HighLevel/HighLevel.cpp.inc"
        >();
    }

    using string_ref = llvm::StringRef;
    using DialectParser = mlir::DialectAsmParser;
    using DialectPrinter = mlir::DialectAsmPrinter;

    std::optional< Mnemonic > parse_mnemonic(DialectParser &parser)
    {
        auto loc = parser.getCurrentLocation();

        llvm::StringRef keyword;
        if (parser.parseKeyword(&keyword))
            return parser.emitError(loc, "missing type mnemonic"), std::nullopt;

        if (auto mnemonic = mnemonic_parser()(keyword))
            return result(mnemonic);

        return parser.emitError(loc, "invalid type mnemonic: ") << keyword, std::nullopt;
    }

    std::optional< Qualifier > parse_qualifier(DialectParser &parser)
    {
        auto loc = parser.getCurrentLocation();

        llvm::StringRef keyword;
        if (parser.parseKeyword(&keyword))
            return parser.emitError(loc, "missing type qualifier"), std::nullopt;

        if (auto qual = qualifier_parser()(keyword))
            return result(qual);
        return parser.emitError(loc, "invalid type qualifier: ") << keyword, std::nullopt;
    }

    std::vector< Qualifier > parse_qualifiers_list(DialectParser &parser)
    {
        std::vector< Qualifier > qualifiers;
        do {
            if (auto qual = parse_qualifier(parser))
                qualifiers.push_back(qual.value());
        } while ( failed(parser.parseOptionalGreater()) );
        return qualifiers;
    }

    std::vector< Qualifier > parse_qualifiers(DialectParser &parser)
    {
        if (succeeded(parser.parseOptionalLess()))
            return parse_qualifiers_list(parser);
        return {};
    }

    HighLevelType parse_pointer_type(Context *ctx, DialectParser &parser);

    HighLevelType parse_type(Context *ctx, DialectParser &parser)
    {
        if (auto mnem = parse_mnemonic(parser)) {
            return std::visit( overloaded {
                [&] (VoidMnemonic) -> HighLevelType { return VoidType::get(ctx); },
                [&] (BoolMnemonic) -> HighLevelType { return BoolType::get(ctx, parse_qualifiers(parser)); },
                [&] (IntegerKind kind)  -> HighLevelType { return IntegerType::get(ctx, kind, parse_qualifiers(parser)); },
                [&] (FloatingKind kind) -> HighLevelType { return FloatingType::get(ctx, kind, parse_qualifiers(parser)); },
                [&] (PointerMnemonic)   -> HighLevelType { return parse_pointer_type(ctx, parser); },
                [&] (RecordMnemonic)    -> HighLevelType { return RecordType::get(ctx); },
                [&] (ArrayMnemonic)     -> HighLevelType { return ArrayType::get(ctx); }
            }, mnem.value());
        }

        return HighLevelType();
    }

    HighLevelType parse_pointer_type(Context *ctx, DialectParser &parser)
    {
        HighLevelType result;

        auto fail = [&] (std::string_view msg) {
            auto loc = parser.getCurrentLocation();
            return parser.emitError(loc, msg), HighLevelType();
        };

        if (parser.parseLess()) {
            return HighLevelType();
        }

        auto element = parse_type(ctx, parser);
        if (!element) {
            return fail("wron pointer element type");
        }

        std::vector< Qualifier > qualifiers;
        if (succeeded(parser.parseOptionalComma())) {
            qualifiers = parse_qualifiers_list(parser);
            if (qualifiers.empty()) {
                return fail("expecting type qualifier list");
            }
        }

        result = PointerType::get(ctx, element, qualifiers);

        if (qualifiers.empty() && parser.parseGreater()) {
            return HighLevelType();
        }

        return result;
    }

    // Parse a type registered to this dialect.
    Type HighLevelDialect::parseType(DialectParser &parser) const
    {
        return parse_type(getContext(), parser);
    }

    void HighLevelDialect::printType(Type type, DialectPrinter &os) const
    {
        os << to_string(type.cast<HighLevelType>());
    }

} // namespace vast::hl

#include "vast/Dialect/HighLevel/HighLevelDialect.cpp.inc"

// Provide implementations for the enums we use.
#include "vast/Dialect/HighLevel/HighLevelEnums.cpp.inc"

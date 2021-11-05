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

    std::vector< Qualifier > parse_qualifiers(DialectParser &parser)
    {
        std::vector< Qualifier > qualifiers;

        if (succeeded(parser.parseOptionalLess())) {
            do {
                if (auto qual = parse_qualifier(parser))
                    qualifiers.push_back(qual.value());
            } while ( failed(parser.parseOptionalGreater()) );
        }

        return qualifiers;
    }

    Type build_type(Context *ctx, Mnemonic mnemonic)
    {
        return std::visit( overloaded {
            [&] (VoidMnemonic)      -> Type { return VoidType::get(ctx); },
            [&] (BoolMnemonic)      -> Type { return BoolType::get(ctx); },
            [&] (IntegerKind kind)  -> Type { return IntegerType::get(ctx, kind); },
            [&] (FloatingKind kind) -> Type { return FloatingType::get(ctx, kind); }
        }, mnemonic);
    }

    Type build_type(Context *ctx, Mnemonic mnemonic, QualifiersList qualifiers)
    {
        return std::visit( overloaded {
            [&] (VoidMnemonic)      -> Type { return llvm_unreachable("void cannot be qualified"), Type(); },
            [&] (BoolMnemonic)      -> Type { return BoolType::get(ctx, qualifiers); },
            [&] (IntegerKind kind)  -> Type { return IntegerType::get(ctx, kind, qualifiers); },
            [&] (FloatingKind kind) -> Type { return FloatingType::get(ctx, kind, qualifiers); }
        }, mnemonic);
    }

    // Parse a type registered to this dialect.
    Type HighLevelDialect::parseType(DialectParser &parser) const
    {
        if (auto mnem = parse_mnemonic(parser)) {
            auto ctx = getContext();
            if (auto quals = parse_qualifiers(parser); !quals.empty())
                return build_type(ctx, mnem.value(), quals);
            return build_type(ctx, mnem.value());
        }

        return Type();
    }

    void HighLevelDialect::printType(Type type, DialectPrinter &os) const
    {
        os << to_string(type.cast<HighLevelType>());
    }

} // namespace vast::hl

#include "vast/Dialect/HighLevel/HighLevelDialect.cpp.inc"

// Provide implementations for the enums we use.
#include "vast/Dialect/HighLevel/HighLevelEnums.cpp.inc"

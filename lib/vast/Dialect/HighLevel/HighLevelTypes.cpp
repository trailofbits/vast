// Copyright (c) 2021-present, Trail of Bits, Inc.

#include "vast/Dialect/HighLevel/HighLevelTypes.hpp"
#include "vast/Util/TypeList.hpp"
#include <sstream>

VAST_RELAX_WARNINGS
#include <llvm/ADT/TypeSwitch.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/DialectImplementation.h>
VAST_RELAX_WARNINGS

namespace vast::hl
{

  mlir::FunctionType getFunctionType(PointerType functionPointer)
  {
    if (functionPointer.isa< mlir::FunctionType > ()) {
        return functionPointer.cast< mlir::FunctionType >();
    } else {
        mlir::TypeRange inputs = {::vast::hl::VoidType::get(functionPointer.getContext())};
        mlir::TypeRange result = {::vast::hl::VoidType::get(functionPointer.getContext())};
        return mlir::FunctionType::get(functionPointer.getContext(), inputs, result);
    }
  }

    mlir::FunctionType getFunctionType(mlir::Type functionPointer)
    {
        if (functionPointer.isa< PointerType > ()) {
            return getFunctionType(functionPointer.cast< PointerType >());
        } else if (functionPointer.isa< mlir::FunctionType > ()) {
            return functionPointer.cast< mlir::FunctionType >();
        }
        mlir::TypeRange inputs = {::vast::hl::VoidType::get(functionPointer.getContext())};
        mlir::TypeRange result = {::vast::hl::VoidType::get(functionPointer.getContext())};
        return mlir::FunctionType::get(functionPointer.getContext(), inputs, result);
    }

    void HighLevelDialect::registerTypes() {
        addTypes<
            #define GET_TYPEDEF_LIST
            #include "vast/Dialect/HighLevel/HighLevelTypes.cpp.inc"
        >();
    }

    bool isBoolType(mlir::Type type)
    {
        return type.isa< BoolType >();
    }

    bool isIntegerType(mlir::Type type)
    {
        return util::is_one_of< integer_types >(type);
    }

    bool isFloatingType(mlir::Type type)
    {
        return util::is_one_of< floating_types >(type);
    }

    bool isSigned(mlir::Type type)
    {
        if (isBoolType(type)) {
            return false;
        }

        VAST_ASSERT(isIntegerType(type));
        return util::dispatch< integer_types, bool >(type, [] (auto ty) {
            return ty.isSigned();
        });
    }

    bool isUnsigned(mlir::Type type)
    {
        return !(isSigned(type));
    }

    bool isHighLevelType(mlir::Type type)
    {
        return util::is_one_of< high_level_types >(type);
    }

    template< typename CVType >
    Type parse_cv_type(Context *ctx, DialectParser &parser)
    {
        if (failed(parser.parseOptionalLess())) {
            return CVType::get(ctx);
        }

        bool c = succeeded(parser.parseOptionalKeyword("const"));
        bool v = succeeded(parser.parseOptionalKeyword("volatile"));

        if (failed(parser.parseGreater())) {
            return Type();
        }

        return CVType::get(ctx, c, v);
    }

    template< typename CVType >
    void print_cv_type(const CVType &type, DialectPrinter &printer)
    {
        printer << type.getMnemonic();

        if ( !(type.getIsVolatile() || type.getIsConst()) ) {
            return;
        }

        bool first = true;
        auto print = [&] (auto qual) {
            printer << (!first ? " " : "") << qual;
            first = false;
        };

        printer << "<";
        if (type.isConst())    { print("const"); }
        if (type.isVolatile()) { print("volatile"); }
        printer << ">";
    }

    template< typename IntegerType >
    Type parse_integer_type(Context *ctx, DialectParser &parser)
    {
        if (failed(parser.parseOptionalLess())) {
            return IntegerType::get(ctx);
        }

        bool u = succeeded(parser.parseOptionalKeyword("unsigned"));
        bool c = succeeded(parser.parseOptionalKeyword("const"));
        bool v = succeeded(parser.parseOptionalKeyword("volatile"));

        if (failed(parser.parseGreater())) {
            return Type();
        }

        return IntegerType::get(ctx, u, c, v);
    }

    template< typename IntegerType >
    void print_integer_type(const IntegerType &type, DialectPrinter &printer)
    {
        printer << type.getMnemonic();

        if ( !(type.getIsVolatile() || type.getIsConst() || type.getIsUnsigned()) ) {
            return;
        }

        bool first = true;
        auto print = [&] (auto qual) {
            printer << (!first ? " " : "") << qual;
            first = false;
        };

        printer << "<";
        if (type.isUnsigned()) { print("unsigned"); }
        if (type.isConst())    { print("const"); }
        if (type.isVolatile()) { print("volatile"); }
        printer << ">";
    }

    Type parse_pointer_type(Context *ctx, DialectParser &parser) {
        if (failed(parser.parseLess())) {
            return Type();
        }

        Type element;
        if (failed(parser.parseType(element))) {
            auto loc = parser.getCurrentLocation();
            parser.emitError(loc, "expected pointer element type");
            return Type();
        }

        bool c = false, v = false;
        if (succeeded(parser.parseOptionalComma())) {
            c = succeeded(parser.parseOptionalKeyword("const"));
            v = succeeded(parser.parseOptionalKeyword("volatile"));
        }

        if (failed(parser.parseGreater())) {
            return Type();
        }

        return PointerType::get(ctx, element, c, v);
    }

    void print_pointer_type(const PointerType &type, DialectPrinter &printer)
    {
        printer << type.getMnemonic() << "<";
        printer.printType(type.getElementType());

        if ( type.getIsVolatile() || type.getIsConst() ) {
            printer << ",";
            if (type.isConst())    { printer << " const"; }
            if (type.isVolatile()) { printer << " volatile"; }
        }

        printer << ">";
    }

    Type ConstantArrayType::parse(Context *ctx, DialectParser &parser) {
        if (failed(parser.parseLess())) {
            return Type();
        }

        llvm::APInt size;
        if (failed(parser.parseInteger(size))) {
            auto loc = parser.getCurrentLocation();
            parser.emitError(loc, "expected array size");
            return Type();
        }

        if (parser.parseComma()) {
            auto loc = parser.getCurrentLocation();
            parser.emitError(loc, "expected comma after size");
            return Type();
        }

        Type element;
        if (failed(parser.parseType(element))) {
            auto loc = parser.getCurrentLocation();
            parser.emitError(loc, "expected array element type");
            return Type();
        }

        bool constness = false, volatility = false;
        if (succeeded(parser.parseOptionalComma())) {
            constness = succeeded(parser.parseOptionalKeyword("const"));
            volatility = succeeded(parser.parseOptionalKeyword("volatile"));
        }

        if (failed(parser.parseGreater())) {
            return Type();
        }

        return ConstantArrayType::get(ctx, element, size, constness, volatility);
    }

    void ConstantArrayType::print(DialectPrinter &printer) const
    {
        printer << getMnemonic() << "<" << getSize() << ", ";
        printer.printType(getElementType());

        if ( getIsVolatile() || getIsConst() ) {
            printer << ",";
            if (isConst())    { printer << " const"; }
            if (isVolatile()) { printer << " volatile"; }
        }

        printer << ">";
    }

} // namespace vast::hl


#define  GET_TYPEDEF_CLASSES
#include "vast/Dialect/HighLevel/HighLevelTypes.cpp.inc"

namespace vast::hl
{
    Type HighLevelDialect::parseType(DialectParser &parser) const
    {
        auto loc = parser.getCurrentLocation();
        llvm::StringRef mnemonic;
        if (parser.parseKeyword(&mnemonic))
            return Type();

        Type result;
        if (generatedTypeParser(getContext(), parser, mnemonic, result).hasValue()) {
            return result;
        }

        parser.emitError(loc, "unknown high-level type");
        return Type();
    }

    void HighLevelDialect::printType(Type type, DialectPrinter &os) const
    {
        if (failed(generatedTypePrinter(type, os)))
            VAST_UNREACHABLE("unexpected high-level type kind");
    }

    template< typename T >
    using walk_fn = llvm::function_ref< void( T ) >;

    using walk_types = walk_fn< mlir::Type >;
    using walk_attrs = walk_fn< mlir::Attribute >;

    void PointerType::walkImmediateSubElements(walk_attrs, walk_types tys) const {
        tys( this->getElementType() );
    }

    void ConstantArrayType::walkImmediateSubElements(walk_attrs, walk_types tys) const {
        tys( this->getElementType() );
    }

    // TODO(lukas): Generalize and pull into header as it will probably be needed
    //              for all other array types as well.
    auto ConstantArrayType::dim_and_type() -> std::tuple< dimensions_t, mlir::Type >
    {
        dimensions_t out;
        // If this ever is generalised investigate if `SubElementTypeInterface` can be used
        // do this recursion?
        auto collect = [&](ConstantArrayType t, auto &fwd) -> mlir::Type {
            out.push_back(t.getNumElems());
            if (auto c_array = t.getElementType().dyn_cast< ConstantArrayType >())
                return fwd(c_array, fwd);
            return t.getElementType();
        };
        return { std::move(out), collect(*this, collect) };
    }

    Type NamedType::parse(Context *ctx, DialectParser &parser) {
        if (failed(parser.parseLess())) {
            return Type();
        }

        mlir::SymbolRefAttr name;
        // TODO(Heno): use parseSymbolName from MLIR 14
        if (failed(parser.parseAttribute(name))) {
            auto loc = parser.getCurrentLocation();
            parser.emitError(loc, "expected type name symbol");
            return Type();
        }

        if (failed(parser.parseGreater())) {
            return Type();
        }

        return NamedType::get(ctx, name);
    }

    void NamedType::print(DialectPrinter &printer) const {
        printer << getMnemonic() << "<" << getName() << ">";
    }

} // namespace vast::hl

// Copyright (c) 2023, Trail of Bits, Inc.

#pragma once

namespace vast {

    template <typename Ty> struct EnumTraits {};

    #define VAST_REGISTER_ENUM_TYPE(Dialect, Ty) \
    template<> struct EnumTraits< Dialect::Ty > { \
        static llvm::StringRef stringify(Dialect::Ty value) { return stringify##Ty(value); } \
        static unsigned getMaxEnumVal() { return Dialect::getMaxEnumValFor##Ty(); } \
    }

    // Parses one of the keywords provided in the list `keywords` and returns the
    // position of the parsed keyword in the list. If none of the keywords from the
    // list is parsed, returns -1.
    static int parse_optional_keyword_alternative(
        mlir::AsmParser &parser, llvm::ArrayRef< llvm::StringRef > keywords
    ) {
        for (auto en : llvm::enumerate(keywords)) {
            if (mlir::succeeded(parser.parseOptionalKeyword(en.value()))) {
                return en.index();
            }
        }
        return -1;
    }

    /// Parse an enum from the keyword, or default to the provided default value.
    /// The return type is the enum type by default, unless overriden with the
    /// second template argument.
    template< typename EnumTy, typename RetTy = EnumTy >
    static RetTy parse_optional_vast_keyword(mlir::AsmParser &parser, EnumTy default_value) {
        llvm::SmallVector< llvm::StringRef, 10 > names;
        for (unsigned i = 0, e = EnumTraits< EnumTy >::getMaxEnumVal(); i <= e; ++i) {
            names.push_back(EnumTraits< EnumTy >::stringify(static_cast< EnumTy >(i)));
        }

        int index = parse_optional_keyword_alternative(parser, names);
        if (index == -1) {
            return static_cast< RetTy >(default_value);
        }
        return static_cast< RetTy >(index);
    }

} // namespace vast

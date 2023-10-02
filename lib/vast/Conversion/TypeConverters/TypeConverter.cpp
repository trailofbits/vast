// Copyright (c) 2023, Trail of Bits, Inc.

#include "vast/Conversion/TypeConverters/TypeConverter.hpp"

namespace vast::conv::tc
{
    bool base_type_converter::isSignatureLegal(core::FunctionType ty)
    {
        return base::isLegal(llvm::concat<const mlir_type>(ty.getInputs(), ty.getResults()));
    }

} // namespace vast::tc

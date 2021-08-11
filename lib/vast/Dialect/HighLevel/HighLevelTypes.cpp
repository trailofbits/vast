// Copyright (c) 2021-present, Trail of Bits, Inc.

#include "vast/Dialect/HighLevel/HighLevelTypes.hpp"

#include <mlir/IR/TypeSupport.h>
#include <llvm/ADT/Hashing.h>

#include <tuple>

namespace vast::hl
{
    namespace detail
    {
        using storage_allocator = mlir::TypeStorageAllocator;

        struct integer_type_storage
        {
            using KeyTy = std::tuple< integer_qualifier, integer_kind >;

            integer_type_storage(const KeyTy& key) : fields(key) {}

            bool operator==(const KeyTy &key) const { return fields == key; }

            static integer_type_storage* construct(storage_allocator &allocator, KeyTy &key)
            {
                return new (allocator.allocate< integer_type_storage >()) integer_type_storage(key);
            }

            static llvm::hash_code hashKey(const KeyTy& key)
            {
                return llvm::hash_combine(key);
            }

            KeyTy fields;
        };
    } // namespace detail

    void_type void_type::get(context *ctx) { return Base::get(ctx); }

    // integer_type integer_type::get(context *ctx, integer_qualifier qual, integer_kind kind)
    // {
    //     return Base::get(ctx, qual, kind);
    // }

} // namespace vast::hl
#ifndef CORE_TRAITS_H
#define CORE_TRAITS_H

#include "mlir/IR/Types.h"

namespace mlir {
namespace TypeTrait {

#define CORE_TRAIT( trait ) template < typename type > \
class trait : public TypeTrait::TraitBase< type, trait > {};

CORE_TRAIT(IntegralTypeTrait);
CORE_TRAIT(CharTypeTrait);
CORE_TRAIT(ShortTypeTrait);
CORE_TRAIT(IntegerTypeTrait);
CORE_TRAIT(LongTypeTrait);
CORE_TRAIT(LongLongTypeTrait);
CORE_TRAIT(Int128TypeTrait);

CORE_TRAIT(PointerTypeTrait);
} // namespace TypeTrait
} // namespace mlir
#endif //CORE_TRAITS_H

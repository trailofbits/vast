#ifndef CORE_TRAITS_H
#define CORE_TRAITS_H

#include "mlir/IR/Types.h"

namespace vast::core {

#define CORE_TRAIT( trait ) template < typename type > \
class trait : public mlir::TypeTrait::TraitBase< type, trait > {};

CORE_TRAIT(IntegralTypeTrait);
CORE_TRAIT(BoolTypeTrait);
CORE_TRAIT(CharTypeTrait);
CORE_TRAIT(ShortTypeTrait);
CORE_TRAIT(IntegerTypeTrait);
CORE_TRAIT(LongTypeTrait);
CORE_TRAIT(LongLongTypeTrait);
CORE_TRAIT(Int128TypeTrait);

CORE_TRAIT(FloatingTypeTrait);

CORE_TRAIT(PointerTypeTrait);

CORE_TRAIT(TypedefTrait);

CORE_TRAIT(TypeOfTrait);

CORE_TRAIT(VoidTrait);

} // namespace vast::core
#endif // CORE_TRAITS_H

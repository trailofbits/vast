// Copyright (c) 2024, Trail of Bits, Inc.

#pragma once

namespace vast::util
{
    template< typename attribute_type >
    attribute_type get_attr(auto op) {
        return op->template getAttrOfType< attribute_type >(attribute_type::getMnemonic());
    }

    template< typename attribute_type >
    bool has_attr(auto op) {
        return op->hasAttr(attribute_type::getMnemonic());
    }

} // namespace vast::util

#pragma once

namespace vast::analyses {

    class CFGBlock_T {
    public:
        unsigned BlockID;

        unsigned getBlockID() const {
            return BlockID;
        }

        explicit CFGBlock_T(unsigned blockid) : BlockID(blockid) {}
    };

} // namespace vast::analyses

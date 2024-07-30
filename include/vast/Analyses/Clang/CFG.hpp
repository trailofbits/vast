#pragma once

namespace vast::analyses {

    class CFGBlockT {
    public:
        unsigned BlockID;

        unsigned getBlockID() const {
            return BlockID;
        }

        explicit CFGBlockT(unsigned blockid) : BlockID(blockid) {}
    };

} // namespace vast::analyses

#pragma once

namespace vast::analysis {

    class CFGBlock {
    public:
        unsigned BlockID;

        unsigned getBlockID() const {
            return BlockID;
        }

        explicit CFGBlock(unsigned blockid) : BlockID(blockid) {}
    };

} // namespace vast::analysis

#include "vast/Frontend/Targets.hpp"

namespace vast::cc {
    target_dialect parse_target_dialect(string_ref from) {
        auto trg = from.lower();
        if (trg == "hl" || trg == "high_level") {
            return target_dialect::high_level;
        }
        if (trg == "std") {
            return target_dialect::std;
        }
        if (trg == "llvm") {
            return target_dialect::llvm;
        }
        if (trg == "cir") {
            return target_dialect::cir;
        }
        VAST_UNREACHABLE("Unknown option of target dialect: {0}", trg);
    }

    std::string to_string(target_dialect target) {
        switch (target) {
            case target_dialect::high_level:
                return "high_level";
            case target_dialect::std:
                return "std";
            case target_dialect::llvm:
                return "llvm";
            case target_dialect::cir:
                return "cir";
        }
    }

} // namespace vast::cc

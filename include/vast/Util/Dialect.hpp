// Copyright (c) 2023-present, Trail of Bits, Inc.

#define GRAPH_REGION_OP( op ) mlir::RegionKind op::getRegionKind(unsigned index) { \
        return mlir::RegionKind::Graph; \
    }

#define SSACFG_REGION_OP( op ) mlir::RegionKind op::getRegionKind(unsigned index) { \
        return mlir::RegionKind::SSACFG; \
    }

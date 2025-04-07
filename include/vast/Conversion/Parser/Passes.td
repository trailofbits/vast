#ifndef VAST_CONVERSION_PARSER_PASSES_TD
#define VAST_CONVERSION_PARSER_PASSES_TD

include "mlir/Pass/PassBase.td"

def HLToParser : Pass<"vast-hl-to-parser", "core::ModuleOp"> {
    let summary = "Transform hl to parser dialect.";
    let description = [{ WIP }];

    let options = [
        Option< "config", "config", "std::string", "",
            "Configuration file for parser transformation."
        >,
        Option< "socket", "socket", "std::string", "",
            "Unix socket path to use for server."
        >,
        Option< "yaml_out", "yaml-out", "std::string", "",
            "Path to YAML output file for models got from user."
        >
    ];

    let constructor = "vast::createHLToParserPass()";
    let dependentDialects = [
        "vast::pr::ParserDialect"
    ];
}

def ParserReconcileCasts : Pass<"vast-parser-reconcile-casts", "core::ModuleOp"> {
    let summary = "Reconcile casts in parser dialect.";
    let description = [{ WIP }];

    let constructor = "vast::createParserReconcileCastsPass()";
    let dependentDialects = [
        "vast::pr::ParserDialect"
    ];
}

def ParserRefine : Pass<"vast-parser-refine", "core::ModuleOp"> {
    let summary = "Refine parser dialect.";
    let description = [{ WIP }];

    let constructor = "vast::createParserRefinePass()";
    let dependentDialects = [
        "vast::pr::ParserDialect"
    ];
}

def ParserRefineCleanUp : Pass<"vast-parser-refine-cleanup", "core::ModuleOp"> {
    let summary = "Refine parser dialect.";
    let description = [{ WIP }];

    let constructor = "vast::createParserRefineCleanUpPass()";
    let dependentDialects = [
        "vast::pr::ParserDialect"
    ];
}

def PruneDeadCode : Pass<"vast-prune-dead-code", "core::ModuleOp"> {
    let summary = "Prune dead code for parser analysis.";
    let description = [{ WIP }];

    let constructor = "vast::createPruneDeadCodePass()";
    let dependentDialects = [
        "vast::pr::ParserDialect"
    ];
}

#endif // VAST_CONVERSION_PARSER_PASSES_TD

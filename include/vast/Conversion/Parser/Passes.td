#ifndef VAST_CONVERSION_PARSER_PASSES_TD
#define VAST_CONVERSION_PARSER_PASSES_TD

include "mlir/Pass/PassBase.td"

def HLToParser : Pass<"vast-hl-to-parser", "core::ModuleOp"> {
    let summary = "Transform hl to parser dialect.";
    let description = [{ WIP }];

    let options = [
        Option< "config", "config", "std::string", "",
            "Configuration file for parser transformation."
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

#endif // VAST_CONVERSION_PARSER_PASSES_TD

add_vast_executable(vast-detect-parsers
    main.cpp
    ParserSourceDetector.cpp

    LINK_LIBS
        MLIROptLib
)

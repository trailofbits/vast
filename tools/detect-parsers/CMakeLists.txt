add_vast_executable(detect-parsers
    main.cpp
    ParserSourceDetector.cpp

    LINK_LIBS
        MLIROptLib
)

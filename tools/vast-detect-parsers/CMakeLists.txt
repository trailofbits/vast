add_vast_executable(vast-detect-parsers
    main.cpp
    ParserCategoryDetector.cpp

    LINK_LIBS
        MLIROptLib
)

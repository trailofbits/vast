include(CTest)

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(VAST_TEST_DEPENDS
  vast-query
  vast-opt
  vast-front
  vast-detect-parsers
)

add_lit_testsuite(check-vast "Running the VAST regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${VAST_TEST_DEPENDS}
)

set_target_properties(check-vast PROPERTIES FOLDER "Tests")

add_lit_testsuites(VAST ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS ${VAST_TEST_DEPENDS})
add_test(NAME lit
         COMMAND lit -v "${CMAKE_CURRENT_BINARY_DIR}"
         --param BUILD_TYPE=$<CONFIG>)

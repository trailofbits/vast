# -*- Python -*-

import os
import platform
import re
import subprocess
import tempfile

import lit.formats
import lit.util

from lit.llvm import llvm_config
from lit.llvm.subst import ToolSubst
from lit.llvm.subst import FindTool

# Configuration file for the 'lit' test runner.

# name: The name of this test suite.
config.name = 'VAST'

config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)

# suffixes: A list of file extensions to treat as test files.
config.suffixes = ['.mlir', '.c', '.cpp', '.ll']

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.vast_obj_root, 'test')

config.substitutions.append(('%PATH%', config.environment['PATH']))
config.substitutions.append(('%shlibext', config.llvm_shlib_ext))

llvm_config.with_system_environment(
    ['HOME', 'INCLUDE', 'LIB', 'TMP', 'TEMP'])

# excludes: A list of directories to exclude from the testsuite. The 'Inputs'
# subdirectories contain auxiliary inputs for various tests in their parent
# directories.
config.excludes = ['Inputs', 'Examples', 'CMakeLists.txt', 'README.txt', 'LICENSE.txt']

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.vast_obj_root, 'test')
config.vast_test_util = os.path.join(config.vast_src_root, 'test/utils')
config.vast_tools_dir = os.path.join(config.vast_obj_root, 'tools')

tools = [
    ToolSubst('vast-opt', command = 'vast-opt'),
    ToolSubst('vast-cc', command = 'vast-cc'),
    ToolSubst('vast-query', command = 'vast-query'),
    ToolSubst('vast-front', command = 'vast-front'),
    ToolSubst('vast-repl', command = 'vast-repl'),
    ToolSubst('%vast-cc1', command = 'vast-front',
        extra_args=[
            "-cc1",
            "-internal-isystem",
            "-nostdsysteminc"
        ]
    ) ]



if 'BUILD_TYPE' in lit_config.params:
    config.vast_build_type = lit_config.params['BUILD_TYPE']
else:
    config.vast_build_type = "Debug"

for tool in tools:
    path = [config.vast_tools_dir, tool.command, config.vast_build_type]
    llvm_config.add_tool_substitutions([tool], os.path.join(*path))

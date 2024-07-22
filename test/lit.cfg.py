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
    ToolSubst('%vast-opt', command = 'vast-opt'),
    ToolSubst('%vast-opt-irs-to-llvm', command = 'vast-opt',
        extra_args=[
            "--vast-hl-lower-elaborated-types",
            "--vast-hl-lower-typedefs",
            "--vast-hl-lower-types",
            "--vast-hl-to-ll-func",
            "--vast-hl-to-ll-vars",
            "--vast-hl-to-ll-cf",
            "--vast-hl-to-ll-geps",
            "--vast-fn-args-to-alloca",
            "--vast-lower-value-categories",
            "--vast-hl-to-lazy-regions",
            "--vast-irs-to-llvm",
            "--vast-core-to-llvm",
        ]
    ),
    ToolSubst('%vast-opt-lower-value-categories', command = 'vast-opt',
        extra_args=[
            "--vast-hl-lower-elaborated-types",
            "--vast-hl-lower-typedefs",
            "--vast-hl-lower-types",
            "--vast-hl-to-ll-func",
            "--vast-hl-to-ll-vars",
            "--vast-hl-to-ll-cf",
            "--vast-hl-to-ll-geps",
            "--vast-fn-args-to-alloca",
            "--vast-lower-value-categories",
        ]
    ),
    ToolSubst('%vast-opt-core-to-llvm', command = 'vast-opt',
        extra_args=[
            "--vast-hl-lower-types",
            "--vast-hl-to-ll-cf",
            "--vast-hl-to-ll-vars",
            "--vast-hl-lower-elaborated-types",
            "--vast-hl-lower-typedefs",
            "--vast-hl-to-lazy-regions",
            "--vast-irs-to-llvm",
            "--vast-core-to-llvm"
        ]
    ),
    ToolSubst('%vast-cc', command = 'vast-cc'),
    ToolSubst('%vast-query', command = 'vast-query'),
    ToolSubst('%vast-front', command = 'vast-front'),
    ToolSubst('%vast-repl', command = 'vast-repl'),
    ToolSubst('%vast-cc1', command = 'vast-front',
        extra_args=[
            "-cc1",
            "-internal-isystem",
            "-nostdsysteminc"
        ]
    ),
    ToolSubst('%file-check', command = 'FileCheck'),
    ToolSubst('%cc', command = 'cc')
]

passes = [
      "vast-hl-splice-trailing-scopes"
    , "vast-hl-to-hl-builtin"
    , "vast-hl-ude"
    , "vast-hl-dce"
    , "vast-hl-lower-elaborated-types"
    , "vast-hl-lower-typedefs"
    , "vast-hl-lower-enums"
    , "vast-hl-lower-types"
    , "vast-hl-to-ll-func"
    , "vast-hl-to-ll-vars"
    , "vast-hl-to-ll-cf"
    , "vast-hl-to-ll-geps"
    , "vast-fn-args-to-alloca"
    , "vast-lower-value-categories"
    , "vast-hl-to-lazy-regions"
    , "vast-emit-abi"
    , "vast-lower-abi"
    , "vast-irs-to-llvm"
    , "vast-core-to-llvm"
]

for p in passes:
    name = "%check-" + p[len("vast-"):]
    tools.append(ToolSubst(name, command = 'vast-front',
                           extra_args = ['-vast-emit-mlir-after=' + p, '-o', '-']))

if 'BUILD_TYPE' in lit_config.params:
    config.vast_build_type = lit_config.params['BUILD_TYPE']
else:
    config.vast_build_type = "Debug"

for tool in tools:
    if tool.command.startswith('vast'):
        path = [config.vast_tools_dir, tool.command, config.vast_build_type]
        tool.command = os.path.join(*path, tool.command)
    llvm_config.add_tool_substitutions([tool])

stdbit_test = subprocess.run(["cc", "-x", "c", "-", "-o", "/dev/null"],
                             input=b'#include <stdbit.h>\n int main() {}')
if stdbit_test.returncode == 0:
    config.available_features.add("stdbit")

uchar_input = b'''
#include <uchar.h>

int main() {
    const char *mbstr = "aa";
    char8_t c8;
    mbstate_t state;
    size_t x = mbrtoc8(&c8, mbstr, 1, &state);
}
'''
uchar_test = subprocess.run(["cc", "-std=c2x", "-x", "c", "-", "-o", "/dev/null"], input=uchar_input)
if uchar_test.returncode == 0:
    config.available_features.add("ucharc23")

miamcu_test = subprocess.run(["vast-front", "-vast-emit-mlir=hl", "-x", "c", "-", "-o", "/dev/null", "-m32", "-miamcu"],
                             input=b'int main() {}')
if miamcu_test.returncode == 0:
    config.available_features.add("miamcu")

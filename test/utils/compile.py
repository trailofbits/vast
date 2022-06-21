# Copyright (c) 2022 Trail of Bits, Inc.

import argparse
import os
import subprocess
import tempfile
import sys

# Configuration of test framework adds them to path.
vast_cc = 'vast-cc'
vast_opt = 'vast-opt'

class PipelineFail(Exception):
    def __init__(self, component, out='', err='', cause=None):
        self._out = out
        self._err = err
        self._component = component
        self._cause = cause

    def fmt_streams(self):
        msg = 'Stdout:\n' + self._out + '\n'
        msg += '\n ----- \n'
        msg += 'Stderr:\n' + self._err + '\n'
        return msg

    def __str__(self):
        msg = 'Pipeline fail in ' + self._component + '!\n'
        if self._cause is not None:
            msg += 'Cause: ' + self._cause + '\n'
        msg += self.fmt_streams()
        return msg

class TimeoutFail(PipelineFail):
    def __init__(self, component, out='', err=''):
        super().__init__(component, out, err, 'Timeout')

def check_retcode(component, pipes):
    try:
        out, err = pipes.communicate(timeout = 45)
    except subprocess.TimeoutExpired:
        pipes.kill()
        out, err = pipes.communicate()
        raise TimeoutFail(component, out, err)
    ret_code = pipes.returncode
    if ret_code != 0:
        raise PipelineFail(component, out, err, 'Ret code is ' + str(ret_code) + ' expected 0')

# Returns mlir file that corresponds to lowered `source`.
def to_vast_mlir(lang, source):
    mlir_file =  source + '.vast.mlir'

    args = [vast_cc]
    if lang == 'c':
        args += ['--ccopts', '-xc']
    args += ['--from-source', source]
    args += ['-o=' + mlir_file]

    pipes = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    check_retcode('to-vast-mlir', pipes)

    return mlir_file

# Returns (output file with bitcode, all passes required to get there)
def get_passes(mlir_file):
    bc_file = mlir_file + '.ll'

    passes = [
        '--vast-hl-lower-types',
        '--vast-hl-structs-to-tuples',
        '--vast-hl-to-scf',
        '--convert-scf-to-std',
        '--vast-hl-to-ll',
        '--vast-llvm-dump=bc-file=' + bc_file
    ]

    return (bc_file, passes)

def to_llvm_ir(mlir_file):
    bc_file, passes = get_passes(mlir_file)
    args = [vast_opt]
    args += passes
    args += [mlir_file]

    pipes = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    check_retcode('vast-opt', pipes)

    return bc_file

def compile_vast(args):
    mlir_file = to_vast_mlir(args.lang, args.source)
    llvm_ir_file = to_llvm_ir(mlir_file)

    args = ['clang', llvm_ir_file, '-o' , args.vast_out]

    pipes = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    check_retcode('vast-clang-recompile', pipes)

def compile_clang(args):
    args = ['clang', args.source, '-o', args.clang_out]
    pipes = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    check_retcode('vast-clang-recompile', pipes)

def main():
    main_desc = """Wrapper script that compiles programs using vast-cc and vast-opt before clang. It is expected vast-cc and vast-opt are in PATH."""

    args_p = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description=main_desc)
    args_p.add_argument('--lang',
                        help='Language, currently supported: [ c, cpp ]',
                        choices=['c', 'cpp'],
                        default='c')
    args_p.add_argument('--source',
                        help='Source file to be compiled by vast.' \
                             'Currently only one is supported',
                        required=True)
    args_p.add_argument('--link-with',
                        help='Additional files to be forwarded to compiler',
                        action='extend',
                        nargs='+')
    args_p.add_argument('--vast-out',
                        help='Name of the executable compiled using vast',
                        required=True)
    args_p.add_argument('--clang-out',
                        help='Replicate same build but with clang')

    # Extra arguments are currently not used
    args, _ = args_p.parse_known_args()

    if args.vast_out:
        compile_vast(args)
    if args.clang_out:
        compile_clang(args)

    return 0

if __name__ == "__main__":
    return_code = main()
    sys.exit(return_code)

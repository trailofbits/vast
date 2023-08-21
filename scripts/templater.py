#!/usr/bin/env python3

from PyInquirer import prompt
from examples import custom_style_2
from prompt_toolkit.validation import Validator, ValidationError
from typing import Dict, Any

import jinja2

import datetime
import sys
import os

#
# Setup Jinja
#

script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
templates_dir = os.path.join(script_dir, 'templates')

project_dir = os.path.dirname(script_dir)

template_loader = jinja2.FileSystemLoader(searchpath=templates_dir)
templates = jinja2.Environment(loader=template_loader)

#
# Dialect Generation
#

def append_and_sort(line: str, file: str):
    with open(file, 'r') as cmake:
        lines = cmake.readlines()

    license = lines[:2] # license lines

    entries = lines[2:]
    if line in entries:
        return

    entries.append(line)
    entries.sort()

    with open(file, "w") as file:
        file.writelines(license)
        file.writelines(entries)

def generate_dialect_includes(opts):
    dialect = opts['dialect_name']
    includes = os.path.join(project_dir, f"include/vast/Dialect/")
    dialect_includes = os.path.join(includes, dialect)

    if not proceed_query(f"Generate dialect include files into: { dialect_includes }"):
        return

    if os.path.exists(dialect_includes):
        if not proceed_query(f"Dialect already exists. Do you want to overwrite its file?"):
            exit()

    # Create Includes Directory
    os.makedirs(dialect_includes, exist_ok=True)

    def create_in(root: str, dst: str, template_name: str):
        template = templates.get_template(template_name)
        destination = os.path.join(root, dst)
        template.stream(opts).dump(destination)
        print(f"Creating: { destination }")

    def create_in_includes(dst: str, template_name: str):
        create_in(dialect_includes, dst, template_name)

    # Generate dialect includes CMakeLists
    create_in_includes('CMakeLists.txt', 'dialect.includes.cmake.in')

    # Register dialect subdirectory in cmake
    includes_cmake_lists = os.path.join(includes, 'CMakeLists.txt')
    append_and_sort(f"add_subdirectory({ dialect })\n", includes_cmake_lists)
    print(f"Updating: { includes_cmake_lists }")

    # Generate headers
    create_in_includes(f'{ dialect }.td', 'Dialect.td.in')
    create_in_includes(f'{ dialect }Dialect.hpp', 'Dialect.hpp.in')
    create_in_includes(f'{ dialect }Ops.td', 'Ops.td.in')
    create_in_includes(f'{ dialect }Ops.hpp', 'Ops.hpp.in')

    if opts['has_types']:
        create_in_includes(f'{ dialect }Types.td', 'Types.td.in')
        create_in_includes(f'{ dialect }Types.hpp', 'Types.hpp.in')

    if opts['has_attributes']:
        create_in_includes(f'{ dialect }Attributes.td', 'Attributes.td.in')
        create_in_includes(f'{ dialect }Attributes.hpp', 'Attributes.hpp.in')

    if opts['has_internal_transforms']:
        create_in_includes(f'Passes.td', 'Passes.td.in')
        create_in_includes(f'Passes.hpp', 'Passes.hpp.in')

    # register dialect in Dialects.hpp

    # TODO dump info about generated files

def generate_dialect_templates(opts):
    # Generate include templates

    if opts['has_internal_transforms']:
        opts['internal_transforms'] = '_and_passes'

    generate_dialect_includes(opts)

dialect_config = {
    'generator': generate_dialect_templates,
    'option_name': 'dialect',
    'help': 'TODO',
    'options': [
        {
            'type': 'input',
            'name': 'dialect_name',
            'message': 'What is dialect name?',
        },
        {
            'type': 'input',
            'name': 'dialect_mnemonic',
            'message': 'What is dialect mnemonic?',
        },
        {
            'type': 'input',
            'name': 'dialect_namespace',
            'message': 'What is dialect C++ namespace?',
        },
        {
            'type': 'confirm',
            'name': 'has_types',
            'message': 'Does dialect provide types?',
        },
        {
            'type': 'confirm',
            'name': 'has_attributes',
            'message': 'Does dialect provide attributes?',
        },
        {
            'type': 'confirm',
            'name': 'has_internal_transforms',
            'message': 'Does dialect provide internal transformations?',
        },
    ]
}

#
# Conversion Generation
#

def generate_conversion_templates(opts):
    pass

conversion_config = {
    'generator': generate_conversion_templates,
    'option_name': 'conversion',
    'help': 'TODO',
    'options': [
        {
            'type': 'confirm',
            'name': 'endomorphism',
            'message': 'Is this endomorphic conversion in dialect?',
        }
    ]
}

#
# Operation Generation
#

def generate_operation_templates(opts):
    pass

operation_config = {
    'generator': generate_operation_templates,
    'option_name': 'operation',
    'help': 'TODO',
    'options': [
        {
            'type': 'confirm',
            'name': 'endomorphism',
            'message': 'Is this endomorphic conversion in dialect?',
        }
    ]
}

def generate_type_templates(opts):
    pass

type_config = {
    'generator': generate_type_templates,
    'option_name': 'type',
    'help': 'TODO',
    'options': [
    ]
}

#
# Config
#

configs = [
    dialect_config, conversion_config, operation_config, type_config
]

prologue = [
    {
        'type': 'list',
        'name': 'generate',
        'message': 'Welcome to VAST libraries template generator. Choose template:',
        'choices': [config['option_name'] for config in configs]
    }
]

#
# Utils
#

def query(opts : str) -> Dict[str, Any]:
    return prompt(opts, style=custom_style_2)

def proceed_query(msg: str) -> bool:
    return query({
        'type': 'confirm',
        'name': 'proceed',
        'message': msg
    })['proceed']


def fill_template_options(opts, target):
    opts['year'] = str(datetime.date.today().year)

def get_target_config(target):
    for config in configs:
        if config['option_name'] == target:
            return config

def main():
    # query for target templates to be generated
    target = query(prologue)
    config = get_target_config(target['generate'])
    # query for options of desired target templates
    opts = query(config['options'])
    # fill in deduced target options
    fill_template_options(opts, target)
    # generate demplates from gathered options
    config['generator'](opts)

if __name__ == "__main__":
    main()

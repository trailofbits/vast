#!/usr/bin/env python3

from PyInquirer import prompt
from examples import custom_style_2
from prompt_toolkit.validation import Validator, ValidationError
from typing import Dict, Any

import jinja2

import datetime
import sys
import os
import re

#
# Setup Jinja
#

script_dir  = os.path.dirname(os.path.abspath(sys.argv[0]))
project_dir = os.path.dirname(script_dir)
dialects_includes = os.path.join(project_dir, f"include/vast/Dialect/")

def gather_dialects(includes: str):
    isdir = lambda path: os.path.isdir(os.path.join(includes, path))
    dirs  = [path for path in os.listdir(includes) if isdir(path)]

    dialects = []
    for dialect in dirs:
        definition = os.path.join(includes, dialect, dialect + '.td')
        if not os.path.isfile(definition):
            continue
        pattern = r'let cppNamespace = "::vast::(.+)"'
        with open(definition, 'r') as td:
            for line in td:
                if match := re.search(pattern, line):
                    dialects.append({
                        'name': dialect,
                        'namespace': match.group(1)
                    })
                    break

    return sorted(dialects, key=lambda dialect: dialect['name'])


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

def add_subdirectory(path: str, subdir: str):
    cmake_lists_path = os.path.join(path, 'CMakeLists.txt')
    append_and_sort(f"add_subdirectory({ subdir })\n", cmake_lists_path)
    print(f"Updating: { cmake_lists_path }")


class file_generator:
    def __init__(self):
        script_dir      = os.path.dirname(os.path.abspath(sys.argv[0]))
        templates_dir   = os.path.join(script_dir, 'templates')
        template_loader = jinja2.FileSystemLoader(searchpath=templates_dir)
        self.templates  = jinja2.Environment(loader=template_loader)

    def create(self, dir: str, template_name: str, maps):
        template = self.templates.get_template(template_name)
        action = 'Updating' if os.path.exists(dir) else 'Creating'
        template.stream(maps).dump(dir)
        print(f"{ action }: { dir }")


class dialect_generator:
    def __init__(self, opts):
        if opts['passes']:
            opts['internal_transforms'] = '_and_passes'

        self.opts = opts
        self.name = opts['dialect_name']

        self.dialects_includes = os.path.join(project_dir, f"include/vast/Dialect/")
        self.includes = os.path.join(self.dialects_includes, self.name)

        self.dialects_sources = os.path.join(project_dir, f"lib/vast/Dialect/")
        self.sources = os.path.join(self.dialects_sources, self.name)

        self.transforms = os.path.join(self.sources, 'Transforms')

        self.dialect_names = [dialect['name'] for dialect in
            gather_dialects(os.path.join(project_dir, f"include/vast/Dialect/"))
        ]

        self.generator = file_generator()

    def create_include(self, file_name: str, template_name: str):
        path = os.path.join(self.includes, file_name)
        self.generator.create(path, template_name, self.opts)

    def create_source(self, file_name: str, template_name: str):
        path = os.path.join(self.sources, file_name)
        self.generator.create(path, template_name, self.opts)

    def create_transform(self, file_name: str, template_name: str):
        path = os.path.join(self.transforms, file_name)
        self.generator.create(path, template_name, self.opts)

    def generate_dialect_includes(self):
        # Create dialect includes directory
        os.makedirs(self.includes, exist_ok=True)

        # Register dialect subdirectory in cmake
        add_subdirectory(self.dialects_includes, self.name)

        # Generate dialect includes CMakeLists.txt
        self.create_include('CMakeLists.txt', 'dialect.includes.cmake.in')

        # Generate headers
        self.create_include(f'{ self.name }.td', 'Dialect.td.in')
        self.create_include(f'{ self.name }Dialect.hpp', 'Dialect.hpp.in')
        self.create_include(f'{ self.name }Ops.td', 'Ops.td.in')
        self.create_include(f'{ self.name }Ops.hpp', 'Ops.hpp.in')

        if self.opts['has_types']:
            self.create_include(f'{ self.name }Types.td', 'Types.td.in')
            self.create_include(f'{ self.name }Types.hpp', 'Types.hpp.in')

        if self.opts['has_attributes']:
            self.create_include(f'{ self.name }Attributes.td', 'Attributes.td.in')
            self.create_include(f'{ self.name }Attributes.hpp', 'Attributes.hpp.in')

        if self.opts['passes']:
            self.create_include(f'Passes.td', 'Passes.td.in')
            self.create_include(f'Passes.hpp', 'Passes.hpp.in')

    def update_registered_dialects(self):
        dialects = { 'dialects': gather_dialects(self.dialects_includes) }
        path = os.path.join(self.dialects_includes, 'Dialects.hpp')
        self.generator.create(path, 'Dialects.hpp.in', dialects)

    def generate_dialect_sources(self):
        # Create dialect sources directory
        os.makedirs(self.sources, exist_ok=True)

        # Register dialect subdirectory in cmake
        add_subdirectory(self.dialects_sources, self.name)

        # Generate dialect sources CMakeLists.txt
        self.create_source('CMakeLists.txt', 'dialect.sources.cmake.in')

        # generate Dialect.cpp
        self.create_source(f'{ self.name }Dialect.cpp', 'Dialect.cpp.in')
        self.create_source(f'{ self.name }Ops.cpp', 'Ops.cpp.in')

        if self.opts['has_types']:
            self.create_source(f'{ self.name }Types.cpp', 'Types.cpp.in')

        if self.opts['has_attributes']:
            self.create_source(f'{ self.name }Attributes.cpp', 'Attributes.cpp.in')

        if self.opts['passes']:
            os.makedirs(self.transforms, exist_ok=True)
            self.create_transform(f'CMakeLists.txt', 'dialect.transforms.cmake.in')
            self.create_transform(f'PassesDetails.hpp', 'PassesDetails.hpp.in')
            # TODO query for pass name
            self.create_transform(f'ExamplePass.cpp', 'Pass.cpp.in')

    def run(self):
        def check_proceed(dir: str, file_kind: str):
            if not proceed_query(f"Generate dialect { file_kind } files into: { dir }"):
                return False
            if os.path.exists(dir):
                return proceed_query(f"Dialect '{ self.name }' already exists. " \
                                     f"Do you want to overwrite its { file_kind } files?")
            return True

        if check_proceed(self.includes, 'includes'):
            self.generate_dialect_includes()
            self.update_registered_dialects()

        if check_proceed(self.sources, 'sources'):
            self.generate_dialect_sources()


def generate_dialect_templates(opts):
    generator = dialect_generator(opts)
    generator.run()


dialect_config = {
    'generator': generate_dialect_templates,
    'option_name': 'dialect',
    'help': 'TODO',
    'options': [
        {
            'type': 'input',
            'name': 'dialect_name',
            'message': "What is the dialect's name?",
        },
        {
            'type': 'input',
            'name': 'dialect_mnemonic',
            'message': "What is the dialect's mnemonic?",
        },
        {
            'type': 'input',
            'name': 'dialect_namespace',
            'message': "What is the dialect's C++ namespace?",
        },
        {
            'type': 'confirm',
            'name': 'has_types',
            'message': 'Does the dialect provide types?',
        },
        {
            'type': 'confirm',
            'name': 'has_attributes',
            'message': 'Does the dialect provide attributes?',
        }
    ]
}

dialect_passes_options = [
    {
        'type': 'confirm',
        'name': 'continue',
        'message': f'Do you want to generate internal dialect pass?',
        'default': False
    },
    {
        'type': 'input',
        'name': 'pass',
        'message': 'Enter pass name:',
        'when': lambda answers: answers['continue']
    }
]

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
            'message': 'Is the conversion endomorphism?',
        }
    ]
}

#
# Interface Generation
#

def generate_interface_templates(opts):
    pass

interface_config = {
    'generator': generate_interface_templates,
    'option_name': 'interface',
    'help': 'TODO',
    'options': [
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
            'type': 'list',
            'name': 'dialect',
            'message': 'Choose dialect for type:',
            'choices': gather_dialects(dialects_includes)
        },
        {
            'type': 'input',
            'name': 'name',
            'message': "What is the operation's name?",
        }
    ]
}

#
# Type Generation
#

def generate_type_templates(opts):
    pass

type_config = {
    'generator': generate_type_templates,
    'option_name': 'type',
    'help': 'TODO',
    'options': [
        {
            'type': 'list',
            'name': 'dialect',
            'message': 'Choose dialect for type:',
            'choices': gather_dialects(dialects_includes)
        },
        {
            'type': 'input',
            'name': 'name',
            'message': "What is type's name?",
        }
    ]
}

#
# Attribute Generation
#

def generate_attr_templates(opts):
    pass

attr_config = {
    'generator': generate_attr_templates,
    'option_name': 'attribute',
    'help': 'TODO',
    'options': [
        {
            'type': 'list',
            'name': 'dialect',
            'message': 'Choose dialect for attribute:',
            'choices': gather_dialects(dialects_includes)
        },
        {
            'type': 'input',
            'name': 'name',
            'message': "What is attribute's name?",
        }
    ]
}

#
# Config
#

configs = [
    dialect_config, conversion_config, interface_config, operation_config, type_config, attr_config
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

#
# Main
#

def main():
    # query for target templates to be generated
    target = query(prologue)
    config = get_target_config(target['generate'])
    # query for options of desired target templates
    opts = query(config['options'])

    # gather dialect passes
    if target['generate'] == 'dialect':
        opts['passes'] = []
        while True:
            answers = query(dialect_passes_options)
            if not answers['continue']:
                break
            # hack around unssuported conditional message
            dialect_passes_options[0]['message'] = f'Do you want to generate another internal dialect pass?'
            if pass_name := answers['pass']:
                opts['passes'].append(pass_name)

    # fill in deduced target options
    fill_template_options(opts, target)
    # generate demplates from gathered options
    config['generator'](opts)

if __name__ == "__main__":
    main()

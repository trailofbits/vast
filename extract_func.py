#!/usr/bin/env python3
# primitive call/def grabber tool

import sys
import os
from typing import List
import re

class location:
    def __init__(self, reference: int, filename: str, line: int):
        self.reference = reference
        self.filename = filename
        self.line = line 
    
    def __str__(self):
        return "#loc" + self.reference + " = " + self.filename + ":" + str(self.line)

    @staticmethod
    def parse(literal: str):
        if m := re.match(r'#loc(\d+) = loc\(("[^"]+"):(\d+):(\d+)\)', literal):
            return location(m.group(1), m.group(2), int(m.group(3)))

        return None

class fdef:
    def __init__(self, signature: str, line: int, ref):
        self.signature = signature
        self.line = line
        self.ref = ref

    @staticmethod
    def parse(literal: str, line: int):
        if m := re.search(r"hl.func @([^:^\s]+)", literal):
            mm = re.search(r"loc\(#loc(\d+)\)", literal)
            if mm == None:
                mm = re.search(r'loc\(("[^"]+"):(\d+):(\d+)\)', literal)
                if mm == None:
                    print(f"Debug: Could not find reference for function def in literal '{literal}'")
                    return fdef(m.group(1), line, None)
                else: 
                    return fdef(m.group(1), line, mm.group(1))
            else: 
                return fdef(m.group(1), line, mm.group(1))
        
        return None
        
    def __str__(self):
        return self.signature + ":" + str(self.line)
    
class fcall:
    def __init__(self, definition: fdef, line: int, ref = None):
        self.function = definition
        self.line = line
        self.ref = ref

    @staticmethod
    def parse(literal: str, line: int):
        if m := re.search(r"hl.call @([^\)]+\))", literal):
            mm = re.search(r"loc\(#loc(\d+)\)", literal)
            if mm == None:
                print(f"Error: function call '{literal}' does not have refernce check")
                exit(1)
            return fcall(m.group(1), line, mm.group(1))
        else:
            print(f"literal {literal} is not a valid function call.")
            return None
    
    def __str__(self):
        return self.function.__str__() + " call"
        

def smart_parse(line: str, loc: int):
    if re.search(r"hl.call @([^:^\s]+)", line):
        return fcall.parse(line, loc)
    if re.match(r'#loc(\d+) = loc\(("[^"]+"):(\d+):(\d+)\)', line):
        return location.parse(line)
    if re.search(r"hl.func @([^:^\s]+)", line):
        return fdef.parse(line, loc)
    
    return None

def main():
    if len(sys.argv) != 2:
        print("USAGE: python3 extract_func.py <path to mlir file>")
        exit(0)

    if not os.path.exists(sys.argv[1]) or not os.path.isfile(sys.argv[1]):
        print(f"file {sys.argv[1]} does not exist, or is not a file.")
        exit(1)
    
    locations: List[location] = []
    functions: List[fdef] = []
    calls: List[fcall] = []

    with open(sys.argv[1], 'r') as f:
        for (index, line) in enumerate(f.readlines()):
            obj = smart_parse(line, index+1)
            
            if not obj: continue

            if isinstance(obj, location):
                # line location -> location info.
                locations.append(obj)
            elif isinstance(obj, fdef):
                functions.append(obj)
            elif isinstance(obj, fcall):
                calls.append(obj)

        f.close()
    
    consolidated_calls = {}

    for function_call in calls:
        if function_call.function in consolidated_calls.keys():
            consolidated_calls[function_call.function][0].append(function_call.line)
            consolidated_calls[function_call.function][1].append(function_call.ref)
        else:
            consolidated_calls[function_call.function] =  [[function_call.line], [function_call.ref]]

    with open("analysis.yaml", "w") as f:
        
        for function in consolidated_calls.keys():

            # gathering function calls...
            filerefs = set()

            for refs in consolidated_calls[function][1]:
                found = False
                for l in locations:
                    if l.reference == refs:
                        filerefs.add(l.filename)
                        found = True
                        break
                if not found:
                    print(f"cannot find ref {refs}!")
                    exit(1)

            defs = set()

            function_friendly_name = re.search(r"[^\(]+", function).group(0)

            for definition in functions:
                if definition.signature == function_friendly_name:
                    defs.add(definition.line) 

                    found = False
                    for l in locations:
                        if l.reference == definition.ref:
                            filerefs.add(l.filename)
                            found = True
                            break
                    if not found:
                        print(f"cannot find ref {definition.ref} of {definition.signature} on line {definition.line}, considering as file name")
                        filerefs.add(definition.ref)
                        
            
            # write to yaml
            f.write(f" - signature: {function}\n\t call lines: ")

            f.writelines(["\n\t\t- " + str(i) if i != None else "" for i in consolidated_calls[function][0]])

            f.write("\n\t definitions: ")

            f.writelines(["\n\t\t- " + str(d) if d != None else "" for d in defs])

            f.write("\n\t file references: ")

            f.writelines(["\n\t\t- " + (f if f != None else "unknown") for f in filerefs])

            f.write("\n")
            

if __name__ == '__main__':
    if sys.version_info < (3, 8):
        print("Python version above 3.8 is required.")
        exit(1)
    main()
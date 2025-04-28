""" A simple compiler for the B language """
from argparse import ArgumentParser
from dataclasses import dataclass
from enum import Enum
import os


B_KEYWORDS = ["extrn"]


def is_keyword(keyword: str) -> bool:
    return keyword in B_KEYWORDS

# where do we store variables
# auto     is that we manage memory and keep track of that
# external is that it must defined somewhere else and
#          it is up to the user to have it defined
class Storage(Enum):
    auto: int = 1
    external: int = 2


# variables
@dataclass
class Variable:
    name: str
    index: int
    where: str
    storage: Storage


# functions
class Arg:
    def __init__(self, name, value):
        self.value = value


@dataclass
class FunctionCall:
    name: str
    args: list[Arg] | None


@dataclass
class ExternVariable:
    name: str


Operations = FunctionCall | ExternVariable


# lexer for parsing
@dataclass
class Lexer():
    l: int = 0
    r: int = 0

operations: list[Operations] = []
variables: list[Variable] = []

def variable_exists(name: str) -> Variable | None:
    for var in variables:
        if var.name == name:
            return var
    return None

@dataclass
class FilePosition:
    line: int
    column: int


def index_to_file_position(index: int, s: str) -> FilePosition:
    count = 0
    line = 1
    for i, char in enumerate(s):
        if char == "\n":
            count = 0
            line += 1
        else:
            count += 1
        if i == index:
            return FilePosition(line=line, column=count)
    raise ValueError("ERROR: could not locate index in file.")


def advance(p: Lexer) -> Lexer:
    return Lexer(l = p.l, r = p.r + 1)

def shift(p: Lexer, n: int) -> Lexer:
    return Lexer(p.l + n, p.r + n)

def shift_right(p: Lexer) -> Lexer:
    return shift(p, 1)

def parse(s: str, p: Lexer, level: int = 0):

    indent = " "*(level +2)
    while True:

        if s[p.l] == ")":
            p = shift_right(p)

        if s[p.l] == " ":
            p = shift_right(p)

        if s[p.l] == "{":
            p = shift_right(p)

        if s[p.r] == "\n":
            p = shift(p, 2)

        if s[p.r] == ";":
            p = shift(p, 2)

        if s[p.l] == "}":
            return None

        p = advance(p)
        if s[p.r] == "(":
            token = s[p.l:p.r]
            # print(f"  {indent:s}function :: NAME at [l, r] = [{p.l:d}, {p.r:d}] content = '{token:s}'")
            p.l = p.r
            args = parse(s, p, level +2)
            operations.append(FunctionCall(name=token, args=args))
            offset = 0
            if args is not None:
                offset += len(str(args))
            p = shift(Lexer(p.r, p.r), offset+1)

        if s[p.r] == " " or s[p.r] == ";":
            token = s[p.l:p.r]
            if token == " ":
                p = shift_right(p)
                continue

            if is_keyword(token):
                # print(f"  {indent:s}token :: KEYWORD at [l, r] = [{p.l:d}, {p.r:d}] content = '{token:s}'")
                p.l = p.r
                variable_name = parse(s, p, level +2)
                if token == "extrn":
                    if existing_var := variable_exists(variable_name):
                        new_position = index_to_file_position(p.r+1, s)
                        old_position = index_to_file_position(existing_var.where, s)
                        print(f"ERROR: Variable '{variable_name:s}' at line {new_position.line:d} column {new_position.column:d} is already defined at line {old_position.line:d} column {old_position.column:d}")
                        exit(1)
                    operations.append(ExternVariable(name=variable_name))
                    variables.append(Variable(name=variable_name, index=0, where=p.r+1, storage=Storage.external))
                # we shift to after the keyword
                p = shift(Lexer(p.r, p.r), len(variable_name)+2)
            else:
                # print(f"  {indent:s}token :: NAME at [l, r] = [{p.l:d}, {p.r:d}] content = '{token:s}'")
                return token
    
        if s[p.r] == ")":
            if s[p.l] == "(":
                token = s[p.l+1:p.r]
                # print(f"  {indent:s}function :: ARGUMENT = at [l, r] = [{p.l:d}, {p.r:d}] content = '{token:s}'")
                if p.r - p.l == 1:
                    return None
                else:
                    token = s[p.l+1:p.r]
                    return int(token)
    
        if p.r == len(s) -1:
            break

def compile_source():
    # make_prolog()
    s = "#!/usr/bin/env python3\n"

    # first argument in operations is the calling function
    first_func = operations.pop(0)
    s += "def {0:s}():\n".format(first_func.name)

    # we start by defining variables
    s += "    variables = []\n"
    for variable in variables:
        if variable.storage == Storage.external:
            continue

    for op in operations:
        if type(op) is FunctionCall:
            if op.args is None:
                s += "    {0:s}()\n".format(op.name)
            else:
                s += "    {0:s}({1:d})\n".format(op.name, op.args)
        elif type(op) is ExternVariable:
           s += "    from extrn import {0:s}\n".format(op.name)
        else:
           raise ValueError(f"Operation '{type(op):s}' not supported")

    # make_epilog()
    # the epilog in python is the magic that makes it run
    s += "if __name__ == \"__main__\":\n"
    s += "    {0:s}()\n".format(first_func.name)
    return s

if __name__ == "__main__":
    ap = ArgumentParser(description="""
Compiler for the b programming language.
Currently compiles it into python.
Provides a minimal standard library.
    """)
    ap.add_argument("input", help=".b source file to compile")
    ap.add_argument("-o", dest="output", default=None, help="output file of compilation. if not given will print program to screen.")
    args = ap.parse_args()
    filename = args.input

    s = ""
    with open(filename, "r") as f:
        for line in f:
            s += line #[:-1]

    print("-----------------------")
    print(" B-COMPILER by CSTEIN  ")
    print("    v. 0.0a            ")
    print("                       ")
    print("-----------------------")
    print("                       ")
    print("---- input program ----")
    print(s)
    # print("\n----   lexer   ----\n")
    p = Lexer(l = 0, r = 0)
    parse(s, p)
    # print("\n---- functions ----\n")
    # print(operations)
    # print("\n---- variables ----\n")
    # print(variables)
    print("\n--- compiled source ---\n")
    output = compile_source()
    if args.output is not None:
        with open(args.output, "w") as f:
            f.write(output)
        os.chmod(args.output, 0o744)
    else:
        print(output)

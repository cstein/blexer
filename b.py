""" A simple compiler for the B language """
from argparse import ArgumentParser
from dataclasses import dataclass
from enum import Enum
import os
import sys


DEBUG = False
B_KEYWORDS = ["extrn", "auto"]


def exit_with_error(error: str) -> None:
    """ Exits the compiler with an error

        :param error: the error to report to the user
    """
    print(error)
    sys.exit(1)


def is_keyword(keyword: str) -> bool:
    return keyword in B_KEYWORDS


class Storage(Enum):
    """ Specifies where variables are stored.

        :cvar AUTO: we manage memory and keep track of it using an internal stack
        :cvar EXTERNAL: it must defined somewhere else and it is up to the user to have it defined
    """
    AUTO: int = 1
    EXTERNAL: int = 2


@dataclass
class Variable:
    name: str
    index: int
    where: str
    storage: Storage


# functions can have arguments
class Argument:
    pass


@dataclass
class AutoVariable(Argument):
    index: int


@dataclass
class Literal(Argument):
    value: int


ArgumentType = AutoVariable | Literal


@dataclass
class FunctionCall:
    name: str
    args: list[ArgumentType] | None


@dataclass
class ExternVariable:
    name: str


@dataclass
class AutoAlloc:
    usize: int


@dataclass
class AutoAssign:
    index: int
    value: ArgumentType


Operations = FunctionCall | ExternVariable | AutoAlloc | AutoAssign


# lexer for parsing
@dataclass
class Lexer():
    l: int = 0
    r: int = 0


# TODO: make these local to the parser, and return them to the actual compiler instead of globals
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


def bomb_out(variable_name: str, index_old: int, index_new: int, s: str):
    new_position = index_to_file_position(index_old, s)
    old_position = index_to_file_position(index_new, s)
    exit_with_error(f"ERROR: Variable '{variable_name:s}' at line {new_position.line:d} column {new_position.column:d} is already defined at line {old_position.line:d} column {old_position.column:d}")


def advance(p: Lexer) -> Lexer:
    return Lexer(l = p.l, r = p.r + 1)


def shift(p: Lexer, n: int) -> Lexer:
    return Lexer(p.l + n, p.r + n)


def shift_right(p: Lexer) -> Lexer:
    return shift(p, 1)

def shift_to_end_of_expression(s: str, p: Lexer) -> Lexer:
    while s[p.r] not in ["\n", ";"]:
        p = advance(p)
    return shift(Lexer(p.r, p.r), 0)


def parse_function_argument(s: str, p: Lexer, level: int = 0) -> ArgumentType | None:
    indent = " " * (level +2)
    assert s[p.l] == "("
    while True:
        if DEBUG: print(f"[l, r] = [{p.l:d}, {p.r:d}]   ->   '{s[p.l:p.r]:s}'")

        if s[p.l] == "}":
            raise ValueError("did not expect end of function.")

        p = advance(p)

        if s[p.r] == ")":
            if s[p.l] == "(":
                token = s[p.l+1:p.r]
                if DEBUG: print(f"  {indent:s}function :: ARGUMENT = at [l, r] = [{p.l:d}, {p.r:d}] content = '{token:s}'")

                if p.r - p.l == 1:
                    return None
                else:
                    # check if it is either a literal
                    token = s[p.l+1:p.r]
                    try:
                        value = int(token)
                    except ValueError:
                        # expression or variable
                        if existing_var := variable_exists(token):
                            return AutoVariable(index=existing_var.index)
                        else:
                            raise ValueError(f"Variable '{token:s}' is not defined")
                    else:
                        return Literal(value=value)


def parse_variable(s: str, p: Lexer, level: int = 0) -> str:
    indent = " " * (level +2)
    while True:
        if DEBUG: print(f"{indent:s}[l, r] = [{p.l:d}, {p.r:d}]   ->   '{s[p.l:p.r]:s}'")
        if s[p.r] in ["\n", ";"]:
            p = shift(p, 2)
            continue

        if s[p.l] == "}":
            return None

        p = advance(p)

        if s[p.r] == " " or s[p.r] == ";":
            token = s[p.l:p.r]
            if is_keyword(token):
                raise ValueError("This was not expected.")

            if DEBUG: print(f"  {indent:s}variable :: NAME = at [l, r] = [{p.l:d}, {p.r:d}] content = '{token:s}'")
            return token


def parse_operation(s: str, p: Lexer):
    indent = "      parse_operation :: "
    while True:
        if DEBUG: print(f"{indent:s}[l, r] = [{p.l:d}, {p.r:d}]   ->   '{s[p.l:p.r]:s}'")
        if s[p.l] in [")", " ", "{"]:
            p = shift(p, 1)
            continue

        if s[p.r] in ["\n", ";"]:
            # end of expression - for now, we just die rather gracefully
            raise ValueError(f"got either newline (\\n) or end of expression (;) while parsing.")

        if s[p.l] == "=":
            # lets find the end of the expression (i.e., locate the ";")
            p = shift(p, 1)
            while s[p.r] not in ["\n", ";"]:
                p = advance(p)

            token = s[p.l:p.r]
            # we first try a literal
            try:
                value = int(token)
            except ValueError:
                # we are parsing an expression
                variable_name = token.strip()
                if existing_var := variable_exists(variable_name):
                    return AutoAssign, AutoVariable(index=existing_var.index)
                else:
                    exit_with_error(f"ERROR: Cannot assign expression '{token.strip()}' to variable. Only literals or variables are implemented now.")
            else:
                # we have a literal
                return AutoAssign, Literal(value=value)
            return None


def parse(s: str, p: Lexer, level: int = 0):
    indent = " " * (level +2)
    while True:
        if DEBUG: print(f"[l, r] = [{p.l:d}, {p.r:d}]   ->   '{s[p.l:p.r]:s}'")

        if s[p.l] == "}":
            return

        if s[p.l] in [")", " ", "{"]:
            p = shift(p, 1)
            continue

        if s[p.r] in ["\n", ";"]:
            p = shift(p, 2)
            continue


        p = advance(p)

        # parse functions
        if s[p.r] == "(":
            function_name = s[p.l:p.r]
            if DEBUG: print(f"  {indent:s}function :: NAME at [l, r] = [{p.l:d}, {p.r:d}] content = '{function_name:s}'")

            # we reset the lexer p to the right hand pointer
            # so we can parse any function arguments
            p.l = p.r
            args = parse_function_argument(s, p, level +2)
            match args:
                case None:
                    operations.append(FunctionCall(name=function_name, args=args))
                case Literal(value):
                    operations.append(FunctionCall(name=function_name, args=args))
                case AutoVariable(index):
                    operations.append(FunctionCall(name=function_name, args=args))
                case _:
                    raise ValueError(f"could not parse argument")

            # we shift to the end of the parsing of the argument
            p = shift_to_end_of_expression(s, p)
            continue

        if s[p.r] == " " or s[p.r] == ";":
            token = s[p.l:p.r]
            if token.isspace():
                p = shift_right(p)
                continue

            if is_keyword(token):
                if DEBUG: print(f"  {indent:s}keyword :: NAME at [l, r] = [{p.l:d}, {p.r:d}] content = '{token:s}'")
                p.l = p.r
                p = shift(p, 1)
                variable_name = parse_variable(s, p, level +2)  # TODO: maybe make specific parser for variables
                if existing_var := variable_exists(variable_name):
                    bomb_out(variable_name, p.r+1, existing_var.where, s)

                if token == "extrn":
                    operations.append(ExternVariable(name=variable_name))
                    variables.append(Variable(name=variable_name, index=-1, where=p.r+1, storage=Storage.EXTERNAL))

                if token == "auto":
                    operations.append(AutoAlloc(usize=1))
                    n_existing_auto = len([v for v in variables if v.storage == Storage.AUTO])
                    variables.append(Variable(name=variable_name, index=n_existing_auto, where=p.r+1, storage=Storage.AUTO))

                # we shift to after the keyword
                p = shift(Lexer(p.r, p.r), len(variable_name)+2)

            else:
                # if it is *not* a keyword, it could be a variable assignment
                if DEBUG: print(f"  {indent:s}token :: NAME at [l, r] = [{p.l:d}, {p.r:d}] content = '{token:s}'")

                # we check for existing variable with the same name
                if variable := variable_exists(token):
                    if variable.storage == Storage.EXTERNAL:
                        exit_with_error(f"ERROR: Cannot assign value to variable '{variable.name:s}' because it is declared EXTERNAL.")

                    elif variable.storage == Storage.AUTO:
                        p.l = p.r-1
                        # here we scan for assignment first I suppose? or do we scan for some operation?
                        op, rhs = parse_operation(s, shift(p,1))
                        if isinstance(rhs, Literal):
                            operations.append(AutoAssign(index=variable.index, value = rhs))
                        elif isinstance(rhs, AutoVariable):
                            operations.append(AutoAssign(index=variable.index, value = rhs))
                        else:
                            raise NotImplementedError("Not implemented yet.")
                        p = shift_to_end_of_expression(s, p)
                    else:
                        exit_with_error(f"ERROR: Variable '{variable.name:s}' has unspecified storage.")

                else:
                   position = index_to_file_position(p.l, s)
                   exit_with_error(f"ERROR: variable '{token:s}' at line {position.line:d} column {position.column:d} is not declared.")

        # exit if we are at the end of the code
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
        if variable.storage == Storage.EXTERNAL:
            continue

    for op in operations:
        if isinstance(op, FunctionCall):
            if op.args is None:
                s += "    {0:s}()\n".format(op.name)
            elif isinstance(op.args, Literal):
                s += "    {0:s}({1:d})\n".format(op.name, op.args.value)
            elif isinstance(op.args, AutoVariable):
                s += "    {0:s}(variables[{1:d}])\n".format(op.name, op.args.index)
            else:
                raise ValueError("type not understood.")

        elif isinstance(op, ExternVariable):
            s += "    from extrn import {0:s}\n".format(op.name)
        elif isinstance(op, AutoAlloc):
            s += "    variables.append(0)\n"
        elif isinstance(op, AutoAssign):
            # looks funky, but it is assignment (op) that takes an
            match op.value:
                case Literal(value):
                    s += "    variables[{0:d}] = {1:d}\n".format(op.index, op.value.value)
                case AutoVariable(index):
                    s += "    variables[{0:d}] = variables[{1:d}]\n".format(op.index, op.value.index)
                case _:
                    raise ValueError("Assignment of expressions are not supported.")
        else:
            raise ValueError(f"Operation '{str(type(op)):s}' not supported")

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
    print("    v. 0.0c            ")
    print("                       ")
    print("-----------------------")
    print("                       ")
    print("---- input program ----")
    print(s)
    # print("\n----   lexer   ----\n")
    p = Lexer(l = 0, r = 0)
    parse(s, p)
    if DEBUG: print("\n---- functions ----\n")
    if DEBUG: print(operations)
    if DEBUG: print("\n---- variables ----\n")
    if DEBUG: print(variables)
    print("\n--- compiled source ---\n")
    output = compile_source()
    if args.output is not None:
        with open(args.output, "w") as f:
            f.write(output)
        os.chmod(args.output, 0o744)
    else:
        print(output)

""" A simple compiler for the B language """
from argparse import ArgumentParser
from dataclasses import dataclass
from enum import Enum
import os
import sys


DEBUG = False
B_KEYWORDS = ["extrn", "auto"]
B_SPECIAL_CHARS = [","]


def exit_with_error(error: str) -> None:
    """ Exits the compiler with an error

        :param error: the error to report to the user
    """
    print(f"ERROR: {error:s}")
    sys.exit(1)


def is_keyword(keyword: str) -> bool:
    return keyword in B_KEYWORDS

def is_special_char(char: str) -> bool:
    return char in B_SPECIAL_CHARS

def is_valid_token(token: str) -> bool:
    if is_keyword(token):
        raise ValueError(f"'{token:s}' is a keyword and cannot be a variable name.")
    if is_special_char(token):
        raise ValueError(f"'{token:s}' is a special character and cannot be a variable name.")
    return is_keyword(token) and is_special_char(token)


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


# Arguments are generally to functions
class Argument:
    pass


@dataclass
class AutoVariable(Argument):
    """ A variable which we are responsible for memory
        :cvar index: a pointer to where the variable is stored in memory
    """
    index: int


@dataclass
class Literal(Argument):
    """ A literal value - a number. can be considered constant for the runtime of the program
        :cvar value: the value of store
    """
    value: int


ArgumentType = AutoVariable | Literal


@dataclass
class FunctionCall:
    """ A function has a name and takes one or more arguments

        :cvar name: the name of the function
        :cvar args: a list of arguments or None
    """
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


class Operation(Enum):
    """ Specifies the type of operation

        :cvar PLUS: An addition
    """
    PLUS: int = 1

@dataclass
class AutoBinaryOperation:
    """ We call it auto binary operation because we handle the memory allocation
        and perhaps deallocation

        :cvar op: Operation type (add, subtract ...)
        :cvar index: index for temporary storage if needed
        :cvar lhs: left hand side of the operator
        :cvar rhs: right hand side of the operator
    """
    op: Operation
    index: int
    lhs: ArgumentType
    rhs: ArgumentType


Operation = FunctionCall | ExternVariable | AutoAlloc | AutoAssign | AutoBinaryOperation


# lexer for parsing
@dataclass
class Lexer():
    l: int = 0
    r: int = 0


# TODO: make these local to the parser, and return them to the actual compiler instead of globals
# TODO: one way to make *multiple* functions is to use lists of lists.
operations: list[Operation] = []
variables: list[Variable] = []
# operations: list[list[Operation]] = [[]]
# variables: list[list[Variable]] = [[]]


def variable_exists(name: str, scope: list[Variable]) -> Variable | None:
    for var in scope:
        if var.name == name:
            return var
    return None


@dataclass
class FilePosition:
    line: int
    column: int


def index_to_file_position(index: int, s: str) -> FilePosition:
    """ converts an index (counter) to a FilePosition (line and column) """
    assert isinstance(index, int), "Input argument index is not integer. Got {0:s}".format(str(type(index)))
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
    raise IndexError("Could not locate position in file")


def error_variable_already_defined(variable_name: str, index_old: int, index_new: int, s: str):
    new_position = index_to_file_position(index_old, s)
    old_position = index_to_file_position(index_new, s)
    exit_with_error(f"ERROR: Variable '{variable_name:s}' at line {new_position.line:d} column {new_position.column:d} is already defined at line {old_position.line:d} column {old_position.column:d}")


def advance(p: Lexer) -> Lexer:
    return Lexer(l = p.l, r = p.r + 1)


def shift(p: Lexer, n: int) -> Lexer:
    if p.l < 0:
        raise ValueError("cannot have negative indices in the lexer pointer")
    return Lexer(p.l + n, p.r + n)


def shift_right(p: Lexer) -> Lexer:
    return shift(p, 1)


def shift_to_end_of_expression(s: str, p: Lexer) -> Lexer:
    while s[p.r] not in ["\n", ";"]:
        p = advance(p)
    return shift(Lexer(p.r, p.r), 0)


def parse_function_argument(s: str, p: Lexer, scope: list[Variable]) -> ArgumentType | None:
    indent = "    "
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
                        if existing_var := variable_exists(token, scope):
                            return AutoVariable(index=existing_var.index)
                        else:
                            raise ValueError(f"Variable '{token:s}' is not defined")
                    else:
                        return Literal(value=value)


def parse_variable(s: str, p: Lexer) -> str:
    indent = "    "
    while True:
        p = advance(p)
        if DEBUG: print(f"  {indent:s}[l, r] = [{p.l:d}, {p.r:d}]   ->   '{s[p.l:p.r]:s}' [[len = {len(s[p.l:p.r]):d}]]")
        if "}" in s[p.l:p.r]:
            raise ValueError("Parsing of source failed.")

        if ";" in s[p.l:p.r]:
            token = s[p.l:p.r-1]
            if DEBUG: print(f"  {indent:s}variable :: NAME = at [l, r] = [{p.l:d}, {p.r:d}] content = '{token:s}'")
            return Lexer(l = p.l, r = p.r-1), token

        if s[p.l:p.r] == ",":
            p = shift(p,1)

        if s[p.l:p.r] == " ":
            p = shift(p,1)

        if s[p.r] == ",":
            token = s[p.l:p.r]
            if DEBUG: print(f"  {indent:s}variable :: NAME = at [l, r] = [{p.l:d}, {p.r:d}] content = '{token:s}'")
            return p, token


def parse_expression(s: str, scope: list[Variable]):
    p = Lexer(l = 0, r = 1)
    while p.r != len(s):
        # we scan for either an operator (+, -, etc.)
        if s[p.r] == "(":
            # we have to find matching bracket and parse what is inside
            p.l = p.r
            if ")" not in s:
                exit_with_error(f"Could not find matching parenthesis in expression '{s:s}'")

        if s[p.r] == "+":
            # we have found the + operator, now we must parse the lhs and the rhs
            lhs = parse_expression(s[p.l:p.r].strip(), scope)
            rhs = parse_expression(s[p.r+1:].strip(), scope)
            # we basically return the statement as is and we attempt in no way
            # to optimize out operations. For example, we could simplify the
            # additions of two Literals into a single literal.
            return AutoBinaryOperation(op=Operation.PLUS, index=-1, lhs=lhs, rhs=rhs)
        p = advance(p)

    else:
        # we have parsed the expression but have not encountered weird stuff
        # so it must be a variable or a literal
        variable_name = s[p.l:p.r]
        if existing_var := variable_exists(variable_name, scope):
            return AutoVariable(index=existing_var.index)
        else:
            return Literal(value=int(s[p.l:p.r]))
        exit_with_error(f"Could not parse expression '{s:s}'")



def parse_operation(s: str, p: Lexer, scope: list[Variable]):
    indent = "      parse_operation :: "
    while True:
        if DEBUG: print(f"{indent:s}[l, r] = [{p.l:d}, {p.r:d}]   ->   '{s[p.l:p.r]:s}'")
        if s[p.l] in [")", " ", "{"]:
            p = shift(p, 1)
            continue

        if s[p.r] in ["\n", ";"]:
            # end of expression - for now, we just die rather gracefully
            raise ValueError(f"got either newline (\\n) or end of expression (;) while parsing.")

        # Assignment
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
                # if it is not a literal it must be a variable we try to assign from
                # but we require that this variable exists already.
                # we first look for whether that variable exists and just assign it
                variable_name = token.strip()
                if existing_var := variable_exists(variable_name, scope):
                    return AutoAssign, AutoVariable(index=existing_var.index)
                else:
                    # if it is *not* a variable it is an expression (and thus more complicated)
                    # position = index_to_file_position(variable_name, s)
                    operation = parse_expression(token.strip(), scope)
                    return AutoAssign, operation
                    # print(operation)
                    # exit_with_error(f"Cannot assign expression '{token.strip()}' to variable {variable_name}. Only literals or variables are implemented now.")
            else:
                # we have a literal
                return AutoAssign, Literal(value=value)
            return None


def parse(s: str, p: Lexer, level: int = 0):
    indent = " " * (level +2)
    index = 0
    scope_operations: list[Operation] = []
    scope_variables: list[Variables] = []
    while True:
        if DEBUG: print(f"[l, r] = [{p.l:d}, {p.r:d}]   ->   '{s[p.l:p.r]:s}'")

        if p.l > len(s) -1:
            return

        if s[p.l] in ["{"]:
            p = shift(p, 1)

        if s[p.l] in ["}"]:
            operations.append(scope_operations[:])
            variables.append(scope_variables[:])
            scope_operations.clear()
            scope_variables.clear()
            p = shift(p, 1)

        if s[p.l] in [")", " "]:
            p = shift(p, 1)
            continue

        if s[p.r] in ["\n", ";"]:
            p = shift(p, 1)
            continue

        p = advance(p)

        # exit if we are at the end of the code
        if p.r == len(s) -1:
            break

        # parse functions
        if s[p.r] == "(":
            function_name = s[p.l:p.r]
            if DEBUG: print(f"  {indent:s}function :: NAME at [l, r] = [{p.l:d}, {p.r:d}] content = '{function_name:s}'")
            p_end = shift_to_end_of_expression(s, p)

            # find matching end parenthesis from the back
            while s[p_end.r] != ")":
                p_end = shift(p_end, -1)

            # we reset the lexer p to the right hand pointer
            # so we can parse any function arguments
            p.l = p.r
            arguments = parse_function_argument(s, p, scope_variables)
            match arguments:
                case None:
                    scope_operations.append(FunctionCall(name=function_name, args=arguments))
                case Literal(value):
                    scope_operations.append(FunctionCall(name=function_name, args=arguments))
                case AutoVariable(index):
                    scope_operations.append(FunctionCall(name=function_name, args=arguments))
                case _:
                    raise ValueError(f"could not parse argument")

            # we shift to the end of the parsing of the argument
            p = shift(p_end, 1)
            continue

        if s[p.r] == " " or s[p.r] == ";":
            token = s[p.l:p.r]
            if token.isspace():
                p = shift_right(p)
                continue

            if is_keyword(token):
                if DEBUG: print(f"  {indent:s}keyword :: NAME at [l, r] = [{p.l:d}, {p.r:d}] content = '{token:s}'")
                p.l = p.r
                if token == "extrn":
                    p = shift(p, 1)
                    _, variable_name = parse_variable(s, p)
                    if existing_var := variable_exists(variable_name, scope_variables):
                        error_variable_already_defined(variable_name, p.r+1, existing_var.where, s)

                    scope_operations.append(ExternVariable(name=variable_name))
                    scope_variables.append(Variable(name=variable_name, index=-1, where=p.r+1, storage=Storage.EXTERNAL))

                if token == "auto":
                    while s[p.r] != ";":
                        p = shift(p, 1)
                        p_variable, variable_name = parse_variable(s, p)
                        if variable_name is None:
                            break

                        if existing_var := variable_exists(variable_name, scope_variables):
                            error_variable_already_defined(variable_name, p.r+1, existing_var.where, s)

                        scope_operations.append(AutoAlloc(usize=1))
                        n_existing_auto = len([v for v in scope_variables if v.storage == Storage.AUTO])
                        scope_variables.append(Variable(name=variable_name, index=n_existing_auto, where=p.r+1, storage=Storage.AUTO))
                        #p = shift(p, len(variable_name))
                        #p = shift(p_variable, 1)
                        p = shift(Lexer(l = p_variable.r, r = p_variable.r), 0)

                # we shift to after the keyword
                #p = shift(Lexer(p.r, p.r), len(variable_name)+2)
                p = shift_to_end_of_expression(s, p)

            else:
                # if it is *not* a keyword, it could be a variable assignment
                if DEBUG: print(f"  {indent:s}token :: NAME at [l, r] = [{p.l:d}, {p.r:d}] content = '{token:s}'")

                # we check for existing variable with the same name
                if variable := variable_exists(token, scope_variables):
                    if variable.storage == Storage.EXTERNAL:
                        exit_with_error(f"Cannot assign value to variable '{variable.name:s}' because it is declared EXTERNAL.")

                    elif variable.storage == Storage.AUTO:
                        p.l = p.r-1
                        # here we scan for assignment first I suppose? or do we scan for some operation?
                        # we get an operation back (op) and a list of arguments (args)
                        op: Operation
                        op, *args = parse_operation(s, shift(p,1), scope_variables)
                        if len(args) == 1:
                            rhs = args[0]
                            if isinstance(rhs, Literal):
                                scope_operations.append(op(index=variable.index, value = rhs))
                            elif isinstance(rhs, AutoVariable):
                                scope_operations.append(op(index=variable.index, value = rhs))
                            # temporary?
                            elif isinstance(rhs, AutoBinaryOperation):
                                scope_operations.append(op(index=variable.index, value=rhs))
                            # temporary end?
                            else:
                                raise NotImplementedError("Not implemented yet." + str(type(rhs)))
                        elif len(args) == 2:
                            lhs = args[0]
                            rhs = args[1]
                            #raise NotImplementedError(f"two arguments not supported")
                            # TODO add new storage
                            scope_operations.append(AutoBinaryOperation(op, index=-1, lhs=lhs, rhs=rhs))
                        else:
                            raise NotImplementedError(f"Compiler ERROR: number of arguments returned from parse_operation was {len(args):d} which is not supported.")
                            # more arguments
                        p = shift_to_end_of_expression(s, p)
                    else:
                        exit_with_error(f"Variable '{variable.name:s}' has unspecified storage.")

                else:
                   position = index_to_file_position(p.l, s)
                   exit_with_error(f"Variable '{token:s}' at line {position.line:d} column {position.column:d} is not declared.")


def compile_function(_operations: list[Operation], _variables: list[Variable]):
    first_func = _operations.pop(0)
    s = ""

    if first_func.name != "main":
        s += "def {0:s}():\n".format(first_func.name)

    # we start by defining variables
    s += "    variables = []\n"
    for variable in _variables:
        if variable.storage == Storage.EXTERNAL:
            continue

    for op in _operations:
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
            # looks funky, but it is assignment (op) that takes an argument
            match op.value:
                case Literal(value):
                    s += "    variables[{0:d}] = {1:d}\n".format(op.index, op.value.value)
                case AutoVariable(index):
                    s += "    variables[{0:d}] = variables[{1:d}]\n".format(op.index, op.value.index)
                case AutoBinaryOperation(operator, index, lhs, rhs):
                    s += "    variables[{0:d}] = ".format(op.index)
                    match lhs:
                        case Literal(value):
                            s += "{0:d}".format(lhs.value)
                        case AutoVariable(index):
                            s += "variables[{0:d}]".format(lhs.index)
                        case _:
                            raise ValueError("LHS could not be compiled.")
                    match operator:
                        case Operation.PLUS:
                            s += " + "
                        case _:
                            raise ValueError("Operator not understood")
                    match rhs:
                        case Literal(value):
                            s += "{0:d}".format(rhs.value)
                        case AutoVariable(index):
                            s += "variables[{0:d}]".format(rhs.index)
                        case _:
                            raise ValueError("LHS could not be compiled.")
                    s += "\n"
                case _:
                    raise ValueError("Assignment of expressions are not supported." + op.value)
        else:
            raise ValueError(f"Operation '{str(type(op)):s}' not supported")

    return s


def compile_source():
    """ Compiles the source into python3 compatible code """
    s = "#!/usr/bin/env python3\n"

    # we first find the main entrypoint to the program and pop that
    # so we can write that to the epilog
    index = 0
    for i, op in enumerate(operations):
        if op[0].name == "main":
            index = i

    main_operations = operations.pop(index)
    main_variables = variables.pop(index)

    for op, var in zip(operations, variables):
       s += compile_function(op, var)

    # the epilog in python is the magic that makes it run
    s += "if __name__ == \"__main__\":\n"
    s += compile_function(main_operations, main_variables)

    return s

if __name__ == "__main__":
    ap = ArgumentParser(description="""
Compiler for the b programming language.
Currently compiles it into python.
Provides a minimal standard library.
    """)
    ap.add_argument("input", help=".b source file to compile")
    ap.add_argument("-o", dest="output", default=None, help="output file of compilation. if not given will print program to screen.")
    ap.add_argument("--debug", dest="debug", action="store_true", default=False, help="set flag to enable debug output")
    args = ap.parse_args()
    DEBUG = args.debug
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

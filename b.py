""" A simple compiler for the B language """
from argparse import ArgumentParser
from dataclasses import dataclass
from enum import Enum
import os
import sys

DEBUG = False
VERBOSE = False
B_KEYWORDS = ["extrn", "auto", "while"]


def exit_with_error(error: str) -> None:
    """ Exits the compiler with an error

        :param error: the error to report to the user
    """
    print(f"ERROR: {error:s}")
    sys.exit(1)


def unreachable() -> None:
    raise RuntimeError("Unreachable code detected. This is a bug.")


def debug_info(message: str) -> None:
    if DEBUG:
        print(message)


def verbose_info(message: str) -> None:
    if VERBOSE:
        print(message)


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


# Arguments are generally to functions
class Argument:
    pass


@dataclass
class AutoVariable(Argument):
    """ A variable for which we are responsible for memory
        :cvar index: a pointer to where the variable is stored in memory
    """
    index: int


@dataclass
class Literal(Argument):
    """ A literal value, i.e. a number. Can be considered constant for the runtime of the program
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
    """ Requests memory to be allocated for usize variables

        :cvar usize: the number of items we need memory for
    """
    usize: int


@dataclass
class AutoAssign:
    """ A variable that is assigned a value

        :cvar index: the index of the memory we are storing
        :cvar value: the value we are storing into memory
    """
    index: int
    value: ArgumentType


class Operation(Enum):
    """ Specifies the type of operation

        :cvar PLUS: An addition of two numbers (a + b)
        :svar LESS: A test if a condition is met (a < b)
    """
    PLUS: int = 1
    LESS: int = 2


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

@dataclass
class ConditionalJump:
    condition: ArgumentType


@dataclass
class Jump:
    pass


OperationType = (
    FunctionCall
    | ExternVariable
    | AutoAlloc
    | AutoAssign
    | AutoBinaryOperation
    | ConditionalJump
    | Jump
)


@dataclass
class Lexer():
    l: int = 0
    r: int = 0


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

def shift_to_end_of_block(s: str, p: Lexer) -> Lexer:
    while s[p.r] not in ["}"]:
        p = advance(p)
    return shift(Lexer(p.r, p.r), 0)


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
            lhs = parse_expression(s[p.l:p.r].strip(), scope)
            rhs = parse_expression(s[p.r+1:].strip(), scope)
            # we basically return the statement as is and we attempt in no way
            # to optimize out operations. For example, we could simplify the
            # additions of two Literals into a single literal.
            return AutoBinaryOperation(op=Operation.PLUS, index=-1, lhs=lhs, rhs=rhs)

        if s[p.r] == "<":
            lhs = parse_expression(s[p.l:p.r].strip(), scope)
            rhs = parse_expression(s[p.r+1:].strip(), scope)
            # we basically return the statement as is and we attempt in no way
            # to optimize out operations. For example, we could simplify the
            # additions of two Literals into a single literal.
            return AutoBinaryOperation(op=Operation.LESS, index=-1, lhs=lhs, rhs=rhs)

        p = advance(p)

    else:
        # we have parsed the expression but have not encountered weird stuff
        # so it must be a variable or a literal
        variable_name = s[p.l:p.r]
        if existing_var := variable_exists(variable_name, scope):
            return AutoVariable(index=existing_var.index)
        else:
            return Literal(value=int(s[p.l:p.r]))
        unreachable()


def get_next_token(s: str, p: Lexer) -> str:
    indent = "    get_next_token::"
    while True:
        p = advance(p)
        debug_info(f"  {indent:s}[l, r] = [{p.l:d}, {p.r:d}]   ->   '{s[p.l:p.r]:s}' [[len = {len(s[p.l:p.r]):d}]]")
        if "}" in s[p.l:p.r]:
            raise RuntimeError("Parsing of source failed.")

        if ";" in s[p.l:p.r]:
            token = s[p.l:p.r-1]
            debug_info(f"  {indent:s}variable :: NAME = at [l, r] = [{p.l:d}, {p.r:d}] content = '{token:s}'")
            return Lexer(l = p.l, r = p.r-1), token

        if s[p.l:p.r] == ",":
            p = shift(p,1)

        if s[p.l:p.r] == " ":
            p = shift(p,1)

        if s[p.r] == "\n":
            position = index_to_file_position(p.r-1, s)
            exit_with_error(f"Unexpected end of line at line {position.line:d} column {position.column:d}. Expected ';' but got '\\n'.")

        if s[p.r] == ",":
            token = s[p.l:p.r]
            debug_info(f"  {indent:s}variable :: NAME = at [l, r] = [{p.l:d}, {p.r:d}] content = '{token:s}'")
            return p, token


def parse_function_arguments(s: str, scope: list[Variable]) -> list[ArgumentType | None]:
    """ Parses arguments to functions

        :param s: the argument(s) to parse
        :param scope: the current scope of variables already in use
    """
    if len(s) == 0:
        return [None]

    if "," not in s:
        return [parse_expression(s, scope)]
    else:
        return [parse_expression(item.strip(), scope) for item in s.split(",")]
    unreachable()


def parse_operation(s: str, p: Lexer, scope: list[Variable]):
    indent = "      parse_operation :: "
    while True:
        debug_info(f"{indent:s}[l, r] = [{p.l:d}, {p.r:d}]   ->   '{s[p.l:p.r]:s}'")
        if s[p.l] in [")", " ", "{"]:
            p = shift(p, 1)
            continue

        if s[p.r] in ["\n", ";"]:
            # end of expression - for now, we just die rather gracefully
            fp = index_to_file_position(p, s)
            exit_with_error(f"got unexpected character (\\n or ;) on line {fp.line:d}")

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
            else:
                # we have a literal
                return AutoAssign, Literal(value=value)

    unreachable()


def parse(s: str, p: Lexer, operations: list, variables: list, level: int = 0):
    indent = "   " * (level +2)
    scope_operations: list[Operation] = []
    scope_variables: list[Variable] = []
    while True:

        if p.l > len(s) -1:
            return operations, variables

        debug_info(f"{indent:s} [l, r] = [{p.l:d}, {p.r:d}]   ->   '{s[p.l:p.r]:s}'")

        if s[p.l] in ["{"]:
            p = shift(p, 1)

        if s[p.l] in ["}"]:
            if scope_operations:
                operations.append(scope_operations[:])
            variables.append(scope_variables[:])
            scope_operations.clear()
            scope_variables.clear()
            p = shift(p, 1)
            continue

        if s[p.l] in [")", " ", "\n"]:
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
            if is_keyword(function_name):

                fp = index_to_file_position(p.l, s)
                exit_with_error(f"Function name '{function_name:s}' at line {fp.line:d} column {fp.column:d} is a reserved keyword. Did you forget to put space ' ' between while and ()?")

            debug_info(f"  {indent:s}function :: NAME at [l, r] = [{p.l:d}, {p.r:d}] content = '{function_name:s}'")
            p_end = shift_to_end_of_expression(s, p)

            # find matching end parenthesis from the back
            while s[p_end.r] != ")":
                p_end = shift(p_end, -1)

            # we reset the lexer p to the right hand pointer
            # so we can parse any function arguments
            p.l = p.r
            arguments = parse_function_arguments(s[p.r+1:p_end.r], scope_variables + variables)
            #scope_operations.append(FunctionCall(name=function_name, args=arguments))
            args = []
            for argument in arguments:
                match argument:
                    case None:
                        args.append(argument)
                    case Literal(_):
                        args.append(argument)
                    case AutoVariable(_):
                        args.append(argument)
                    case AutoBinaryOperation(_, _, lhs, rhs):
                        # for function arguments, we cannot have expressions but only variables
                        # (at least if we ever want to support assembler)
                        # so if index is -1 we define a new variable and store the results in
                        # this variable **before** we call the function with that variable as an argument
                        scope_operations.append(AutoAlloc(usize=1))
                        n_existing_auto = len([v for v in scope_variables if v.storage == Storage.AUTO])
                        scope_variables.append(Variable(name="None", index=n_existing_auto, where=p.r+1, storage=Storage.AUTO))
                        scope_operations.append(AutoAssign(index=n_existing_auto, value=argument))
                        args.append(AutoVariable(index=n_existing_auto))
                    case _:
                        raise ValueError(f"could not parse argument of type {str(type(arguments)):s}")
            scope_operations.append(FunctionCall(name=function_name, args=args))

            # we shift to the end of the parsing of the argument
            p = shift(p_end, 1)
            continue

        if s[p.r] == " " or s[p.r] == ";":
            token = s[p.l:p.r]
            if token.isspace():
                p = shift_right(p)
                continue

            if is_keyword(token):
                debug_info(f"  {indent:s}keyword :: NAME at [l, r] = [{p.l:d}, {p.r:d}] content = '{token:s}'")
                p.l = p.r
                if token == "extrn":
                    p = shift(p, 1)
                    _ , variable_name = get_next_token(s, p)
                    if existing_var := variable_exists(variable_name, scope_variables):
                        error_variable_already_defined(variable_name, p.r+1, existing_var.where, s)

                    scope_operations.append(ExternVariable(name=variable_name))
                    scope_variables.append(Variable(name=variable_name, index=-1, where=p.r+1, storage=Storage.EXTERNAL))

                if token == "auto":
                    while s[p.r] != ";":
                        p = shift(p, 1)
                        p_variable, variable_name = get_next_token(s, p)
                        if variable_name is None:
                            break

                        if existing_var := variable_exists(variable_name, scope_variables):
                            error_variable_already_defined(variable_name, p.r+1, existing_var.where, s)

                        scope_operations.append(AutoAlloc(usize=1))
                        n_existing_auto = len([v for v in scope_variables if v.storage == Storage.AUTO])
                        scope_variables.append(Variable(name=variable_name, index=n_existing_auto, where=p.r+1, storage=Storage.AUTO))
                        p = shift(Lexer(l = p_variable.r, r = p_variable.r), 0)

                if token == "while":
                    p = shift(p, 1)
                    assert s[p.l] == "("
                    while s[p.r] != ")":
                        p = advance(p)
                    else:
                        # TODO: it works, but could use some cleanup
                        p = advance(p)
                        condition = parse_expression(s[p.l+1:p.r-1], scope_variables)
                        debug_info(f"{indent:s} condition for while loop: '{s[p.l+1:p.r-1]:s}' -> {str(condition):s}")
                        scope_operations.append(ConditionalJump(condition=condition))
                        p = shift(Lexer(l = p.r, r = p.r), 1)
                        assert s[p.l] == "{"
                        p_end = shift(shift_to_end_of_block(s, p), 0)
                        parse(s[p.l:p_end.r +1], Lexer(l = 0, r = 0), scope_operations, scope_variables, level=12)
                        scope_operations.append(Jump())
                        p = shift(shift_to_end_of_block(s, p), 0)

                p = shift_to_end_of_expression(s, p)

            else:
                # if it is *not* a keyword, it could be a variable assignment
                debug_info(f"  {indent:s}token :: NAME at [l, r] = [{p.l:d}, {p.r:d}] content = '{token:s}'")

                # we check for existing variable with the same name
                # NOTE: we look both in the current (local) scope but also in the scope of the parent
                if variable := variable_exists(token, scope_variables + variables):
                    if variable.storage == Storage.EXTERNAL:
                        exit_with_error(f"Cannot assign value to variable '{variable.name:s}' because it is declared EXTERNAL.")

                    elif variable.storage == Storage.AUTO:
                        p.l = p.r-1

                        op: OperationType
                        op, *args = parse_operation(s, shift(p,1), scope_variables + variables)
                        if len(args) == 1:
                            rhs = args[0]
                            if isinstance(rhs, Literal):
                                scope_operations.append(op(index=variable.index, value = rhs))
                            elif isinstance(rhs, AutoVariable):
                                scope_operations.append(op(index=variable.index, value = rhs))
                            elif isinstance(rhs, AutoBinaryOperation):
                                scope_operations.append(op(index=variable.index, value=rhs))
                            else:
                                raise NotImplementedError("Not implemented yet." + str(type(rhs)))
                        elif len(args) == 2:
                            lhs = args[0]
                            rhs = args[1]
                            scope_operations.append(AutoBinaryOperation(op, index=-1, lhs=lhs, rhs=rhs))
                        else:
                            raise NotImplementedError(f"Compiler ERROR: number of arguments returned from parse_operation was {len(args):d} which is not supported.")
                        p = shift_to_end_of_expression(s, p)
                    else:
                        exit_with_error(f"Variable '{variable.name:s}' has unspecified storage.")

                else:
                   position = index_to_file_position(p.l, s)
                   exit_with_error(f"Variable '{token:s}' at line {position.line:d} column {position.column:d} is not declared.")

    unreachable()


def compile_binop(operator: Operation, index: int, lhs: ArgumentType, rhs: ArgumentType) -> str:
    """ Compiles a binary operator

        :param operator: the operation (plus, minus, etc.) to perform
        :param index: the index of memory storage
        :param lhs: the left hand side of the operator
        :param rhs: the right hand side of the operator
    """
    s = ""
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
        case Operation.LESS:
            s += " < "
        case _:
            raise ValueError("Operator not understood")
    match rhs:
        case Literal(value):
            s += "{0:d}".format(value)
        case AutoVariable(index):
            s += "variables[{0:d}]".format(index)
        case _:
            raise ValueError("LHS could not be compiled.")
    return s


def compile_source(operations: list):
    """ Compiles the source into python3 compatible code """
    s = "#!/usr/bin/env python3\n"

    # we first find the main entrypoint to the program and pop that
    # so we can write that to the epilog
    index = 0
    for i, op in enumerate(operations):
        if op[0].name == "main":
            index = i
            break

    main_operations = operations.pop(index)

    for op in operations:
        s += compile_function(op)

    # the epilog in python is the magic that makes it run
    s += "if __name__ == \"__main__\":\n"
    s += compile_function(main_operations)

    return s


def compile_function(_operations: list[Operation]):
    first_func = _operations.pop(0)
    s = ""
    s_indent = " " * 4

    if first_func.name != "main":
        s += "def {0:s}():\n".format(first_func.name)

    # we start by defining local variable storage
    s += f"{s_indent:s}variables = []\n"

    s += compile_function_body(_operations, indent=4)
    return s


def compile_function_body(_operations: list[Operation], indent: int):
    s = ""
    s_indent = " " * indent
    for op in _operations:
        if isinstance(op, FunctionCall):
            s += "{0:s}{1:s}(".format(s_indent, op.name)
            for iarg, arg in enumerate(op.args, start=1):
                if arg is None:
                    break
                elif isinstance(arg, Literal):
                    s += "{0:d}".format(arg.value)
                elif isinstance(arg, AutoVariable):
                    s += "variables[{0:d}]".format(arg.index)
                else:
                    raise ValueError(f"type '{str(type(arg)):s}' not understood.")
                if len(op.args) != iarg:
                    s += ", "
            s += ")\n"
        elif isinstance(op, ExternVariable):
            s += "{0:s}from extrn import {1:s}\n".format(s_indent, op.name)
        elif isinstance(op, AutoAlloc):
            s += "{0:s}variables.append(0)\n".format(s_indent)
        elif isinstance(op, AutoAssign):
            # looks funky, but it is assignment (op) that takes an argument
            match op.value:
                case Literal(value):
                    s += "{0:s}variables[{1:d}] = {2:d}\n".format(s_indent, op.index, value)
                case AutoVariable(index):
                    s += "{0:s}variables[{1:d}] = variables[{2:d}]\n".format(s_indent, op.index, index)
                case AutoBinaryOperation(operator, index, lhs, rhs):
                    s += "{0:s}variables[{1:d}] = ".format(s_indent, op.index)
                    s += compile_binop(operator, index, lhs, rhs)
                    s += "\n"
                case _:
                    raise ValueError("Assignment of expressions are not supported." + str(op.value))
        elif isinstance(op, ConditionalJump):
            s += "{0:s}while ".format(s_indent)
            match op.condition:
                case Literal(value):
                    s += "{0:d}".format(value)
                case AutoVariable(index):
                    s += "variables[{0:d}]".format(index)
                case AutoBinaryOperation(operator, index, lhs, rhs):
                    s += compile_binop(operator, index, lhs, rhs)
                case _:
                    unreachable()
            s += ":\n"
        elif isinstance(op, Jump):
            s += "\n"
        elif isinstance(op, list):
            s += compile_function_body(op, indent + 4)
        else:
            raise RuntimeError(f"Operation '{str(type(op)):s}' not supported")

    return s


if __name__ == "__main__":
    ap = ArgumentParser(description="""
Compiler for the b programming language.
Currently compiles it into python.
Provides a minimal standard library.
    """)
    ap.add_argument("input", help=".b source file to compile.")
    ap.add_argument("-o", dest="output", default=None, help="Output of compilation. Uses same name as .b file if not given here.")
    ap.add_argument("-v", dest="verbose", default=False, action="store_true", help="Enable verbose output.")
    ap.add_argument("--debug", dest="debug", action="store_true", default=False, help="Enable debug output.")
    args = ap.parse_args()
    DEBUG = args.debug
    VERBOSE = args.verbose
    input_filename = args.input
    output_filename = args.output
    if output_filename is None:
        output_filename, _ = os.path.splitext(os.path.basename(input_filename))

    s = ""
    with open(input_filename, "r") as f:
        for line in f:
            s += line

    verbose_info("---- input program ----")
    verbose_info(s)

    # get ready for parsing source
    p = Lexer(l=0, r=0)
    operations: list[Operation] = []
    variables: list[Variable] = []
    operations, variables = parse(s, p, operations, variables)

    debug_info("\n---- functions ----\n")
    debug_info(operations)
    debug_info("\n---- variables ----\n")
    debug_info(variables)

    # compile the source
    output = compile_source(operations)
    verbose_info("\n--- compiled source ---\n")
    verbose_info(output)

    with open(output_filename, "w") as f:
        f.write(output)
    os.chmod(output_filename, 0o744)

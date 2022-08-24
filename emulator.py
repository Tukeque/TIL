from lark import Lark, Token, Tree
from typing import TypeAlias
from enum import Enum
import argparse

syntax = """
start: (program_block)*

program_block: /globals/ types                              -> directive
             | /fn/ "$" INT header code                     -> fn
             | /inline/ "$" INT header code                 -> inline
             | /extern/ "$" INT c_header "=" ESCAPED_STRING -> extern

header: (h_item)*
h_item: types      -> args
      | "->" types -> rets
      | "+" types  -> locals
c_header: (c_item)*
c_item: types      -> args
      | "->" types -> rets
types: "(" (INT_TYPES | FLOAT_TYPES)* ")"
code: "{" element* "}"
side: "[" (LETTER*)* "]"

INT_TYPES: "i4" | "i8" | "i16" | "i32" | "i64"
FLOAT_TYPES: "f32" | "f64"
CONTROL_OPS: "block" | "loop" | "jump" | "branch" | "ret" | "call" | "icall" | "perm"

INT_REGEX: /(i4|i8|i16|i32|i64)\.(add|sub|mul|div_u|div_s|mod_u|mod_s|xor|and|or|xnor|nand|nor|not|shl|shr_u|shr_s|rotl|rotr|load|store|const|eq|ne|lt_u|lt_s|gt_u|gt_s|le_u|le_s|ge_u|ge_s)/
FLOAT_REGEX: /(i32|i64)\.(todo)/ // todo!
SPECIAL_REGEX: /(global|local)\.(get|set)/

element: INT_REGEX                   -> int_op
       | FLOAT_REGEX                 -> float_op
       | SPECIAL_REGEX               -> special_op
       | CONTROL_OPS                 -> control
       | (HEX | BIN | INT | DECIMAL) -> imm
       | /\$[1-9][0-9]*|\$0/         -> func_label
       | "%" INT                     -> label
       | "{" element* "}"            -> code
       | side "->" side              -> perm
       | (c_item)+                   -> call_header

DOLLAR: "$"
HEX: /0x/ ("0".."9" | "A".."F" | "a".."f")+
BIN: /0b/ ("1" | "0")+
%import common.INT
%import common.DECIMAL
%import common.LETTER
%import common.ESCAPED_STRING
%import common.WS
%ignore WS

%import common.CPP_COMMENT
%import common.C_COMMENT
%ignore CPP_COMMENT
%ignore C_COMMENT

%import common.NEWLINE
%ignore NEWLINE
"""

parser = argparse.ArgumentParser(description="Emulator for TIL (Tukeque's Intermediate Language)")
parser.add_argument(
    "input", help="file to run (.til)"
)
parser.add_argument(
    "-d", "--debug", action="store_true",
    help="run in debug mode"
)
parser.add_argument(
    "-m", "--memory", metavar="MEM", default=256, type=int,
    help="amount of memory the emulator should run with (integer) - default 256"
)
parser.add_argument(
    "-s", "--stack", metavar="STK", default=256, type=int,
    help="length of stack the emulator should run with (integer) - default 256"
)
parse_args = parser.parse_args()

parser = Lark(syntax)
code = [line.replace("\n", "") for line in open(parse_args.input, "r").readlines()]
tree = parser.parse(open(parse_args.input, "r").read())
if parse_args.debug:
    print(tree.pretty())

def special_error(main: str, note: str = "", notes: list[str] = []) -> None:
    if note != "":
        notes.append(note)

    print(f"{main}")
    for note in notes:
        print(f"= {note}")
    exit()

def error(token: Token, main: str = "", secondary: str = "", note: str = "", notes: list[str] = []) -> None:
    line_str = str(token.line)
    pad = " " * len(line_str)

    if note != "":
        notes.append(note)

    print(f"{main}")
    print(f"{pad}--> {parse_args.input}:{token.line}:{token.column}")
    print(f"{pad} | ")
    print(f"{line_str} | {code[token.line - 1]}")
    print(f"{pad} | {' ' * (token.column - 1)}{'^' * (token.end_column - token.column)} {secondary}")
    print(f"{pad} | ")
    for note in notes:
        print(f"{pad} = {note}")
    exit()

def make_error_func(error_type: str): # returns a function
    def error_func(token: Token, main: str = "", secondary: str = "", note: str = "", notes: list[str] = []) -> None:
        error(token, f"{error_type}: {main}", secondary, note, notes)

    return error_func

def make_special_error_func(error_type: str): # returns a function
    def error_func(main: str, note: str = "", notes: list[str] = []) -> None:
        special_error(f"{error_type}: {main}", note, notes)

    return error_func

unimplemented_error = make_error_func("UnimplementedError")
validation_error = make_error_func("ValidationError")
runtime_error = make_error_func("RuntimeError")
cool_special_error = make_special_error_func("SpecialError")
todo_special_error = make_special_error_func("TodoError")

### first pass
Number: TypeAlias = int | float
Signature: TypeAlias = tuple[list[str], list[str]]
FuncSignature: TypeAlias = tuple[list[str], list[str], list[str]]
class ItemType(str, Enum):
    INSTR = "INSTR"
    IMM = "IMM"
    LABEL = "LABEL"
class Item:
    def __init__(self, token: Token, *, instr: str = None, imm: int = None, label: str = None):
        self.token = token

        if instr != None:
            self.instr: str = instr
            self.type = ItemType.INSTR
        elif imm != None:
            self.imm: Number = imm
            self.type = ItemType.IMM
        elif label != None:
            self.label: str = label
            self.type = ItemType.LABEL

    def __repr__(self):
        x = self.type.value # if this kind of thing is done a lot, make it a field in Item?
        match self.type:
            case ItemType.INSTR:
                return f"{x}: {self.instr}"
            case ItemType.IMM:
                return f"{x}: {self.imm}"
            case ItemType.LABEL:
                return f"{x}: {self.label}"

signatures: dict[str, Signature] = {
    "i32.add": (["i32", "i32"], ["i32"])
}
consts: dict[str, str] = {
    "i4.const": "i4",
    "i8.const": "i8",
    "i16.const": "i16",
    "i32.const": "i32",
    "i64.const": "i64"
}
call_signatures: dict[str, FuncSignature] = {}
CALL_ARGS = 0
CALL_RETS = 1
CALL_LOCS = 2
global_types: list[str] = []
return_types: list[str] = []
stack_types: list[str] = []
local_types: list[str] = []
gen_code: list[Item] = []
labels: list[int] = {}

def validate(instr: str, instr_token: Token, sig: Signature):
    global stack_types

    pre_sig = []

    for _ in sig[0]:
        if len(stack_types) != 0:
            pre_sig.append(stack_types.pop())
        else:
            validation_error(instr_token, "instruction signature does not match stack types", "stack is too small", note=f"instruction {instr} has signature {pretty_sig(sig)}")

    if pre_sig != sig[0]:
        validation_error(instr_token, f"instruction signature ({pretty_sig(sig)}) does not match stack types ({pretty_types(pre_sig)})", note=f"instruction {instr} has signature {pretty_sig(sig)}")

    for post in sig[1]: # push
        stack_types.append(post)

def pretty_sig(sig: Signature) -> str:
    return f"[{' '.join([t for t in sig[0]])}] -> [{' '.join([t for t in sig[1]])}]"

def pretty_types(types: list[str]) -> str:
    return f"[{' '.join([t for t in types])}]"

def parse_types(tree: Tree) -> list[str]:
    return [x.value for x in tree.children]

def compile_func(tree: Tree):
    global labels, gen_code, local_types, return_types

    def next(i: int) -> tuple[str, Token]:
        l = len(family := tree.children[3].children)
        if i + 1 >= l:
            validation_error(family[i].children[0], "expected token", "needs follow-up token")
        value = (tok := family[i + 1].children[0]).value

        return value, tok

    def assert_type(value: str, token: Token, expected_type: str, source_token: Token) -> Number:
        match expected_type:
            case "i4" | "i8" | "i16" | "i32" | "i64":
                if not value.isnumeric():
                    validation_error(token, f"\"{source_token.value}\" on {parse_args.input}:{source_token.line}:{source_token.column} expected type \"{type}\"", "got this") # todo better error stuff

                i = int(value)
                if i > 2 ** int(expected_type[1:]):
                    validation_error(token, "\"{value}\" has incorrect type", f"expected type {expected_type}", note="make sure to not overflow")

                return i

            case "f32" | "f64":
                todo_special_error("todo float type detection")

    labels[f"${tree.children[1]}"] = len(gen_code)
    header = tree.children[2]
    local_types = []
    return_types = [] # could be local?
    
    for child in header.children:
        match child:
            case Tree(data="args", children=[args, *_]):
                local_types = parse_types(args)

            case Tree(data="rets", children=[rets, *_]):
                return_types = parse_types(rets)

            case Tree(data="locals", children=[locals, *_]):
                local_types += parse_types(locals)

    # header done
    for i, expr in enumerate(tree.children[3].children):
        value = (tok := expr.children[0]).value
        match expr.data: # todo validation
            case "int_op" | "float_op":
                if value in consts:
                    type = consts[value]
                    next_value, next_token = next(i)
                    index = assert_type(next_value, next_token, type, tok)

                    gen_code.append(Item(tok, instr=value))
                    validate(value, tok, ([], [type]))
                else:
                    if value in signatures:
                        gen_code.append(Item(tok, instr=value))
                        validate(value, tok, signatures[value])
                    else:
                        unimplemented_error(expr.children[-1], "instruction doesnt have a signature yet")

            case "imm":
                gen_code.append(Item(tok, imm=int(value, 0)))

            case "special_op":
                match value:
                    case "local.get":
                        next_value, next_token = next(i)
                        index = assert_type(next_value, next_token, "i32", tok)

                        gen_code.append(Item(tok, instr=value))
                        validate(value, tok, ([], [local_types[index]]))

                    case _:
                        unimplemented_error(expr.children[-1], "instruction not implemented yet")

            case "func_label":
                gen_code.append(Item(tok, imm=value))

            case "control":
                match value:
                    case "ret":
                        gen_code.append(Item(tok, instr=value))
                        validate(value, tok, (return_types, []))

                        if len(stack_types) != 0:
                            validation_error(tok, "\"ret\" instruction leaves garbage in the stack", note="clean up before you return")

                    case "call":
                        next_value, next_token = next(i)
                        
                        if not(len(next_value) >= 2 and next_value[0] == "$" and next_value[1:].isnumeric()):
                            validation_error(next_token, f"invalid function label \"{next_value}\"", notes=["should be \"$\" followed by a decimal integer", "examples: $0, $54, $12"])

                        gen_code.append(Item(tok, instr=value))
                        call_signature = call_signatures[next_value]
                        validate(value, tok, (call_signature[0], call_signature[1]))

                    case _:
                        unimplemented_error(expr.children[-1], "instruction not implemented yet")

            case _:
                unimplemented_error(expr.children[-1], "instruction not implemented yet")

        if parse_args.debug:
            print(f"generated {gen_code[-1]}; {stack_types=}")

for program_block in tree.children: # first pass
    match program_block.data:
        case "directive":
            if program_block.children[0] == "globals":
                global_types = parse_types(program_block.children[1])

        case "fn":
            arg, ret, loc = [], [], []
            family = program_block.children
            for child in family[2].children:
                match child:
                    case Tree(data="args", children=[args, *_]):
                        arg = parse_types(args)

                    case Tree(data="rets", children=[rets, *_]):
                        ret = parse_types(rets)

                    case Tree(data="locals", children=[locals, *_]):
                        loc = parse_types(locals)

            call_signatures[f"${family[1]}"] = (arg, ret, loc)

        case _:
            unimplemented_error(program_block.children[0], "function block not implemented yet")

        #case "extern":
        #    unimplemented_error(program_block.children[0])
        #case "inline":
        #    unimplemented_error(program_block.children[0])

for program_block in tree.children: # second pass
    match program_block.data:
        case "fn":
            compile_func(program_block)
        
        case _: pass

for item in gen_code: # third pass
    match item.type:
        case ItemType.IMM:
            break
            if type(item.imm) == str and item.imm[0] == "$":
                item.imm = labels[item.imm]

        case _: pass

### emulate
return_types = []
locals = []
stack_lengths = []
call_stack = []
scope = 0
memory = [0 for _ in range(parse_args.memory)]
stack = []
if "$0" not in labels:
    cool_special_error("program has no entry point", note="entry point is marked by function $0 () -> () + ()") # todo validate signature
pc = labels["$0"]

i32 = 2**32 - 1

def pop() -> Number:
    return stack.pop()

def push(i: Number):
    stack.append(i)
    if len(stack) >= parse_args.stack:
        runtime_error(item.token, "exceeded stack limit")

def try_debug_print():
    if parse_args.debug:
        print(f"instruction {item.instr} at pc {item_pc}")
        print(f"| memory {memory}")
        print(f"| stack {stack}")

if parse_args.debug:
    print(f"\nlabels: {labels}")
    print(f"call signatures: {call_signatures}")
    print("gen_code:")
    pad = len(str(len(gen_code) - 1))

    for i, item in enumerate(gen_code):
        print(f"{str(i).rjust(pad)} | {item}")
    print("")

def finish():
    try_debug_print()
    exit()

run = True
while run:
    if pc >= len(gen_code):
        finish()

    item_pc = pc
    item = gen_code[pc]
    match item.type:
        case ItemType.INSTR:
            match item.instr:
                case "i32.add":
                    b = pop(); a = pop()
                    push((b + a) & i32)
                case "i32.const":
                    pc += 1
                    push(gen_code[pc].imm) # validate gen_code[pc + 1] is imm and i32
                case "local.get":
                    pc += 1
                    push(locals[-1][gen_code[pc].imm])
                case "ret":
                    if scope == 0:
                        finish()
                    scope -= 1

                    ret = return_types.pop()
                    stack_len = stack_lengths.pop()
                    ret_addr = call_stack.pop()

                    t = []
                    for r in ret: t.append(pop())
                    for r in t: push(r)
                    pc = ret_addr - 1
                case "call":
                    pc += 1
                    convention = call_signatures[gen_code[pc].imm]

                    locals.append([0 for _ in convention[CALL_ARGS] + convention[CALL_LOCS]])
                    stack_lengths.append(len(stack))
                    return_types.append(convention[CALL_RETS])
                    call_stack.append(pc + 1)
                    scope += 1

                    for i, arg in enumerate(convention[CALL_ARGS]):
                        locals[-1][-i-1] = pop() # pop in reverse order

                    pc = labels[gen_code[pc].imm] - 1 # jump

                case _:
                    runtime_error(item.token, "instruction's emulation behaviour not implemented yet")

        case _:
            runtime_error(item.token, "incorrect validation")

    pc += 1
    try_debug_print()

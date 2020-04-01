from inspect import getframeinfo, stack
import colorama


def varprint(*value):
    """Prints out a variable with extra logging info like what the var is"""
    my_stack = stack()
    self_stack = getframeinfo(my_stack[0][0])
    this_func_name = self_stack.function
    caller = getframeinfo(my_stack[1][0])
    line_text: str = caller.code_context[0].strip()
    assert line_text.startswith(this_func_name + "(")
    assert line_text.endswith(")")
    var_name = line_text[len(this_func_name + "("):-1]
    value_str = " ".join(map(str, value))
    print(f"{caller.filename}:{caller.lineno}\n"
          f"   {colorama.Fore.BLUE + var_name + colorama.Fore.RESET}={value_str}")

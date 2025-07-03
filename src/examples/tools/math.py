from langchain_core.tools import tool


@tool(response_format="content_and_artifact", parse_docstring=True)
def add(a: int, b: int) -> tuple[str, int]:
    """Adds a and b.

    Args:
        a: first int
        b: second int
    """
    result = a + b
    message = f"Adding {a} and {b} gives {result}"
    return message, result


@tool(response_format="content_and_artifact", parse_docstring=True)
def multiply(a: int, b: int) -> tuple[str, int]:
    """Multiply a and b.

    Args:
        a: first int
        b: second int
    """
    result = a * b
    message = f"Multiplying {a} and {b} gives {result}"
    return message, result


@tool(response_format="content_and_artifact", parse_docstring=True)
def divide(a: int, b: int) -> tuple[str, float]:
    """Divide a and b.

    Args:
        a: first int
        b: second int
    """
    if b == 0:
        raise ValueError("Cannot divide by zero")
    result = a / b
    message = f"Dividing {a} by {b} gives {result}"
    return message, result

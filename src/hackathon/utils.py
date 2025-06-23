import re


def remove_thinking_tags(response: str) -> str:
    return re.sub(r"<think>[\s\S]*?</think>\n+", "", response)

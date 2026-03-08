from __future__ import annotations


def extract_json_object(text: str) -> str:
    first_brace = text.find("{")
    if first_brace == -1:
        raise ValueError("Model response did not contain a JSON object.")

    depth = 0
    in_string = False
    escaped = False

    for index in range(first_brace, len(text)):
        char = text[index]

        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
            continue

        if char == "{":
            depth += 1
            continue

        if char == "}":
            depth -= 1
            if depth == 0:
                return text[first_brace : index + 1]

    raise ValueError("Model response contained an unterminated JSON object.")


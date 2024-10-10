from pathlib import Path
import re

def make_tempdirs(folder_path):
    Path(folder_path).mkdir(parents=True, exist_ok=True)


def extract_json(text):
    # Regular expression to capture JSON-like objects
    json_regex = r'(\{[^{}]*\})'

    # Find the first match
    matches = re.search(json_regex, text)

    if matches:
            # Get the prefix, json string, and postfix
        json_str = matches.group(1)
        prefix = text[:matches.start()]
        postfix = text[matches.end():]
        
        try:
            # Parse the matched JSON string into a Python dictionary
            json_obj = json.loads(json_str)
            print("Prefix:", prefix)
            prefix = prefix.replace("```json", "")
            print("Extracted JSON string:", json_str)
            print("Postfix:", postfix)
            print("Parsed JSON object:", json_obj)
            return (prefix, json_obj, postfix)
        except json.JSONDecodeError:
            print("Matched string is not a valid JSON object")
    else:
        print("No JSON object found")
    return (None, None, None)


def add_system_message(message_history, msg):
    message_history.append({"role": "system", "content": msg})

def add_user_message(message_history, msg):
    message_history.append({"role": "user", "content": msg})

def add_assistant_message(message_history, msg):
    message_history.append({"role": "assistant", "content": msg})
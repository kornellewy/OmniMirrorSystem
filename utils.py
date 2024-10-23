import yaml
from pathlib import Path

from dotenv import load_dotenv


def read_yaml_file(file_path):
    """
    Reads a YAML file and returns its content as a Python dictionary.
    :param file_path: The path to the YAML file.
    :return: A dictionary representing the YAML file's content.
    """
    try:
        with open(file_path, "r") as file:
            content = yaml.safe_load(file)
            return content
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return None
    except yaml.YAMLError as e:
        print(f"Error: Failed to parse the YAML file. {e}")
        return None


def write_yaml_file(file_path, data):
    """
    Writes a Python dictionary to a YAML file.

    :param file_path: The path to the YAML file.
    :param data: The dictionary to write to the file.
    """
    try:
        with open(file_path, "w") as file:
            yaml.safe_dump(data, file, default_flow_style=False)
        print(f"Successfully wrote to '{file_path}'")
    except Exception as e:
        print(f"Error: Could not write to the file. {e}")


def save_openai_api_key(
    api_key: str, env_file: str = str(Path(__file__).parent / ".env")
):
    """
    Saves the provided API key as OPENAI_API_KEY in the .env file.

    :param api_key: The API key to save.
    :param env_file: The path to the .env file (default is ".env").
    """
    # Read the existing content of the .env file
    try:
        with open(env_file, "r") as file:
            lines = file.readlines()
    except FileNotFoundError:
        lines = []

    # Check if OPENAI_API_KEY already exists and replace it, otherwise append it
    found = False
    for i, line in enumerate(lines):
        if line.startswith("OPENAI_API_KEY="):
            lines[i] = f"OPENAI_API_KEY={api_key}\n"
            found = True
            break

    if not found:
        lines.append(f"OPENAI_API_KEY='{api_key}'\n")

    # Write the updated content back to the .env file
    with open(env_file, "w") as file:
        file.writelines(lines)

    print(f"API key saved to {env_file} successfully.")
    load_dotenv()
    return "OPENAI_API_KEY saved successfully"

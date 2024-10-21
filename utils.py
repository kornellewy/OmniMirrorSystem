import yaml


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

import json
import os
import importlib
import gradio as gr
import sys

now_dir = os.getcwd()
folder = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "assets",
    "themes",
)
config_file = os.path.join(now_dir, "assets", "config.json")

sys.path.append(folder)


def read_json_file(filename):
    """Helper function to read a JSON file and return its contents."""
    with open(filename, "r", encoding="utf8") as json_file:
        return json.load(json_file)


def get_class(filename):
    """Retrieve the name of the first class found in the specified Python file."""
    with open(filename, "r", encoding="utf8") as file:
        for line in file:
            if "class " in line:
                class_name = line.split("class ")[1].split(":")[0].split("(")[0].strip()
                return class_name
    return None


def get_theme_list():
    """Compile a list of available themes from Python files and a JSON file."""
    themes_from_files = [
        os.path.splitext(name)[0]
        for root, _, files in os.walk(folder)
        for name in files
        if name.endswith(".py") and root == folder
    ]

    json_file_path = os.path.join(folder, "theme_list.json")
    themes_from_url = []

    try:
        themes_from_url = [item["id"] for item in read_json_file(json_file_path)]
    except FileNotFoundError:
        print("theme_list.json not found, proceeding with available files only.")

    return list(set(themes_from_files + themes_from_url))


def select_theme(name):
    """Select a theme by its name, updating the configuration file accordingly."""
    selected_file = f"{name}.py"
    full_path = os.path.join(folder, selected_file)

    config_data = read_json_file(config_file)

    if not os.path.exists(full_path):
        config_data["theme"]["file"] = None
        config_data["theme"]["class"] = name
    else:
        class_found = get_class(full_path)
        if class_found:
            config_data["theme"]["file"] = selected_file
            config_data["theme"]["class"] = class_found
        else:
            print(f"Theme class not found in {selected_file}.")
            return

    with open(config_file, "w", encoding="utf8") as json_file:
        json.dump(config_data, json_file, indent=2)

    message = f"Theme {name} successfully selected. Restart the application."
    print(message)
    gr.Info(message)


def load_theme():
    """Load the selected theme based on the configuration file."""
    try:
        config_data = read_json_file(config_file)
        selected_file = config_data["theme"]["file"]
        class_name = config_data["theme"]["class"]

        if class_name:
            if selected_file:
                module = importlib.import_module(selected_file[:-3])
                obtained_class = getattr(module, class_name)
                return obtained_class()
            else:
                return class_name
        else:
            print("No valid theme class found.")
            return None

    except Exception as error:
        print(f"An error occurred while loading the theme: {error}")
        return None


def read_current_theme():
    """Read the current theme class from the configuration file."""
    try:
        config_data = read_json_file(config_file)
        selected_file = config_data["theme"]["file"]
        class_name = config_data["theme"]["class"]

        return class_name if class_name else "ParityError/Interstellar"

    except Exception as error:
        print(f"An error occurred loading the theme: {error}")
        return "ParityError/Interstellar"

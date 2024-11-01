import os
import sys
import json

now_dir = os.getcwd()
sys.path.append(now_dir)

import gradio as gr
from assets.i18n.i18n import I18nAuto

i18n = I18nAuto()


def set_model_author(model_author: str):
    with open(os.path.join(now_dir, "assets", "config.json"), "r") as f:
        config = json.load(f)

    config["model_author"] = model_author

    with open(os.path.join(now_dir, "assets", "config.json"), "w") as f:
        json.dump(config, f, indent=4)

    print(f"Model author set to {model_author}.")
    return f"Model author set to {model_author}."


def get_model_author():
    with open(os.path.join(now_dir, "assets", "config.json"), "r") as f:
        config = json.load(f)

    return config["model_author"] if "model_author" in config else None


def model_author_tab():
    model_author_name = gr.Textbox(
        label=i18n("Model Author Name"),
        info=i18n("The name that will appear in the model information."),
        value=get_model_author(),
        placeholder=i18n("Enter your nickname"),
        interactive=True,
    )
    model_author_output_info = gr.Textbox(
        label=i18n("Output Information"),
        info=i18n("The output information will be displayed here."),
        value="",
        max_lines=1,
    )
    button = gr.Button(i18n("Set name"))

    button.click(
        fn=set_model_author,
        inputs=[model_author_name],
        outputs=[model_author_output_info],
    )

import os, sys
import gradio as gr
import shutil

now_dir = os.getcwd()
sys.path.append(now_dir)

from assets.i18n.i18n import I18nAuto
from core import run_model_blender_script

i18n = I18nAuto()


def update_model_fusion(dropbox):
    return dropbox, None


def voice_blender_tab():
    gr.Markdown(i18n("## Voice Blender"))
    gr.Markdown(
        i18n(
            "Select two voice models, set your desired blend percentage, and blend them into an entirely new voice."
        )
    )
    with gr.Column():
        model_fusion_name = gr.Textbox(
            label=i18n("Model Name"),
            info=i18n("Name of the new model."),
            value="",
            max_lines=1,
            interactive=True,
            placeholder=i18n("Enter model name"),
        )
        with gr.Row():
            with gr.Column():
                model_fusion_a_dropbox = gr.File(
                    label=i18n("Drag and drop your model here"), type="filepath"
                )
                model_fusion_a = gr.Textbox(
                    label=i18n("Path to Model"),
                    value="",
                    interactive=True,
                    placeholder=i18n("Enter path to model"),
                    info=i18n("You can also use a custom path."),
                )
            with gr.Column():
                model_fusion_b_dropbox = gr.File(
                    label=i18n("Drag and drop your model here"), type="filepath"
                )
                model_fusion_b = gr.Textbox(
                    label=i18n("Path to Model"),
                    value="",
                    interactive=True,
                    placeholder=i18n("Enter path to model"),
                    info=i18n("You can also use a custom path."),
                )
        alpha_a = gr.Slider(
            minimum=0,
            maximum=1,
            label=i18n("Blend Ratio"),
            value=0.5,
            interactive=True,
            info=i18n(
                "Adjusting the position more towards one side or the other will make the model more similar to the first or second."
            ),
        )
        model_fusion_button = gr.Button(i18n("Fusion"))
        with gr.Row():
            model_fusion_output_info = gr.Textbox(
                label=i18n("Output Information"),
                info=i18n("The output information will be displayed here."),
                value="",
            )
            model_fusion_pth_output = gr.File(
                label=i18n("Download Model"), type="filepath", interactive=False
            )

    model_fusion_button.click(
        fn=run_model_blender_script,
        inputs=[
            model_fusion_name,
            model_fusion_a,
            model_fusion_b,
            alpha_a,
        ],
        outputs=[model_fusion_output_info, model_fusion_pth_output],
    )

    model_fusion_a_dropbox.upload(
        fn=update_model_fusion,
        inputs=model_fusion_a_dropbox,
        outputs=[model_fusion_a, model_fusion_a_dropbox],
    )

    model_fusion_b_dropbox.upload(
        fn=update_model_fusion,
        inputs=model_fusion_b_dropbox,
        outputs=[model_fusion_b, model_fusion_b_dropbox],
    )

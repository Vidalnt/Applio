import json
import os
import random
import sys

import gradio as gr

now_dir = os.getcwd()
sys.path.append(now_dir)

from assets.i18n.i18n import I18nAuto
from core import run_tts_script
from tabs.inference.inference import (
    change_choices,
    create_folder_and_move_files,
    get_indexes,
    get_speakers_id,
    match_index,
    names,
    refresh_embedders_folders,
)

i18n = I18nAuto()

json_path = os.path.join("rvc", "lib", "tools", "tts_voices.json")
default_weight = random.choice(names) if names else ""

with open(json_path, "r") as file:
    tts_voices_data = json.load(file)

short_names = [voice.get("ShortName", "") for voice in tts_voices_data]


def process_input(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        file_contents = file.read()
    gr.Info(f"The text from the txt file has been loaded!")
    return file_contents, None


# TTS tab
def tts_tab():
    with gr.Column():
        with gr.Row():
            model_file = gr.Dropdown(
                label=i18n("Voice Model"),
                info=i18n("Select the voice model to use for the conversion."),
                choices=sorted(names, key=lambda path: os.path.getsize(path)),
                interactive=True,
                value=default_weight,
                allow_custom_value=True,
            )
            best_default_index_path = match_index(model_file.value)
            index_file = gr.Dropdown(
                label=i18n("Index File"),
                info=i18n("Select the index file to use for the conversion."),
                choices=get_indexes(),
                value=best_default_index_path,
                interactive=True,
                allow_custom_value=True,
            )
        with gr.Row():
            unload_button = gr.Button(i18n("Unload Voice"))
            refresh_button = gr.Button(i18n("Refresh"))

            unload_button.click(
                fn=lambda: (
                    {"value": "", "__type__": "update"},
                    {"value": "", "__type__": "update"},
                ),
                inputs=[],
                outputs=[model_file, index_file],
            )

            model_file.select(
                fn=lambda model_file_value: match_index(model_file_value),
                inputs=[model_file],
                outputs=[index_file],
            )

    gr.Markdown(
        i18n(
            f"Applio is a Speech-to-Speech conversion software, utilizing EdgeTTS as middleware for running the Text-to-Speech (TTS) component. Read more about it [here!](https://docs.applio.org/getting-started/tts#disclaimer)"
        )
    )
    tts_voice = gr.Dropdown(
        label=i18n("TTS Voices"),
        info=i18n("Select the TTS voice to use for the conversion."),
        choices=short_names,
        interactive=True,
        value=None,
    )

    tts_rate = gr.Slider(
        minimum=-100,
        maximum=100,
        step=1,
        label=i18n("TTS Speed"),
        info=i18n("Increase or decrease TTS speed"),
        value=0,
        interactive=True,
    )

    with gr.Tabs():
        with gr.Tab(label="Text to Speech"):
            tts_text = gr.Textbox(
                label=i18n("Text to Synthesize"),
                info=i18n("Enter the text to synthesize."),
                placeholder=i18n("Enter text to synthesize"),
                lines=3,
            )
        with gr.Tab(label="File to Speech"):
            txt_file = gr.File(
                label=i18n("Upload a .txt file"),
                type="filepath",
            )

    with gr.Accordion(i18n("Advanced Settings"), open=False):
        with gr.Column():
            output_tts_path = gr.Textbox(
                label=i18n("Output Path for TTS Audio"),
                placeholder=i18n("Enter output path"),
                value=os.path.join(now_dir, "assets", "audios", "tts_output.wav"),
                interactive=True,
            )
            output_rvc_path = gr.Textbox(
                label=i18n("Output Path for RVC Audio"),
                placeholder=i18n("Enter output path"),
                value=os.path.join(now_dir, "assets", "audios", "tts_rvc_output.wav"),
                interactive=True,
            )
            export_format = gr.Radio(
                label=i18n("Export Format"),
                info=i18n("Select the format to export the audio."),
                choices=["WAV", "MP3", "FLAC", "OGG", "M4A"],
                value="WAV",
                interactive=True,
            )
            sid = gr.Dropdown(
                label=i18n("Speaker ID"),
                info=i18n("Select the speaker ID to use for the conversion."),
                choices=get_speakers_id(model_file.value),
                value=0,
                interactive=True,
            )
            split_audio = gr.Checkbox(
                label=i18n("Split Audio"),
                info=i18n(
                    "Split the audio into chunks for inference to obtain better results in some cases."
                ),
                visible=True,
                value=False,
                interactive=True,
            )
            autotune = gr.Checkbox(
                label=i18n("Autotune"),
                info=i18n(
                    "Apply a soft autotune to your inferences, recommended for singing conversions."
                ),
                visible=True,
                value=False,
                interactive=True,
            )
            clean_audio = gr.Checkbox(
                label=i18n("Clean Audio"),
                info=i18n(
                    "Clean your audio output using noise detection algorithms, recommended for speaking audios."
                ),
                visible=True,
                value=True,
                interactive=True,
            )
            clean_strength = gr.Slider(
                minimum=0,
                maximum=1,
                label=i18n("Clean Strength"),
                info=i18n(
                    "Set the clean-up level to the audio you want, the more you increase it the more it will clean up, but it is possible that the audio will be more compressed."
                ),
                visible=True,
                value=0.5,
                interactive=True,
            )
            upscale_audio = gr.Checkbox(
                label=i18n("Upscale Audio"),
                info=i18n(
                    "Upscale the audio to a higher quality, recommended for low-quality audios. (It could take longer to process the audio)"
                ),
                visible=True,
                value=False,
                interactive=True,
            )
            pitch = gr.Slider(
                minimum=-24,
                maximum=24,
                step=1,
                label=i18n("Pitch"),
                info=i18n(
                    "Set the pitch of the audio, the higher the value, the higher the pitch."
                ),
                value=0,
                interactive=True,
            )
            filter_radius = gr.Slider(
                minimum=0,
                maximum=7,
                label=i18n("Filter Radius"),
                info=i18n(
                    "If the number is greater than or equal to three, employing median filtering on the collected tone results has the potential to decrease respiration."
                ),
                value=3,
                step=1,
                interactive=True,
            )
            index_rate = gr.Slider(
                minimum=0,
                maximum=1,
                label=i18n("Search Feature Ratio"),
                info=i18n(
                    "Influence exerted by the index file; a higher value corresponds to greater influence. However, opting for lower values can help mitigate artifacts present in the audio."
                ),
                value=0.75,
                interactive=True,
            )
            rms_mix_rate = gr.Slider(
                minimum=0,
                maximum=1,
                label=i18n("Volume Envelope"),
                info=i18n(
                    "Substitute or blend with the volume envelope of the output. The closer the ratio is to 1, the more the output envelope is employed."
                ),
                value=1,
                interactive=True,
            )
            protect = gr.Slider(
                minimum=0,
                maximum=0.5,
                label=i18n("Protect Voiceless Consonants"),
                info=i18n(
                    "Safeguard distinct consonants and breathing sounds to prevent electro-acoustic tearing and other artifacts. Pulling the parameter to its maximum value of 0.5 offers comprehensive protection. However, reducing this value might decrease the extent of protection while potentially mitigating the indexing effect."
                ),
                value=0.5,
                interactive=True,
            )
            hop_length = gr.Slider(
                minimum=1,
                maximum=512,
                step=1,
                label=i18n("Hop Length"),
                info=i18n(
                    "Denotes the duration it takes for the system to transition to a significant pitch change. Smaller hop lengths require more time for inference but tend to yield higher pitch accuracy."
                ),
                value=128,
                interactive=True,
            )
            f0_method = gr.Radio(
                label=i18n("Pitch extraction algorithm"),
                info=i18n(
                    "Pitch extraction algorithm to use for the audio conversion. The default algorithm is rmvpe, which is recommended for most cases."
                ),
                choices=[
                    "crepe",
                    "crepe-tiny",
                    "rmvpe",
                    "fcpe",
                    "hybrid[rmvpe+fcpe]",
                ],
                value="rmvpe",
                interactive=True,
            )
            embedder_model = gr.Radio(
                label=i18n("Embedder Model"),
                info=i18n("Model used for learning speaker embedding."),
                choices=[
                    "contentvec",
                    "chinese-hubert-base",
                    "japanese-hubert-base",
                    "korean-hubert-base",
                    "custom",
                ],
                value="contentvec",
                interactive=True,
            )
            with gr.Column(visible=False) as embedder_custom:
                with gr.Accordion(i18n("Custom Embedder"), open=True):
                    with gr.Row():
                        embedder_model_custom = gr.Dropdown(
                            label=i18n("Select Custom Embedder"),
                            choices=refresh_embedders_folders(),
                            interactive=True,
                            allow_custom_value=True,
                        )
                        refresh_embedders_button = gr.Button(i18n("Refresh embedders"))
                    folder_name_input = gr.Textbox(
                        label=i18n("Folder Name"), interactive=True
                    )
                    with gr.Row():
                        bin_file_upload = gr.File(
                            label=i18n("Upload .bin"),
                            type="filepath",
                            interactive=True,
                        )
                        config_file_upload = gr.File(
                            label=i18n("Upload .json"),
                            type="filepath",
                            interactive=True,
                        )
                    move_files_button = gr.Button(
                        i18n("Move files to custom embedder folder")
                    )
            f0_file = gr.File(
                label=i18n(
                    "The f0 curve represents the variations in the base frequency of a voice over time, showing how pitch rises and falls."
                ),
                visible=True,
            )

    convert_button = gr.Button(i18n("Convert"))

    with gr.Row():
        vc_output1 = gr.Textbox(
            label=i18n("Output Information"),
            info=i18n("The output information will be displayed here."),
        )
        vc_output2 = gr.Audio(label=i18n("Export Audio"))

    def toggle_visible(checkbox):
        return {"visible": checkbox, "__type__": "update"}

    def toggle_visible_embedder_custom(embedder_model):
        if embedder_model == "custom":
            return {"visible": True, "__type__": "update"}
        return {"visible": False, "__type__": "update"}

    clean_audio.change(
        fn=toggle_visible,
        inputs=[clean_audio],
        outputs=[clean_strength],
    )
    refresh_button.click(
        fn=change_choices,
        inputs=[model_file],
        outputs=[model_file, index_file, sid],
    )
    txt_file.upload(
        fn=process_input,
        inputs=[txt_file],
        outputs=[tts_text, txt_file],
    )
    embedder_model.change(
        fn=toggle_visible_embedder_custom,
        inputs=[embedder_model],
        outputs=[embedder_custom],
    )
    move_files_button.click(
        fn=create_folder_and_move_files,
        inputs=[folder_name_input, bin_file_upload, config_file_upload],
        outputs=[],
    )
    refresh_embedders_button.click(
        fn=lambda: gr.update(choices=refresh_embedders_folders()),
        inputs=[],
        outputs=[embedder_model_custom],
    )
    convert_button.click(
        fn=run_tts_script,
        inputs=[
            tts_text,
            tts_voice,
            tts_rate,
            pitch,
            filter_radius,
            index_rate,
            rms_mix_rate,
            protect,
            hop_length,
            f0_method,
            output_tts_path,
            output_rvc_path,
            model_file,
            index_file,
            split_audio,
            autotune,
            clean_audio,
            clean_strength,
            export_format,
            upscale_audio,
            f0_file,
            embedder_model,
            embedder_model_custom,
            sid,
        ],
        outputs=[vc_output1, vc_output2],
    )

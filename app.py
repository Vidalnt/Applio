import gradio as gr
import sys
import os
import logging

# Constants
DEFAULT_PORT = 6969
MAX_PORT_ATTEMPTS = 10

# Set up logging
logging.getLogger("uvicorn").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

# Add current directory to sys.path
now_dir = os.getcwd()
sys.path.append(now_dir)

# Zluda hijack
import rvc.lib.zluda

# Import Tabs
from tabs.inference.inference import inference_tab
from tabs.train.train import train_tab
from tabs.extra.extra import extra_tab
from tabs.report.report import report_tab
from tabs.download.download import download_tab
from tabs.tts.tts import tts_tab
from tabs.voice_blender.voice_blender import voice_blender_tab
from tabs.plugins.plugins import plugins_tab
from tabs.settings.settings import settings_tab

# Run prerequisites
from core import run_prerequisites_script

run_prerequisites_script(
    pretraineds_v1_f0=False,
    pretraineds_v1_nof0=False,
    pretraineds_v2_f0=True,
    pretraineds_v2_nof0=False,
    models=True,
    exe=True,
)

# Initialize i18n
from assets.i18n.i18n import I18nAuto

i18n = I18nAuto()

# Start Discord presence if enabled
from tabs.settings.sections.presence import load_config_presence

if load_config_presence():
    from assets.discord_presence import RPCManager

    RPCManager.start_presence()

# Check installation
import assets.installation_checker as installation_checker

installation_checker.check_installation()

# Start Flask server if enabled
from assets.flask.server import start_flask, load_config_flask

if load_config_flask():
    print("Starting Flask server")
    start_flask()

# Load theme
import assets.themes.loadThemes as loadThemes

my_applio = loadThemes.load_theme() or "ParityError/Interstellar"

# Define Gradio interface
with gr.Blocks(
    theme=my_applio, title="Applio", css="footer{display:none !important}"
) as Applio:
    gr.Markdown("# Applio")
    gr.Markdown(
        i18n(
            "A simple, high-quality voice conversion tool focused on ease of use and performance."
        )
    )
    gr.Markdown(
        i18n(
            "[Support](https://discord.gg/urxFjYmYYh) — [Discord Bot](https://discord.com/oauth2/authorize?client_id=1144714449563955302&permissions=1376674695271&scope=bot%20applications.commands) — [GitHub](https://github.com/IAHispano/Applio)"
        )
    )
    with gr.Tab(i18n("Inference")):
        inference_tab()

    with gr.Tab(i18n("Training")):
        train_tab()

    with gr.Tab(i18n("TTS")):
        tts_tab()

    with gr.Tab(i18n("Voice Blender")):
        voice_blender_tab()

    with gr.Tab(i18n("Plugins")):
        plugins_tab()

    with gr.Tab(i18n("Download")):
        download_tab()

    with gr.Tab(i18n("Report a Bug")):
        report_tab()

    with gr.Tab(i18n("Extra")):
        extra_tab()

    with gr.Tab(i18n("Settings")):
        settings_tab()


def launch_gradio(port):
    Applio.launch(
        favicon_path="assets/ICON.ico",
        share="--share" in sys.argv,
        inbrowser="--open" in sys.argv,
        server_port=port,
    )


def get_port_from_args():
    if "--port" in sys.argv:
        port_index = sys.argv.index("--port") + 1
        if port_index < len(sys.argv):
            return int(sys.argv[port_index])
    return DEFAULT_PORT


if __name__ == "__main__":
    port = get_port_from_args()
    for _ in range(MAX_PORT_ATTEMPTS):
        try:
            launch_gradio(port)
            break
        except OSError:
            print(
                f"Failed to launch on port {port}, trying again on port {port - 1}..."
            )
            port -= 1
        except Exception as error:
            print(f"An error occurred launching Gradio: {error}")
            break

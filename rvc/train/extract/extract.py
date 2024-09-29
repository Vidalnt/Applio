import os
import sys
import glob
import time
import tqdm
import torch
import torchcrepe
import numpy as np
import concurrent.futures
import multiprocessing as mp
import json
import shutil
from distutils.util import strtobool

now_dir = os.getcwd()
sys.path.append(os.path.join(now_dir))

# Zluda hijack
import rvc.lib.zluda

from rvc.lib.utils import load_audio, load_embedding
from rvc.train.extract.preparing_files import generate_config, generate_filelist
from rvc.lib.predictors.RMVPE import RMVPE0Predictor
from rvc.configs.config import Config

# Load config
config = Config()

mp.set_start_method("spawn", force=True)


class FeatureInput:
    """Class for F0 extraction."""

    def __init__(self, sample_rate=16000, hop_size=160, device="cpu"):
        self.fs = sample_rate
        self.hop = hop_size
        self.f0_bin = 256
        self.f0_max = 1100.0
        self.f0_min = 50.0
        self.f0_mel_min = 1127 * np.log(1 + self.f0_min / 700)
        self.f0_mel_max = 1127 * np.log(1 + self.f0_max / 700)
        self.device = device
        self.model_rmvpe = None

    def compute_f0(self, np_arr, f0_method, hop_length):
        """Extract F0 using the specified method."""
        if f0_method == "crepe":
            return self.get_crepe(np_arr, hop_length)
        elif f0_method == "rmvpe":
            return self.model_rmvpe.infer_from_audio(np_arr, thred=0.03)
        else:
            raise ValueError(f"Unknown F0 method: {f0_method}")

    def get_crepe(self, x, hop_length):
        """Extract F0 using CREPE."""
        audio = torch.from_numpy(x.astype(np.float32)).to(self.device)
        audio /= torch.quantile(torch.abs(audio), 0.999)
        audio = audio.unsqueeze(0)
        pitch = torchcrepe.predict(
            audio,
            self.fs,
            hop_length,
            self.f0_min,
            self.f0_max,
            "full",
            batch_size=hop_length * 2,
            device=audio.device,
            pad=True,
        )
        source = pitch.squeeze(0).cpu().float().numpy()
        source[source < 0.001] = np.nan
        target = np.interp(
            np.arange(0, len(source) * (x.size // self.hop), len(source))
            / (x.size // self.hop),
            np.arange(0, len(source)),
            source,
        )
        return np.nan_to_num(target)

    def coarse_f0(self, f0):
        """Convert F0 to coarse F0."""
        f0_mel = 1127 * np.log(1 + f0 / 700)
        f0_mel = np.clip(
            (f0_mel - self.f0_mel_min)
            * (self.f0_bin - 2)
            / (self.f0_mel_max - self.f0_mel_min)
            + 1,
            1,
            self.f0_bin - 1,
        )
        return np.rint(f0_mel).astype(int)

    def process_file(self, file_info, f0_method, hop_length):
        """Process a single audio file for F0 extraction."""
        inp_path, opt_path1, opt_path2, _ = file_info

        if os.path.exists(opt_path1) and os.path.exists(opt_path2):
            return

        try:
            np_arr = load_audio(inp_path, 16000)
            feature_pit = self.compute_f0(np_arr, f0_method, hop_length)
            np.save(opt_path2, feature_pit, allow_pickle=False)
            coarse_pit = self.coarse_f0(feature_pit)
            np.save(opt_path1, coarse_pit, allow_pickle=False)
        except Exception as error:
            print(
                f"An error occurred extracting file {inp_path} on {self.device}: {error}"
            )

    def process_files(
        self, files, f0_method, hop_length, device_num, device, n_threads
    ):
        """Process multiple files."""
        self.device = device
        if f0_method == "rmvpe":
            self.model_rmvpe = RMVPE0Predictor(
                os.path.join("rvc", "models", "predictors", "rmvpe.pt"),
                is_half=False,
                device=device,
            )
        else:
            n_threads = 1

        n_threads = 1 if n_threads == 0 else n_threads

        def process_file_wrapper(file_info):
            self.process_file(file_info, f0_method, hop_length)

        with tqdm.tqdm(total=len(files), leave=True, position=device_num) as pbar:
            # using multi-threading
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=n_threads
            ) as executor:
                futures = [
                    executor.submit(process_file_wrapper, file_info)
                    for file_info in files
                ]
                for future in concurrent.futures.as_completed(futures):
                    pbar.update(1)


def run_pitch_extraction(files, devices, f0_method, hop_length, num_processes):
    devices_str = ", ".join(devices)
    print(
        f"Starting pitch extraction with {num_processes} cores on {devices_str} using {f0_method}..."
    )
    start_time = time.time()
    fe = FeatureInput()
    # split the task between devices
    ps = []
    num_devices = len(devices)
    for i, device in enumerate(devices):
        p = mp.Process(
            target=fe.process_files,
            args=(
                files[i::num_devices],
                f0_method,
                hop_length,
                i,
                device,
                num_processes // num_devices,
            ),
        )
        ps.append(p)
        p.start()
    for i, device in enumerate(devices):
        ps[i].join()

    elapsed_time = time.time() - start_time
    print(f"Pitch extraction completed in {elapsed_time:.2f} seconds.")


def process_file_embedding(
    files, version, embedder_model, embedder_model_custom, device_num, device, n_threads
):
    dtype = torch.float16 if config.is_half and "cuda" in device else torch.float32
    model = load_embedding(embedder_model, embedder_model_custom).to(dtype).to(device)
    n_threads = 1 if n_threads == 0 else n_threads

    def process_file_embedding_wrapper(file_info):
        wav_file_path, _, _, out_file_path = file_info
        if os.path.exists(out_file_path):
            return
        feats = torch.from_numpy(load_audio(wav_file_path, 16000)).to(dtype).to(device)
        feats = feats.view(1, -1)
        with torch.no_grad():
            feats = model(feats)["last_hidden_state"]
            feats = (
                model.final_proj(feats[0]).unsqueeze(0) if version == "v1" else feats
            )
        feats = feats.squeeze(0).float().cpu().numpy()
        if not np.isnan(feats).any():
            np.save(out_file_path, feats, allow_pickle=False)
        else:
            print(f"{file} contains NaN values and will be skipped.")

    with tqdm.tqdm(total=len(files), leave=True, position=device_num) as pbar:
        # using multi-threading
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_threads) as executor:
            futures = [
                executor.submit(process_file_embedding_wrapper, file_info)
                for file_info in files
            ]
            for future in concurrent.futures.as_completed(futures):
                pbar.update(1)


def run_embedding_extraction(
    files, devices, version, embedder_model, embedder_model_custom
):
    start_time = time.time()
    devices_str = ", ".join(devices)
    print(
        f"Starting embedding extraction with {num_processes} cores on {devices_str}..."
    )
    # split the task between devices
    ps = []
    num_devices = len(devices)
    for i, device in enumerate(devices):
        p = mp.Process(
            target=process_file_embedding,
            args=(
                files[i::num_devices],
                version,
                embedder_model,
                embedder_model_custom,
                i,
                device,
                num_processes // num_devices,
            ),
        )
        ps.append(p)
        p.start()
    for i, device in enumerate(devices):
        ps[i].join()
    elapsed_time = time.time() - start_time
    print(f"Embedding extraction completed in {elapsed_time:.2f} seconds.")


if __name__ == "__main__":

    exp_dir = sys.argv[1]
    f0_method = sys.argv[2]
    hop_length = int(sys.argv[3])
    num_processes = int(sys.argv[4])
    gpus = sys.argv[5]
    version = sys.argv[6]
    pitch_guidance = sys.argv[7]
    sample_rate = sys.argv[8]
    embedder_model = sys.argv[9]
    embedder_model_custom = sys.argv[10] if len(sys.argv) > 10 else None

    # prep
    wav_path = os.path.join(exp_dir, "sliced_audios_16k")
    os.makedirs(os.path.join(exp_dir, "f0"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "f0_voiced"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, version + "_extracted"), exist_ok=True)
    # write to model_info.json
    chosen_embedder_model = (
        embedder_model_custom if embedder_model_custom else embedder_model
    )

    file_path = os.path.join(exp_dir, "model_info.json")
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            data = json.load(f)
    else:
        data = {}
    data.update(
        {
            "embedder_model": chosen_embedder_model,
        }
    )
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)

    files = []
    for file in glob.glob(os.path.join(wav_path, "*.wav")):
        file_name = os.path.basename(file)
        file_info = [
            file,  # full path to sliced 16k wav
            os.path.join(exp_dir, "f0", file_name + ".npy"),
            os.path.join(exp_dir, "f0_voiced", file_name + ".npy"),
            os.path.join(
                exp_dir, version + "_extracted", file_name.replace("wav", "npy")
            ),
        ]
        files.append(file_info)

    devices = ["cpu"] if gpus == "-" else [f"cuda:{idx}" for idx in gpus.split("-")]
    # Run Pitch Extraction
    run_pitch_extraction(files, devices, f0_method, hop_length, num_processes)

    # Run Embedding Extraction
    run_embedding_extraction(
        files, devices, version, embedder_model, embedder_model_custom
    )

    # Run Preparing Files
    generate_config(version, sample_rate, exp_dir)
    generate_filelist(pitch_guidance, exp_dir, version, sample_rate)

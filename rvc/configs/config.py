import torch
import json
import os

version_config_paths = [
    os.path.join("v1", "32000.json"),
    os.path.join("v1", "40000.json"),
    os.path.join("v1", "44100.json"),
    os.path.join("v1", "48000.json"),
    os.path.join("v2", "48000.json"),
    os.path.join("v2", "40000.json"),
    os.path.join("v2", "44100.json"),
    os.path.join("v2", "32000.json"),
]


def singleton(cls):
    instances = {}

    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return get_instance


@singleton
class Config:
    def __init__(self):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.is_half = self.device != "cpu"
        self.gpu_name = (
            torch.cuda.get_device_name(int(self.device.split(":")[-1]))
            if self.device.startswith("cuda")
            else None
        )
        self.json_config = self.load_config_json()
        self.gpu_mem = None
        self.x_pad, self.x_query, self.x_center, self.x_max = self.device_config()

    def load_config_json(self) -> dict:
        configs = {}
        for config_file in version_config_paths:
            config_path = os.path.join("rvc", "configs", config_file)
            with open(config_path, "r") as f:
                configs[config_file] = json.load(f)
        return configs

    def set_precision(self, precision):
        if precision not in ["fp32", "fp16"]:
            raise ValueError("Invalid precision type. Must be 'fp32' or 'fp16'.")

        fp16_run_value = precision == "fp16"
        preprocess_target_version = "3.7" if precision == "fp16" else "3.0"
        preprocess_path = os.path.join(
            os.path.dirname(__file__),
            os.pardir,
            "rvc",
            "train",
            "preprocess",
            "preprocess.py",
        )

        for config_path in version_config_paths:
            full_config_path = os.path.join("rvc", "configs", config_path)
            try:
                with open(full_config_path, "r") as f:
                    config = json.load(f)
                config["train"]["fp16_run"] = fp16_run_value
                with open(full_config_path, "w") as f:
                    json.dump(config, f, indent=4)
            except FileNotFoundError:
                print(f"File not found: {full_config_path}")

        if os.path.exists(preprocess_path):
            with open(preprocess_path, "r") as f:
                preprocess_content = f.read()
            preprocess_content = preprocess_content.replace(
                "3.0" if precision == "fp16" else "3.7", preprocess_target_version
            )
            with open(preprocess_path, "w") as f:
                f.write(preprocess_content)

        return f"Overwritten preprocess and config.json to use {precision}."

    def get_precision(self):
        if not version_config_paths:
            raise FileNotFoundError("No configuration paths provided.")

        full_config_path = os.path.join("rvc", "configs", version_config_paths[0])
        try:
            with open(full_config_path, "r") as f:
                config = json.load(f)
            fp16_run_value = config["train"].get("fp16_run", False)
            precision = "fp16" if fp16_run_value else "fp32"
            return precision
        except FileNotFoundError:
            print(f"File not found: {full_config_path}")
            return None

    def device_config(self) -> tuple:
        if self.device.startswith("cuda"):
            self.set_cuda_config()
        else:
            self.device = "cpu"
            self.is_half = False
            self.set_precision("fp32")

        # Configuration for 6GB GPU memory
        x_pad, x_query, x_center, x_max = (
            (3, 10, 60, 65) if self.is_half else (1, 6, 38, 41)
        )
        if self.gpu_mem is not None and self.gpu_mem <= 4:
            # Configuration for 5GB GPU memory
            x_pad, x_query, x_center, x_max = (1, 5, 30, 32)

        return x_pad, x_query, x_center, x_max

    def set_cuda_config(self):
        i_device = int(self.device.split(":")[-1])
        self.gpu_name = torch.cuda.get_device_name(i_device)
        low_end_gpus = ["16", "P40", "P10", "1060", "1070", "1080"]
        if (
            any(gpu in self.gpu_name for gpu in low_end_gpus)
            and "V100" not in self.gpu_name.upper()
        ):
            self.is_half = False
            self.set_precision("fp32")

        self.gpu_mem = torch.cuda.get_device_properties(i_device).total_memory // (
            1024**3
        )


def max_vram_gpu(gpu):
    if torch.cuda.is_available():
        gpu_properties = torch.cuda.get_device_properties(gpu)
        total_memory_gb = round(gpu_properties.total_memory / 1024 / 1024 / 1024)
        return total_memory_gb
    else:
        return "8"


def get_gpu_info():
    ngpu = torch.cuda.device_count()
    gpu_infos = []
    if torch.cuda.is_available() or ngpu != 0:
        for i in range(ngpu):
            gpu_name = torch.cuda.get_device_name(i)
            mem = int(
                torch.cuda.get_device_properties(i).total_memory / 1024 / 1024 / 1024
                + 0.4
            )
            gpu_infos.append(f"{i}: {gpu_name} ({mem} GB)")
    if len(gpu_infos) > 0:
        gpu_info = "\n".join(gpu_infos)
    else:
        gpu_info = "Unfortunately, there is no compatible GPU available to support your training."
    return gpu_info


def get_number_of_gpus():
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        return "-".join(map(str, range(num_gpus)))
    else:
        return "-"

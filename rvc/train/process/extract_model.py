import os, sys
import torch
import hashlib
import datetime
from collections import OrderedDict
import json

now_dir = os.getcwd()
sys.path.append(now_dir)


def replace_keys_in_dict(d, old_key_part, new_key_part):
    if isinstance(d, OrderedDict):
        updated_dict = OrderedDict()
    else:
        updated_dict = {}
    for key, value in d.items():
        new_key = key.replace(old_key_part, new_key_part)
        if isinstance(value, dict):
            value = replace_keys_in_dict(value, old_key_part, new_key_part)
        updated_dict[new_key] = value
    return updated_dict


def extract_model(
    ckpt,
    sr,
    pitch_guidance,
    name,
    model_dir,
    epoch,
    step,
    version,
    hps,
    overtrain_info,
    vocoder,
):
    try:
        print(f"Saved model '{model_dir}' (epoch {epoch} and step {step})")

        model_dir_path = os.path.dirname(model_dir)
        os.makedirs(model_dir_path, exist_ok=True)

        if "best_epoch" in model_dir:
            pth_file = f"{name}_{epoch}e_{step}s_best_epoch.pth"
        else:
            pth_file = f"{name}_{epoch}e_{step}s.pth"

        pth_file_old_version_path = os.path.join(
            model_dir_path, f"{pth_file}_old_version.pth"
        )

        model_dir_path = os.path.dirname(model_dir)
        if os.path.exists(os.path.join(model_dir_path, "model_info.json")):
            with open(os.path.join(model_dir_path, "model_info.json"), "r") as f:
                data = json.load(f)
                dataset_lenght = data.get("total_dataset_duration", None)
                embedder_model = data.get("embedder_model", None)
                speakers_id = data.get("speakers_id", 1)
        else:
            dataset_lenght = None

        with open(os.path.join(now_dir, "assets", "config.json"), "r") as f:
            data = json.load(f)
            model_author = data.get("model_author", None)

        opt = OrderedDict(
            weight={
                key: value.half() for key, value in ckpt.items() if "enc_q" not in key
            }
        )
        opt["config"] = [
            hps.data.filter_length // 2 + 1,
            32,
            hps.model.inter_channels,
            hps.model.hidden_channels,
            hps.model.filter_channels,
            hps.model.n_heads,
            hps.model.n_layers,
            hps.model.kernel_size,
            hps.model.p_dropout,
            hps.model.resblock,
            hps.model.resblock_kernel_sizes,
            hps.model.resblock_dilation_sizes,
            hps.model.upsample_rates,
            hps.model.upsample_initial_channel,
            hps.model.upsample_kernel_sizes,
            hps.model.spk_embed_dim,
            hps.model.gin_channels,
            hps.data.sample_rate,
        ]

        opt["epoch"] = epoch
        opt["step"] = step
        opt["sr"] = sr
        opt["f0"] = pitch_guidance
        opt["version"] = version
        opt["creation_date"] = datetime.datetime.now().isoformat()

        hash_input = f"{str(ckpt)} {epoch} {step} {datetime.datetime.now().isoformat()}"
        model_hash = hashlib.sha256(hash_input.encode()).hexdigest()
        opt["model_hash"] = model_hash
        opt["overtrain_info"] = overtrain_info
        opt["dataset_lenght"] = dataset_lenght
        opt["model_name"] = name
        opt["author"] = model_author
        opt["embedder_model"] = embedder_model
        opt["speakers_id"] = speakers_id
        opt["vocoder"] = vocoder

        torch.save(opt, os.path.join(model_dir_path, pth_file))

        model = torch.load(model_dir, map_location=torch.device("cpu"))
        torch.save(
            replace_keys_in_dict(
                replace_keys_in_dict(
                    model, ".parametrizations.weight.original1", ".weight_v"
                ),
                ".parametrizations.weight.original0",
                ".weight_g",
            ),
            pth_file_old_version_path,
        )
        os.remove(model_dir)
        os.rename(pth_file_old_version_path, model_dir)

    except Exception as error:
        print(f"An error occurred extracting the model: {error}")

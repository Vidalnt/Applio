import os
import re
import sys
import glob
import json
import torch
import datetime

from distutils.util import strtobool
from random import randint, shuffle
from time import time as ttime
from time import sleep
from tqdm import tqdm

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.nn import functional as F

import torch.distributed as dist
import torch.multiprocessing as mp

now_dir = os.getcwd()
sys.path.append(os.path.join(now_dir))

# Zluda hijack
import rvc.lib.zluda

from utils import (
    HParams,
    plot_spectrogram_to_numpy,
    summarize,
    load_checkpoint,
    save_checkpoint,
    latest_checkpoint_path,
    load_wav_to_torch,
)

from data_utils import (
    DistributedBucketSampler,
    TextAudioCollateMultiNSFsid,
    TextAudioLoaderMultiNSFsid,
)

from losses import (
    discriminator_loss,
    feature_loss,
    generator_loss,
    kl_loss,
)
from mel_processing import mel_spectrogram_torch, spec_to_mel_torch

from rvc.train.process.extract_model import extract_model

from rvc.lib.algorithm import commons
from rvc.lib.algorithm.discriminators import MultiPeriodDiscriminator
from rvc.lib.algorithm.discriminators import MultiPeriodDiscriminatorV2
from rvc.lib.algorithm.synthesizers import Synthesizer

# Parse command line arguments
model_name = sys.argv[1]
save_every_epoch = int(sys.argv[2])
total_epoch = int(sys.argv[3])
pretrainG = sys.argv[4]
pretrainD = sys.argv[5]
version = sys.argv[6]
gpus = sys.argv[7]
batch_size = int(sys.argv[8])
sample_rate = int(sys.argv[9])
pitch_guidance = strtobool(sys.argv[10])
save_only_latest = strtobool(sys.argv[11])
save_every_weights = strtobool(sys.argv[12])
cache_data_in_gpu = strtobool(sys.argv[13])
overtraining_detector = strtobool(sys.argv[14])
overtraining_threshold = int(sys.argv[15])
sync_graph = strtobool(sys.argv[16])

current_dir = os.getcwd()
experiment_dir = os.path.join(current_dir, "logs", model_name)
config_save_path = os.path.join(experiment_dir, "config.json")
dataset_path = os.path.join(experiment_dir, "sliced_audios")

with open(config_save_path, "r") as f:
    config = json.load(f)
config = HParams(**config)
config.data.training_files = os.path.join(experiment_dir, "filelist.txt")

torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

global_step = 0
last_loss_gen_all = 0
overtrain_save_epoch = 0
loss_gen_history = []
smoothed_loss_gen_history = []
loss_disc_history = []
smoothed_loss_disc_history = []
lowest_value = {"step": 0, "value": float("inf"), "epoch": 0}
training_file_path = os.path.join(experiment_dir, "training_data.json")

import logging

logging.getLogger("torch").setLevel(logging.ERROR)


class EpochRecorder:
    """
    Records the time elapsed per epoch.
    """

    def __init__(self):
        self.last_time = ttime()

    def record(self):
        """
        Records the elapsed time and returns a formatted string.
        """
        now_time = ttime()
        elapsed_time = now_time - self.last_time
        self.last_time = now_time
        elapsed_time = round(elapsed_time, 1)
        elapsed_time_str = str(datetime.timedelta(seconds=int(elapsed_time)))
        current_time = datetime.datetime.now().strftime("%H:%M:%S")
        return f"time={current_time} | training_speed={elapsed_time_str}"


def verify_checkpoint_shapes(checkpoint_path, model):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    checkpoint_state_dict = checkpoint["model"]
    try:
        if hasattr(model, "module"):
            model_state_dict = model.module.load_state_dict(checkpoint_state_dict)
        else:
            model_state_dict = model.load_state_dict(checkpoint_state_dict)
    except RuntimeError:
        print("The sample rate of the pretrain doesn't match the selected one")
        sys.exit(1)
    else:
        del checkpoint
        del checkpoint_state_dict
        del model_state_dict


def main():
    """
    Main function to start the training process.
    """
    global training_file_path, last_loss_gen_all, smoothed_loss_gen_history, loss_gen_history, loss_disc_history, smoothed_loss_disc_history, overtrain_save_epoch

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(randint(20000, 55555))
    # Check sample rate
    wavs = glob.glob(
        os.path.join(os.path.join(experiment_dir, "sliced_audios"), "*.wav")
    )
    if wavs:
        _, sr = load_wav_to_torch(wavs[0])
        if sr != sample_rate:
            print(
                f"Error: Pretrained model sample rate ({sample_rate} Hz) does not match dataset audio sample rate ({sr} Hz)."
            )
            os._exit(1)
    else:
        print("No wav file found.")

    if torch.cuda.is_available():
        device = torch.device("cuda")
        n_gpus = torch.cuda.device_count()
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        n_gpus = 1
    else:
        device = torch.device("cpu")
        n_gpus = 1
        print("Training with CPU, this will take a long time.")

    def start():
        """
        Starts the training process with multi-GPU support or CPU.
        """
        children = []
        pid_data = {"process_pids": []}
        with open(config_save_path, "r") as pid_file:
            try:
                existing_data = json.load(pid_file)
                pid_data.update(existing_data)
            except json.JSONDecodeError:
                pass
        with open(config_save_path, "w") as pid_file:
            for i in range(n_gpus):
                subproc = mp.Process(
                    target=run,
                    args=(
                        i,
                        n_gpus,
                        experiment_dir,
                        pretrainG,
                        pretrainD,
                        pitch_guidance,
                        custom_total_epoch,
                        custom_save_every_weights,
                        config,
                        device,
                    ),
                )
                children.append(subproc)
                subproc.start()
                pid_data["process_pids"].append(subproc.pid)
            json.dump(pid_data, pid_file, indent=4)

        for i in range(n_gpus):
            children[i].join()

    def load_from_json(file_path):
        """
        Load data from a JSON file.

        Args:
            file_path (str): The path to the JSON file.
        """
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                data = json.load(f)
                return (
                    data.get("loss_disc_history", []),
                    data.get("smoothed_loss_disc_history", []),
                    data.get("loss_gen_history", []),
                    data.get("smoothed_loss_gen_history", []),
                )
        return [], [], [], []

    def continue_overtrain_detector(training_file_path):
        """
        Continues the overtrain detector by loading the training history from a JSON file.

        Args:
            training_file_path (str): The file path of the JSON file containing the training history.
        """
        if overtraining_detector:
            if os.path.exists(training_file_path):
                (
                    loss_disc_history,
                    smoothed_loss_disc_history,
                    loss_gen_history,
                    smoothed_loss_gen_history,
                ) = load_from_json(training_file_path)

    if sync_graph:
        print(
            "Sync graph is now activated! With sync graph enabled, the model undergoes a single epoch of training. Once the graphs are synchronized, training proceeds for the previously specified number of epochs."
        )
        custom_total_epoch = 1
        custom_save_every_weights = True

        start()

        # Synchronize graphs by modifying config files
        model_config_file = os.path.join(experiment_dir, "config.json")
        rvc_config_file = os.path.join(
            now_dir, "rvc", "configs", version, str(sample_rate) + ".json"
        )
        if not os.path.exists(rvc_config_file):
            rvc_config_file = os.path.join(
                now_dir, "rvc", "configs", "v1", str(sample_rate) + ".json"
            )

        pattern = rf"{os.path.basename(model_name)}_(\d+)e_(\d+)s\.pth"

        for filename in os.listdir(experiment_dir):
            match = re.match(pattern, filename)
            if match:
                steps = int(match.group(2))

        def edit_config(config_file):
            """
            Edits the config file to synchronize graphs.

            Args:
                config_file (str): Path to the config file.
            """
            with open(config_file, "r", encoding="utf8") as json_file:
                config_data = json.load(json_file)

            config_data["train"]["log_interval"] = steps

            with open(config_file, "w", encoding="utf8") as json_file:
                json.dump(
                    config_data,
                    json_file,
                    indent=2,
                    separators=(",", ": "),
                    ensure_ascii=False,
                )

        edit_config(model_config_file)
        edit_config(rvc_config_file)

        # Clean up unnecessary files
        for root, dirs, files in os.walk(
            os.path.join(now_dir, "logs", model_name), topdown=False
        ):
            for name in files:
                file_path = os.path.join(root, name)
                file_name, file_extension = os.path.splitext(name)
                if (
                    file_extension == ".0"
                    or (file_name.startswith("D_") and file_extension == ".pth")
                    or (file_name.startswith("G_") and file_extension == ".pth")
                    or (file_name.startswith("added") and file_extension == ".index")
                ):
                    os.remove(file_path)
            for name in dirs:
                if name == "eval":
                    folder_path = os.path.join(root, name)
                    for item in os.listdir(folder_path):
                        item_path = os.path.join(folder_path, item)
                        if os.path.isfile(item_path):
                            os.remove(item_path)
                    os.rmdir(folder_path)

        print("Successfully synchronized graphs!")
        custom_total_epoch = total_epoch
        custom_save_every_weights = save_every_weights
        continue_overtrain_detector(training_file_path)
        start()
    else:
        custom_total_epoch = total_epoch
        custom_save_every_weights = save_every_weights
        continue_overtrain_detector(training_file_path)
        start()


def run(
    rank,
    n_gpus,
    experiment_dir,
    pretrainG,
    pretrainD,
    pitch_guidance,
    custom_total_epoch,
    custom_save_every_weights,
    config,
    device,
):
    """
    Runs the training loop on a specific GPU or CPU.

    Args:
        rank (int): The rank of the current process within the distributed training setup.
        n_gpus (int): The total number of GPUs available for training.
        experiment_dir (str): The directory where experiment logs and checkpoints will be saved.
        pretrainG (str): Path to the pre-trained generator model.
        pretrainD (str): Path to the pre-trained discriminator model.
        pitch_guidance (bool): Flag indicating whether to use pitch guidance during training.
        custom_total_epoch (int): The total number of epochs for training.
        custom_save_every_weights (int): The interval (in epochs) at which to save model weights.
        config (object): Configuration object containing training parameters.
        device (torch.device): The device to use for training (CPU or GPU).
    """
    global global_step, smoothed_value_gen, smoothed_value_disc

    smoothed_value_gen = 0
    smoothed_value_disc = 0

    if rank == 0:
        writer = SummaryWriter(log_dir=experiment_dir)
        writer_eval = SummaryWriter(log_dir=os.path.join(experiment_dir, "eval"))

    dist.init_process_group(
        backend="gloo",
        init_method="env://",
        world_size=n_gpus if device.type == "cuda" else 1,
        rank=rank if device.type == "cuda" else 0,
    )

    torch.manual_seed(config.train.seed)

    if torch.cuda.is_available():
        torch.cuda.set_device(rank)

    # Create datasets and dataloaders
    train_dataset = TextAudioLoaderMultiNSFsid(config.data)
    collate_fn = TextAudioCollateMultiNSFsid()
    train_sampler = DistributedBucketSampler(
        train_dataset,
        batch_size * n_gpus,
        [100, 200, 300, 400, 500, 600, 700, 800, 900],
        num_replicas=n_gpus,
        rank=rank,
        shuffle=True,
    )

    train_loader = DataLoader(
        train_dataset,
        num_workers=4,
        shuffle=False,
        pin_memory=True,
        collate_fn=collate_fn,
        batch_sampler=train_sampler,
        persistent_workers=True,
        prefetch_factor=8,
    )

    # Initialize models and optimizers
    net_g = Synthesizer(
        config.data.filter_length // 2 + 1,
        config.train.segment_size // config.data.hop_length,
        **config.model,
        use_f0=pitch_guidance == True,  # converting 1/0 to True/False
        is_half=config.train.fp16_run and device.type == "cuda",
        sr=sample_rate,
    ).to(device)

    if version == "v1":
        net_d = MultiPeriodDiscriminator(config.model.use_spectral_norm).to(device)
    else:
        net_d = MultiPeriodDiscriminatorV2(config.model.use_spectral_norm).to(device)

    optim_g = torch.optim.AdamW(
        net_g.parameters(),
        config.train.learning_rate,
        betas=config.train.betas,
        eps=config.train.eps,
    )
    optim_d = torch.optim.AdamW(
        net_d.parameters(),
        config.train.learning_rate,
        betas=config.train.betas,
        eps=config.train.eps,
    )

    # Wrap models with DDP for multi-gpu processing
    if n_gpus > 1 and device.type == "cuda":
        net_g = DDP(net_g, device_ids=[rank])
        net_d = DDP(net_d, device_ids=[rank])

    # Load checkpoint if available
    try:
        print("Starting training...")
        _, _, _, epoch_str = load_checkpoint(
            latest_checkpoint_path(experiment_dir, "D_*.pth"), net_d, optim_d
        )
        _, _, _, epoch_str = load_checkpoint(
            latest_checkpoint_path(experiment_dir, "G_*.pth"), net_g, optim_g
        )
        epoch_str += 1
        global_step = (epoch_str - 1) * len(train_loader)

    except:
        epoch_str = 1
        global_step = 0
        if pretrainG != "":
            if rank == 0:
                verify_checkpoint_shapes(pretrainG, net_g)
                print(f"Loaded pretrained (G) '{pretrainG}'")
            if hasattr(net_g, "module"):
                net_g.module.load_state_dict(
                    torch.load(pretrainG, map_location="cpu")["model"]
                )
            else:
                net_g.load_state_dict(
                    torch.load(pretrainG, map_location="cpu")["model"]
                )

        if pretrainD != "":
            if rank == 0:
                print(f"Loaded pretrained (D) '{pretrainD}'")
            if hasattr(net_d, "module"):
                net_d.module.load_state_dict(
                    torch.load(pretrainD, map_location="cpu")["model"]
                )
            else:
                net_d.load_state_dict(
                    torch.load(pretrainD, map_location="cpu")["model"]
                )

    # Initialize schedulers and scaler
    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
        optim_g, gamma=config.train.lr_decay, last_epoch=epoch_str - 2
    )
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
        optim_d, gamma=config.train.lr_decay, last_epoch=epoch_str - 2
    )

    optim_g.step()
    optim_d.step()

    scaler = GradScaler(enabled=config.train.fp16_run and device.type == "cuda")

    cache = []
    for epoch in range(epoch_str, total_epoch + 1):
        if rank == 0:
            train_and_evaluate(
                rank,
                epoch,
                config,
                [net_g, net_d],
                [optim_g, optim_d],
                scaler,
                [train_loader, None],
                [writer, writer_eval],
                cache,
                custom_save_every_weights,
                custom_total_epoch,
                device,
            )
        else:
            train_and_evaluate(
                rank,
                epoch,
                config,
                [net_g, net_d],
                [optim_g, optim_d],
                scaler,
                [train_loader, None],
                None,
                cache,
                custom_save_every_weights,
                custom_total_epoch,
                device,
            )
        scheduler_g.step()
        scheduler_d.step()


def train_and_evaluate(
    rank,
    epoch,
    hps,
    nets,
    optims,
    scaler,
    loaders,
    writers,
    cache,
    custom_save_every_weights,
    custom_total_epoch,
    device,
):
    """
    Trains and evaluates the model for one epoch.

    Args:
        rank (int): Rank of the current process.
        epoch (int): Current epoch number.
        hps (Namespace): Hyperparameters.
        nets (list): List of models [net_g, net_d].
        optims (list): List of optimizers [optim_g, optim_d].
        scaler (GradScaler): Gradient scaler for mixed precision training.
        loaders (list): List of dataloaders [train_loader, eval_loader].
        writers (list): List of TensorBoard writers [writer, writer_eval].
        cache (list): List to cache data in GPU memory.
        use_cpu (bool): Whether to use CPU for training.
    """
    global global_step, lowest_value, loss_disc, consecutive_increases_gen, consecutive_increases_disc, smoothed_value_gen, smoothed_value_disc

    if epoch == 1:
        lowest_value = {"step": 0, "value": float("inf"), "epoch": 0}
        last_loss_gen_all = 0.0
        consecutive_increases_gen = 0
        consecutive_increases_disc = 0

    net_g, net_d = nets
    optim_g, optim_d = optims
    train_loader = loaders[0] if loaders is not None else None
    if writers is not None:
        writer = writers[0]

    train_loader.batch_sampler.set_epoch(epoch)

    net_g.train()
    net_d.train()

    # Data caching
    if device.type == "cuda" and cache_data_in_gpu:
        data_iterator = cache
        if cache == []:
            for batch_idx, info in enumerate(train_loader):
                (
                    phone,
                    phone_lengths,
                    pitch,
                    pitchf,
                    spec,
                    spec_lengths,
                    wave,
                    wave_lengths,
                    sid,
                ) = info
                cache.append(
                    (
                        batch_idx,
                        (
                            phone.cuda(rank, non_blocking=True),
                            phone_lengths.cuda(rank, non_blocking=True),
                            (
                                pitch.cuda(rank, non_blocking=True)
                                if pitch_guidance
                                else None
                            ),
                            (
                                pitchf.cuda(rank, non_blocking=True)
                                if pitch_guidance
                                else None
                            ),
                            spec.cuda(rank, non_blocking=True),
                            spec_lengths.cuda(rank, non_blocking=True),
                            wave.cuda(rank, non_blocking=True),
                            wave_lengths.cuda(rank, non_blocking=True),
                            sid.cuda(rank, non_blocking=True),
                        ),
                    )
                )
        else:
            shuffle(cache)
    else:
        data_iterator = enumerate(train_loader)

    epoch_recorder = EpochRecorder()
    with tqdm(total=len(train_loader), leave=False) as pbar:
        for batch_idx, info in data_iterator:
            (
                phone,
                phone_lengths,
                pitch,
                pitchf,
                spec,
                spec_lengths,
                wave,
                wave_lengths,
                sid,
            ) = info
            if device.type == "cuda" and not cache_data_in_gpu:
                phone = phone.cuda(rank, non_blocking=True)
                phone_lengths = phone_lengths.cuda(rank, non_blocking=True)
                pitch = pitch.cuda(rank, non_blocking=True) if pitch_guidance else None
                pitchf = (
                    pitchf.cuda(rank, non_blocking=True) if pitch_guidance else None
                )
                sid = sid.cuda(rank, non_blocking=True)
                spec = spec.cuda(rank, non_blocking=True)
                spec_lengths = spec_lengths.cuda(rank, non_blocking=True)
                wave = wave.cuda(rank, non_blocking=True)
                wave_lengths = wave_lengths.cuda(rank, non_blocking=True)
            else:
                phone = phone.to(device)
                phone_lengths = phone_lengths.to(device)
                pitch = pitch.to(device) if pitch_guidance else None
                pitchf = pitchf.to(device) if pitch_guidance else None
                sid = sid.to(device)
                spec = spec.to(device)
                spec_lengths = spec_lengths.to(device)
                wave = wave.to(device)
                wave_lengths = wave_lengths.to(device)

            # Forward pass
            use_amp = config.train.fp16_run and device.type == "cuda"
            with autocast(enabled=use_amp):
                (
                    y_hat,
                    ids_slice,
                    x_mask,
                    z_mask,
                    (z, z_p, m_p, logs_p, m_q, logs_q),
                ) = net_g(phone, phone_lengths, pitch, pitchf, spec, spec_lengths, sid)
                mel = spec_to_mel_torch(
                    spec,
                    config.data.filter_length,
                    config.data.n_mel_channels,
                    config.data.sample_rate,
                    config.data.mel_fmin,
                    config.data.mel_fmax,
                )
                y_mel = commons.slice_segments(
                    mel,
                    ids_slice,
                    config.train.segment_size // config.data.hop_length,
                    dim=3,
                )
                with autocast(enabled=False):
                    y_hat_mel = mel_spectrogram_torch(
                        y_hat.float().squeeze(1),
                        config.data.filter_length,
                        config.data.n_mel_channels,
                        config.data.sample_rate,
                        config.data.hop_length,
                        config.data.win_length,
                        config.data.mel_fmin,
                        config.data.mel_fmax,
                    )
                if use_amp:
                    y_hat_mel = y_hat_mel.half()
                wave = commons.slice_segments(
                    wave,
                    ids_slice * config.data.hop_length,
                    config.train.segment_size,
                    dim=3,
                )
                y_d_hat_r, y_d_hat_g, _, _ = net_d(wave, y_hat.detach())
                with autocast(enabled=False):
                    loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(
                        y_d_hat_r, y_d_hat_g
                    )
            # Discriminator backward and update
            optim_d.zero_grad()
            scaler.scale(loss_disc).backward()
            scaler.unscale_(optim_d)
            grad_norm_d = commons.clip_grad_value(net_d.parameters(), None)
            scaler.step(optim_d)

            # Generator backward and update
            with autocast(enabled=use_amp):
                y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(wave, y_hat)
                with autocast(enabled=False):
                    loss_mel = F.l1_loss(y_mel, y_hat_mel) * config.train.c_mel
                    loss_kl = (
                        kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * config.train.c_kl
                    )
                    loss_fm = feature_loss(fmap_r, fmap_g)
                    loss_gen, losses_gen = generator_loss(y_d_hat_g)
                    loss_gen_all = loss_gen + loss_fm + loss_mel + loss_kl

                    if loss_gen_all < lowest_value["value"]:
                        lowest_value["value"] = loss_gen_all
                        lowest_value["step"] = global_step
                        lowest_value["epoch"] = epoch
                        # print(f'Lowest generator loss updated: {lowest_value["value"]} at epoch {epoch}, step {global_step}')
                        if epoch > lowest_value["epoch"]:
                            print(
                                "Alert: The lower generating loss has been exceeded by a lower loss in a subsequent epoch."
                            )

            optim_g.zero_grad()
            scaler.scale(loss_gen_all).backward()
            scaler.unscale_(optim_g)
            grad_norm_g = commons.clip_grad_value(net_g.parameters(), None)
            scaler.step(optim_g)
            scaler.update()

            # Logging and checkpointing
            if rank == 0:
                if global_step % config.train.log_interval == 0:
                    lr = optim_g.param_groups[0]["lr"]
                    if loss_mel > 75:
                        loss_mel = 75
                    if loss_kl > 9:
                        loss_kl = 9
                    scalar_dict = {
                        "loss/g/total": loss_gen_all,
                        "loss/d/total": loss_disc,
                        "learning_rate": lr,
                        "grad_norm_d": grad_norm_d,
                        "grad_norm_g": grad_norm_g,
                    }
                    scalar_dict.update(
                        {
                            "loss/g/fm": loss_fm,
                            "loss/g/mel": loss_mel,
                            "loss/g/kl": loss_kl,
                        }
                    )
                    scalar_dict.update(
                        {f"loss/g/{i}": v for i, v in enumerate(losses_gen)}
                    )
                    scalar_dict.update(
                        {f"loss/d_r/{i}": v for i, v in enumerate(losses_disc_r)}
                    )
                    scalar_dict.update(
                        {f"loss/d_g/{i}": v for i, v in enumerate(losses_disc_g)}
                    )
                    image_dict = {
                        "slice/mel_org": plot_spectrogram_to_numpy(
                            y_mel[0].data.cpu().numpy()
                        ),
                        "slice/mel_gen": plot_spectrogram_to_numpy(
                            y_hat_mel[0].data.cpu().numpy()
                        ),
                        "all/mel": plot_spectrogram_to_numpy(mel[0].data.cpu().numpy()),
                    }
                    summarize(
                        writer=writer,
                        global_step=global_step,
                        images=image_dict,
                        scalars=scalar_dict,
                    )

            global_step += 1
            pbar.update(1)

    # Save checkpoint
    model_add = []
    model_del = []
    done = False

    if rank == 0:
        # Save weights every N epochs
        if epoch % save_every_epoch == 0:
            checkpoint_suffix = f"{2333333 if save_only_latest else global_step}.pth"
            save_checkpoint(
                net_g,
                optim_g,
                config.train.learning_rate,
                epoch,
                os.path.join(experiment_dir, "G_" + checkpoint_suffix),
            )
            save_checkpoint(
                net_d,
                optim_d,
                config.train.learning_rate,
                epoch,
                os.path.join(experiment_dir, "D_" + checkpoint_suffix),
            )
            if custom_save_every_weights:
                model_add.append(
                    os.path.join(
                        experiment_dir, f"{model_name}_{epoch}e_{global_step}s.pth"
                    )
                )
        overtrain_info = ""
        # Check overtraining
        if overtraining_detector and rank == 0 and epoch > 1:
            # Add the current loss to the history
            current_loss_disc = float(loss_disc)
            loss_disc_history.append(current_loss_disc)
            # Update smoothed loss history with loss_disc
            smoothed_value_disc = update_exponential_moving_average(
                smoothed_loss_disc_history, current_loss_disc
            )
            # Check overtraining with smoothed loss_disc
            is_overtraining_disc = check_overtraining(
                smoothed_loss_disc_history, overtraining_threshold * 2
            )
            if is_overtraining_disc:
                consecutive_increases_disc += 1
            else:
                consecutive_increases_disc = 0
            # Add the current loss_gen to the history
            current_loss_gen = float(lowest_value["value"])
            loss_gen_history.append(current_loss_gen)
            # Update the smoothed loss_gen history
            smoothed_value_gen = update_exponential_moving_average(
                smoothed_loss_gen_history, current_loss_gen
            )
            # Check for overtraining with the smoothed loss_gen
            is_overtraining_gen = check_overtraining(
                smoothed_loss_gen_history, overtraining_threshold, 0.01
            )
            if is_overtraining_gen:
                consecutive_increases_gen += 1
            else:
                consecutive_increases_gen = 0
            overtrain_info = f"Smoothed loss_g {smoothed_value_gen:.3f} and loss_d {smoothed_value_disc:.3f}"
            # Save the data in the JSON file if the epoch is divisible by save_every_epoch
            if epoch % save_every_epoch == 0:
                save_to_json(
                    training_file_path,
                    loss_disc_history,
                    smoothed_loss_disc_history,
                    loss_gen_history,
                    smoothed_loss_gen_history,
                )

            if (
                is_overtraining_gen
                and consecutive_increases_gen == overtraining_threshold
                or is_overtraining_disc
                and consecutive_increases_disc == overtraining_threshold * 2
            ):
                print(
                    f"Overtraining detected at epoch {epoch} with smoothed loss_g {smoothed_value_gen:.3f} and loss_d {smoothed_value_disc:.3f}"
                )
                done = True
            else:
                print(
                    f"New best epoch {epoch} with smoothed loss_g {smoothed_value_gen:.3f} and loss_d {smoothed_value_disc:.3f}"
                )
                old_model_files = glob.glob(
                    os.path.join(experiment_dir, f"{model_name}_*e_*s_best_epoch.pth")
                )
                for file in old_model_files:
                    model_del.append(file)
                model_add.append(
                    os.path.join(
                        experiment_dir,
                        f"{model_name}_{epoch}e_{global_step}s_best_epoch.pth",
                    )
                )

        # Check completion
        if epoch >= custom_total_epoch:
            lowest_value_rounded = float(lowest_value["value"])
            lowest_value_rounded = round(lowest_value_rounded, 3)
            print(
                f"Training has been successfully completed with {epoch} epoch, {global_step} steps and {round(loss_gen_all.item(), 3)} loss gen."
            )
            print(
                f"Lowest generator loss: {lowest_value_rounded} at epoch {lowest_value['epoch']}, step {lowest_value['step']}"
            )

            pid_file_path = os.path.join(experiment_dir, "config.json")
            with open(pid_file_path, "r") as pid_file:
                pid_data = json.load(pid_file)
            with open(pid_file_path, "w") as pid_file:
                pid_data.pop("process_pids", None)
                json.dump(pid_data, pid_file, indent=4)
            # Final model
            model_add.append(
                os.path.join(
                    experiment_dir, f"{model_name}_{epoch}e_{global_step}s.pth"
                )
            )
            done = True

        if model_add:
            ckpt = (
                net_g.module.state_dict()
                if hasattr(net_g, "module")
                else net_g.state_dict()
            )
            for m in model_add:
                if not os.path.exists(m):
                    extract_model(
                        ckpt=ckpt,
                        sr=sample_rate,
                        pitch_guidance=pitch_guidance
                        == True,  # converting 1/0 to True/False,
                        name=model_name,
                        model_dir=m,
                        epoch=epoch,
                        step=global_step,
                        version=version,
                        hps=hps,
                        overtrain_info=overtrain_info,
                    )
        # Clean-up old best epochs
        for m in model_del:
            os.remove(m)

        # Print training progress
        lowest_value_rounded = float(lowest_value["value"])
        lowest_value_rounded = round(lowest_value_rounded, 3)

        record = f"{model_name} | epoch={epoch} | step={global_step} | {epoch_recorder.record()}"
        if epoch > 1:
            record = (
                record
                + f" | lowest_value={lowest_value_rounded} (epoch {lowest_value['epoch']} and step {lowest_value['step']})"
            )

        if overtraining_detector:
            remaining_epochs_gen = overtraining_threshold - consecutive_increases_gen
            remaining_epochs_disc = (
                overtraining_threshold * 2 - consecutive_increases_disc
            )
            record = (
                record
                + f" | Number of epochs remaining for overtraining: g/total: {remaining_epochs_gen} d/total: {remaining_epochs_disc} | smoothed_loss_gen={smoothed_value_gen:.3f} | smoothed_loss_disc={smoothed_value_disc:.3f}"
            )
        print(record)
        last_loss_gen_all = loss_gen_all

        if done:
            os._exit(2333333)


def check_overtraining(smoothed_loss_history, threshold, epsilon=0.004):
    """
    Checks for overtraining based on the smoothed loss history.

    Args:
        smoothed_loss_history (list): List of smoothed losses for each epoch.
        threshold (int): Number of consecutive epochs with insignificant changes or increases to consider overtraining.
        epsilon (float): The maximum change considered insignificant.
    """
    if len(smoothed_loss_history) < threshold + 1:
        return False

    for i in range(-threshold, -1):
        if smoothed_loss_history[i + 1] > smoothed_loss_history[i]:
            return True
        if abs(smoothed_loss_history[i + 1] - smoothed_loss_history[i]) >= epsilon:
            return False
    return True


def update_exponential_moving_average(
    smoothed_loss_history, new_value, smoothing=0.987
):
    """
    Updates the exponential moving average with a new value.

    Args:
        smoothed_loss_history (list): List of smoothed values.
        new_value (float): New value to be added.
        smoothing (float): Smoothing factor.
    """
    if smoothed_loss_history:
        smoothed_value = (
            smoothing * smoothed_loss_history[-1] + (1 - smoothing) * new_value
        )
    else:
        smoothed_value = new_value
    smoothed_loss_history.append(smoothed_value)
    return smoothed_value


def save_to_json(
    file_path,
    loss_disc_history,
    smoothed_loss_disc_history,
    loss_gen_history,
    smoothed_loss_gen_history,
):
    """
    Save the training history to a JSON file.
    """
    data = {
        "loss_disc_history": loss_disc_history,
        "smoothed_loss_disc_history": smoothed_loss_disc_history,
        "loss_gen_history": loss_gen_history,
        "smoothed_loss_gen_history": smoothed_loss_gen_history,
    }
    with open(file_path, "w") as f:
        json.dump(data, f)


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    main()

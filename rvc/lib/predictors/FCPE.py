from typing import Union

import torch.nn.functional as F
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm
from torchaudio.transforms import Resample
import os
import librosa
import soundfile as sf
import torch.utils.data
from librosa.filters import mel as librosa_mel_fn
import math
from functools import partial

from einops import rearrange, repeat
from local_attention import LocalAttention
from torch import nn

os.environ["LRU_CACHE_CAPACITY"] = "3"


def load_wav_to_torch(full_path, target_sr=None, return_empty_on_exception=False):
    """Loads wav file to torch tensor."""
    try:
        data, sample_rate = sf.read(full_path, always_2d=True)
    except Exception as error:
        print(f"An error occurred loading {full_path}: {error}")
        if return_empty_on_exception:
            return [], sample_rate or target_sr or 48000
        else:
            raise

    data = data[:, 0] if len(data.shape) > 1 else data
    assert len(data) > 2

    # Normalize data
    max_mag = (
        -np.iinfo(data.dtype).min
        if np.issubdtype(data.dtype, np.integer)
        else max(np.amax(data), -np.amin(data))
    )
    max_mag = (
        (2**31) + 1 if max_mag > (2**15) else ((2**15) + 1 if max_mag > 1.01 else 1.0)
    )
    data = torch.FloatTensor(data.astype(np.float32)) / max_mag

    # Handle exceptions and resample
    if (torch.isinf(data) | torch.isnan(data)).any() and return_empty_on_exception:
        return [], sample_rate or target_sr or 48000
    if target_sr is not None and sample_rate != target_sr:
        data = torch.from_numpy(
            librosa.core.resample(
                data.numpy(), orig_sr=sample_rate, target_sr=target_sr
            )
        )
        sample_rate = target_sr

    return data, sample_rate


def dynamic_range_compression(x, C=1, clip_val=1e-5):
    return np.log(np.clip(x, a_min=clip_val, a_max=None) * C)


def dynamic_range_decompression(x, C=1):
    return np.exp(x) / C


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression_torch(x, C=1):
    return torch.exp(x) / C


class STFT:
    def __init__(
        self,
        sr=22050,
        n_mels=80,
        n_fft=1024,
        win_size=1024,
        hop_length=256,
        fmin=20,
        fmax=11025,
        clip_val=1e-5,
    ):
        self.target_sr = sr
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.win_size = win_size
        self.hop_length = hop_length
        self.fmin = fmin
        self.fmax = fmax
        self.clip_val = clip_val
        self.mel_basis = {}
        self.hann_window = {}

    def get_mel(self, y, keyshift=0, speed=1, center=False, train=False):
        sample_rate = self.target_sr
        n_mels = self.n_mels
        n_fft = self.n_fft
        win_size = self.win_size
        hop_length = self.hop_length
        fmin = self.fmin
        fmax = self.fmax
        clip_val = self.clip_val

        factor = 2 ** (keyshift / 12)
        n_fft_new = int(np.round(n_fft * factor))
        win_size_new = int(np.round(win_size * factor))
        hop_length_new = int(np.round(hop_length * speed))

        # Optimize mel_basis and hann_window caching
        mel_basis = self.mel_basis if not train else {}
        hann_window = self.hann_window if not train else {}

        mel_basis_key = str(fmax) + "_" + str(y.device)
        if mel_basis_key not in mel_basis:
            mel = librosa_mel_fn(
                sr=sample_rate, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax
            )
            mel_basis[mel_basis_key] = torch.from_numpy(mel).float().to(y.device)

        keyshift_key = str(keyshift) + "_" + str(y.device)
        if keyshift_key not in hann_window:
            hann_window[keyshift_key] = torch.hann_window(win_size_new).to(y.device)

        # Padding and STFT
        pad_left = (win_size_new - hop_length_new) // 2
        pad_right = max(
            (win_size_new - hop_length_new + 1) // 2,
            win_size_new - y.size(-1) - pad_left,
        )
        mode = "reflect" if pad_right < y.size(-1) else "constant"
        y = torch.nn.functional.pad(y.unsqueeze(1), (pad_left, pad_right), mode=mode)
        y = y.squeeze(1)

        spec = torch.stft(
            y,
            n_fft=n_fft_new,
            hop_length=hop_length_new,
            win_length=win_size_new,
            window=hann_window[keyshift_key],
            center=center,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )
        spec = torch.sqrt(spec.real.pow(2) + spec.imag.pow(2) + (1e-9))

        # Handle keyshift and mel conversion
        if keyshift != 0:
            size = n_fft // 2 + 1
            resize = spec.size(1)
            spec = (
                F.pad(spec, (0, 0, 0, size - resize))
                if resize < size
                else spec[:, :size, :]
            )
            spec = spec * win_size / win_size_new
        spec = torch.matmul(mel_basis[mel_basis_key], spec)
        spec = dynamic_range_compression_torch(spec, clip_val=clip_val)
        return spec

    def __call__(self, audiopath):
        audio, sr = load_wav_to_torch(audiopath, target_sr=self.target_sr)
        spect = self.get_mel(audio.unsqueeze(0)).squeeze(0)
        return spect


stft = STFT()


def softmax_kernel(
    data, *, projection_matrix, is_query, normalize_data=True, eps=1e-4, device=None
):
    b, h, *_ = data.shape

    # Normalize data
    data_normalizer = (data.shape[-1] ** -0.25) if normalize_data else 1.0

    # Project data
    ratio = projection_matrix.shape[0] ** -0.5
    projection = repeat(projection_matrix, "j d -> b h j d", b=b, h=h)
    projection = projection.type_as(data)
    data_dash = torch.einsum("...id,...jd->...ij", (data_normalizer * data), projection)

    # Calculate diagonal data
    diag_data = data**2
    diag_data = torch.sum(diag_data, dim=-1)
    diag_data = (diag_data / 2.0) * (data_normalizer**2)
    diag_data = diag_data.unsqueeze(dim=-1)

    # Apply softmax
    if is_query:
        data_dash = ratio * (
            torch.exp(
                data_dash
                - diag_data
                - torch.max(data_dash, dim=-1, keepdim=True).values
            )
            + eps
        )
    else:
        data_dash = ratio * (torch.exp(data_dash - diag_data + eps))

    return data_dash.type_as(data)


def orthogonal_matrix_chunk(cols, qr_uniform_q=False, device=None):
    unstructured_block = torch.randn((cols, cols), device=device)
    q, r = torch.linalg.qr(unstructured_block.cpu(), mode="reduced")
    q, r = map(lambda t: t.to(device), (q, r))

    if qr_uniform_q:
        d = torch.diag(r, 0)
        q *= d.sign()
    return q.t()


def exists(val):
    return val is not None


def empty(tensor):
    return tensor.numel() == 0


def default(val, d):
    return val if exists(val) else d


def cast_tuple(val):
    return (val,) if not isinstance(val, tuple) else val


class PCmer(nn.Module):
    def __init__(
        self,
        num_layers,
        num_heads,
        dim_model,
        dim_keys,
        dim_values,
        residual_dropout,
        attention_dropout,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dim_model = dim_model
        self.dim_values = dim_values
        self.dim_keys = dim_keys
        self.residual_dropout = residual_dropout
        self.attention_dropout = attention_dropout

        self._layers = nn.ModuleList([_EncoderLayer(self) for _ in range(num_layers)])

    def forward(self, phone, mask=None):
        for layer in self._layers:
            phone = layer(phone, mask)
        return phone


class _EncoderLayer(nn.Module):
    def __init__(self, parent: PCmer):
        super().__init__()
        self.conformer = ConformerConvModule(parent.dim_model)
        self.norm = nn.LayerNorm(parent.dim_model)
        self.dropout = nn.Dropout(parent.residual_dropout)
        self.attn = SelfAttention(
            dim=parent.dim_model, heads=parent.num_heads, causal=False
        )

    def forward(self, phone, mask=None):
        phone = phone + (self.attn(self.norm(phone), mask=mask))
        phone = phone + (self.conformer(phone))
        return phone


def calc_same_padding(kernel_size):
    pad = kernel_size // 2
    return (pad, pad - (kernel_size + 1) % 2)


class Swish(nn.Module):
    def forward(self, x):
        return x * x.sigmoid()


class Transpose(nn.Module):
    def __init__(self, dims):
        super().__init__()
        assert len(dims) == 2, "dims must be a tuple of two dimensions"
        self.dims = dims

    def forward(self, x):
        return x.transpose(*self.dims)


class GLU(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        out, gate = x.chunk(2, dim=self.dim)
        return out * gate.sigmoid()


class DepthWiseConv1d(nn.Module):
    def __init__(self, chan_in, chan_out, kernel_size, padding):
        super().__init__()
        self.padding = padding
        self.conv = nn.Conv1d(chan_in, chan_out, kernel_size, groups=chan_in)

    def forward(self, x):
        x = F.pad(x, self.padding)
        return self.conv(x)


class ConformerConvModule(nn.Module):
    def __init__(
        self, dim, causal=False, expansion_factor=2, kernel_size=31, dropout=0.0
    ):
        super().__init__()

        inner_dim = dim * expansion_factor
        padding = calc_same_padding(kernel_size) if not causal else (kernel_size - 1, 0)

        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            Transpose((1, 2)),
            nn.Conv1d(dim, inner_dim * 2, 1),
            GLU(dim=1),
            DepthWiseConv1d(
                inner_dim, inner_dim, kernel_size=kernel_size, padding=padding
            ),
            Swish(),
            nn.Conv1d(inner_dim, dim, 1),
            Transpose((1, 2)),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


def linear_attention(q, k, v):
    if v is None:
        out = torch.einsum("...ed,...nd->...ne", k, q)
        return out
    else:
        k_cumsum = k.sum(dim=-2)
        D_inv = 1.0 / (torch.einsum("...nd,...d->...n", q, k_cumsum.type_as(q)) + 1e-8)
        context = torch.einsum("...nd,...ne->...de", k, v)
        out = torch.einsum("...de,...nd,...n->...ne", context, q, D_inv)
        return out


def gaussian_orthogonal_random_matrix(
    nb_rows, nb_columns, scaling=0, qr_uniform_q=False, device=None
):
    nb_full_blocks = int(nb_rows / nb_columns)
    block_list = []

    for _ in range(nb_full_blocks):
        q = orthogonal_matrix_chunk(
            nb_columns, qr_uniform_q=qr_uniform_q, device=device
        )
        block_list.append(q)

    remaining_rows = nb_rows - nb_full_blocks * nb_columns
    if remaining_rows > 0:
        q = orthogonal_matrix_chunk(
            nb_columns, qr_uniform_q=qr_uniform_q, device=device
        )
        block_list.append(q[:remaining_rows])

    final_matrix = torch.cat(block_list)

    if scaling == 0:
        multiplier = torch.randn((nb_rows, nb_columns), device=device).norm(dim=1)
    elif scaling == 1:
        multiplier = math.sqrt((float(nb_columns))) * torch.ones(
            (nb_rows,), device=device
        )
    else:
        raise ValueError(f"Invalid scaling {scaling}")

    return torch.diag(multiplier) @ final_matrix


class FastAttention(nn.Module):
    def __init__(
        self,
        dim_heads,
        nb_features=None,
        ortho_scaling=0,
        causal=False,
        generalized_attention=False,
        kernel_fn=nn.ReLU(),
        qr_uniform_q=False,
        no_projection=False,
    ):
        super().__init__()
        nb_features = default(nb_features, int(dim_heads * math.log(dim_heads)))

        self.dim_heads = dim_heads
        self.nb_features = nb_features
        self.ortho_scaling = ortho_scaling

        self.create_projection = partial(
            gaussian_orthogonal_random_matrix,
            nb_rows=self.nb_features,
            nb_columns=dim_heads,
            scaling=ortho_scaling,
            qr_uniform_q=qr_uniform_q,
        )
        projection_matrix = self.create_projection()
        self.register_buffer("projection_matrix", projection_matrix)

        self.generalized_attention = generalized_attention
        self.kernel_fn = kernel_fn
        self.no_projection = no_projection
        self.causal = causal

    @torch.no_grad()
    def redraw_projection_matrix(self):
        projections = self.create_projection()
        self.projection_matrix.copy_(projections)
        del projections

    def forward(self, q, k, v):
        device = q.device

        if self.no_projection:
            q = q.softmax(dim=-1)
            k = torch.exp(k) if self.causal else k.softmax(dim=-2)
        else:
            create_kernel = partial(
                softmax_kernel, projection_matrix=self.projection_matrix, device=device
            )
            q = create_kernel(q, is_query=True)
            k = create_kernel(k, is_query=False)

        attn_fn = linear_attention if not self.causal else self.causal_linear_fn

        if v is None:
            out = attn_fn(q, k, None)
            return out
        else:
            out = attn_fn(q, k, v)
            return out


class SelfAttention(nn.Module):
    def __init__(
        self,
        dim,
        causal=False,
        heads=8,
        dim_head=64,
        local_heads=0,
        local_window_size=256,
        nb_features=None,
        feature_redraw_interval=1000,
        generalized_attention=False,
        kernel_fn=nn.ReLU(),
        qr_uniform_q=False,
        dropout=0.0,
        no_projection=False,
    ):
        super().__init__()
        assert dim % heads == 0, "dimension must be divisible by number of heads"
        dim_head = default(dim_head, dim // heads)
        inner_dim = dim_head * heads
        self.fast_attention = FastAttention(
            dim_head,
            nb_features,
            causal=causal,
            generalized_attention=generalized_attention,
            kernel_fn=kernel_fn,
            qr_uniform_q=qr_uniform_q,
            no_projection=no_projection,
        )

        self.heads = heads
        self.global_heads = heads - local_heads
        self.local_attn = (
            LocalAttention(
                window_size=local_window_size,
                causal=causal,
                autopad=True,
                dropout=dropout,
                look_forward=int(not causal),
                rel_pos_emb_config=(dim_head, local_heads),
            )
            if local_heads > 0
            else None
        )

        self.to_q = nn.Linear(dim, inner_dim)
        self.to_k = nn.Linear(dim, inner_dim)
        self.to_v = nn.Linear(dim, inner_dim)
        self.to_out = nn.Linear(inner_dim, dim)
        self.dropout = nn.Dropout(dropout)

    @torch.no_grad()
    def redraw_projection_matrix(self):
        self.fast_attention.redraw_projection_matrix()

    def forward(
        self,
        x,
        context=None,
        mask=None,
        context_mask=None,
        name=None,
        inference=False,
        **kwargs,
    ):
        _, _, _, h, gh = *x.shape, self.heads, self.global_heads

        cross_attend = exists(context)
        context = default(context, x)
        context_mask = default(context_mask, mask) if not cross_attend else context_mask
        q, k, v = self.to_q(x), self.to_k(context), self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v))
        (q, lq), (k, lk), (v, lv) = map(lambda t: (t[:, :gh], t[:, gh:]), (q, k, v))

        attn_outs = []
        if not empty(q):
            if exists(context_mask):
                global_mask = context_mask[:, None, :, None]
                v.masked_fill_(~global_mask, 0.0)
            if cross_attend:
                pass  # TODO: Implement cross-attention
            else:
                out = self.fast_attention(q, k, v)
            attn_outs.append(out)

        if not empty(lq):
            assert (
                not cross_attend
            ), "local attention is not compatible with cross attention"
            out = self.local_attn(lq, lk, lv, input_mask=mask)
            attn_outs.append(out)

        out = torch.cat(attn_outs, dim=1)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.to_out(out)
        return self.dropout(out)


def l2_regularization(model, l2_alpha):
    l2_loss = []
    for module in model.modules():
        if type(module) is nn.Conv2d:
            l2_loss.append((module.weight**2).sum() / 2.0)
    return l2_alpha * sum(l2_loss)


class FCPE(nn.Module):
    def __init__(
        self,
        input_channel=128,
        out_dims=360,
        n_layers=12,
        n_chans=512,
        use_siren=False,
        use_full=False,
        loss_mse_scale=10,
        loss_l2_regularization=False,
        loss_l2_regularization_scale=1,
        loss_grad1_mse=False,
        loss_grad1_mse_scale=1,
        f0_max=1975.5,
        f0_min=32.70,
        confidence=False,
        threshold=0.05,
        use_input_conv=True,
    ):
        super().__init__()
        if use_siren is True:
            raise ValueError("Siren is not supported yet.")
        if use_full is True:
            raise ValueError("Full model is not supported yet.")

        self.loss_mse_scale = loss_mse_scale if (loss_mse_scale is not None) else 10
        self.loss_l2_regularization = (
            loss_l2_regularization if (loss_l2_regularization is not None) else False
        )
        self.loss_l2_regularization_scale = (
            loss_l2_regularization_scale
            if (loss_l2_regularization_scale is not None)
            else 1
        )
        self.loss_grad1_mse = loss_grad1_mse if (loss_grad1_mse is not None) else False
        self.loss_grad1_mse_scale = (
            loss_grad1_mse_scale if (loss_grad1_mse_scale is not None) else 1
        )
        self.f0_max = f0_max if (f0_max is not None) else 1975.5
        self.f0_min = f0_min if (f0_min is not None) else 32.70
        self.confidence = confidence if (confidence is not None) else False
        self.threshold = threshold if (threshold is not None) else 0.05
        self.use_input_conv = use_input_conv if (use_input_conv is not None) else True

        self.cent_table_b = torch.Tensor(
            np.linspace(
                self.f0_to_cent(torch.Tensor([f0_min]))[0],
                self.f0_to_cent(torch.Tensor([f0_max]))[0],
                out_dims,
            )
        )
        self.register_buffer("cent_table", self.cent_table_b)

        # conv in stack
        _leaky = nn.LeakyReLU()
        self.stack = nn.Sequential(
            nn.Conv1d(input_channel, n_chans, 3, 1, 1),
            nn.GroupNorm(4, n_chans),
            _leaky,
            nn.Conv1d(n_chans, n_chans, 3, 1, 1),
        )

        # transformer
        self.decoder = PCmer(
            num_layers=n_layers,
            num_heads=8,
            dim_model=n_chans,
            dim_keys=n_chans,
            dim_values=n_chans,
            residual_dropout=0.1,
            attention_dropout=0.1,
        )
        self.norm = nn.LayerNorm(n_chans)

        # out
        self.n_out = out_dims
        self.dense_out = weight_norm(nn.Linear(n_chans, self.n_out))

    def forward(
        self, mel, infer=True, gt_f0=None, return_hz_f0=False, cdecoder="local_argmax"
    ):
        if cdecoder == "argmax":
            self.cdecoder = self.cents_decoder
        elif cdecoder == "local_argmax":
            self.cdecoder = self.cents_local_decoder

        x = (
            self.stack(mel.transpose(1, 2)).transpose(1, 2)
            if self.use_input_conv
            else mel
        )
        x = self.decoder(x)
        x = self.norm(x)
        x = self.dense_out(x)
        x = torch.sigmoid(x)

        if not infer:
            gt_cent_f0 = self.f0_to_cent(gt_f0)
            gt_cent_f0 = self.gaussian_blurred_cent(gt_cent_f0)
            loss_all = self.loss_mse_scale * F.binary_cross_entropy(x, gt_cent_f0)
            if self.loss_l2_regularization:
                loss_all = loss_all + l2_regularization(
                    model=self, l2_alpha=self.loss_l2_regularization_scale
                )
            x = loss_all
        if infer:
            x = self.cdecoder(x)
            x = self.cent_to_f0(x)
            x = (1 + x / 700).log() if not return_hz_f0 else x

        return x

    def cents_decoder(self, y, mask=True):
        B, N, _ = y.size()
        ci = self.cent_table[None, None, :].expand(B, N, -1)
        rtn = torch.sum(ci * y, dim=-1, keepdim=True) / torch.sum(
            y, dim=-1, keepdim=True
        )
        if mask:
            confident = torch.max(y, dim=-1, keepdim=True)[0]
            confident_mask = torch.ones_like(confident)
            confident_mask[confident <= self.threshold] = float("-INF")
            rtn = rtn * confident_mask
        return (rtn, confident) if self.confidence else rtn

    def cents_local_decoder(self, y, mask=True):
        B, N, _ = y.size()
        ci = self.cent_table[None, None, :].expand(B, N, -1)
        confident, max_index = torch.max(y, dim=-1, keepdim=True)
        local_argmax_index = torch.arange(0, 9).to(max_index.device) + (max_index - 4)
        local_argmax_index = torch.clamp(local_argmax_index, 0, self.n_out - 1)
        ci_l = torch.gather(ci, -1, local_argmax_index)
        y_l = torch.gather(y, -1, local_argmax_index)
        rtn = torch.sum(ci_l * y_l, dim=-1, keepdim=True) / torch.sum(
            y_l, dim=-1, keepdim=True
        )
        if mask:
            confident_mask = torch.ones_like(confident)
            confident_mask[confident <= self.threshold] = float("-INF")
            rtn = rtn * confident_mask
        return (rtn, confident) if self.confidence else rtn

    def cent_to_f0(self, cent):
        return 10.0 * 2 ** (cent / 1200.0)

    def f0_to_cent(self, f0):
        return 1200.0 * torch.log2(f0 / 10.0)

    def gaussian_blurred_cent(self, cents):
        mask = (cents > 0.1) & (cents < (1200.0 * np.log2(self.f0_max / 10.0)))
        B, N, _ = cents.size()
        ci = self.cent_table[None, None, :].expand(B, N, -1)
        return torch.exp(-torch.square(ci - cents) / 1250) * mask.float()


class FCPEInfer:
    def __init__(self, model_path, device=None, dtype=torch.float32):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        ckpt = torch.load(model_path, map_location=torch.device(self.device))
        self.args = DotDict(ckpt["config"])
        self.dtype = dtype
        model = FCPE(
            input_channel=self.args.model.input_channel,
            out_dims=self.args.model.out_dims,
            n_layers=self.args.model.n_layers,
            n_chans=self.args.model.n_chans,
            use_siren=self.args.model.use_siren,
            use_full=self.args.model.use_full,
            loss_mse_scale=self.args.loss.loss_mse_scale,
            loss_l2_regularization=self.args.loss.loss_l2_regularization,
            loss_l2_regularization_scale=self.args.loss.loss_l2_regularization_scale,
            loss_grad1_mse=self.args.loss.loss_grad1_mse,
            loss_grad1_mse_scale=self.args.loss.loss_grad1_mse_scale,
            f0_max=self.args.model.f0_max,
            f0_min=self.args.model.f0_min,
            confidence=self.args.model.confidence,
        )
        model.to(self.device).to(self.dtype)
        model.load_state_dict(ckpt["model"])
        model.eval()
        self.model = model
        self.wav2mel = Wav2Mel(self.args, dtype=self.dtype, device=self.device)

    @torch.no_grad()
    def __call__(self, audio, sr, threshold=0.05):
        self.model.threshold = threshold
        audio = audio[None, :]
        mel = self.wav2mel(audio=audio, sample_rate=sr).to(self.dtype)
        f0 = self.model(mel=mel, infer=True, return_hz_f0=True)
        return f0


class Wav2Mel:
    def __init__(self, args, device=None, dtype=torch.float32):
        self.sample_rate = args.mel.sampling_rate
        self.hop_size = args.mel.hop_size
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.dtype = dtype
        self.stft = STFT(
            args.mel.sampling_rate,
            args.mel.num_mels,
            args.mel.n_fft,
            args.mel.win_size,
            args.mel.hop_size,
            args.mel.fmin,
            args.mel.fmax,
        )
        self.resample_kernel = {}

    def extract_nvstft(self, audio, keyshift=0, train=False):
        mel = self.stft.get_mel(audio, keyshift=keyshift, train=train).transpose(1, 2)
        return mel

    def extract_mel(self, audio, sample_rate, keyshift=0, train=False):
        audio = audio.to(self.dtype).to(self.device)
        if sample_rate == self.sample_rate:
            audio_res = audio
        else:
            key_str = str(sample_rate)
            if key_str not in self.resample_kernel:
                self.resample_kernel[key_str] = Resample(
                    sample_rate, self.sample_rate, lowpass_filter_width=128
                )
            self.resample_kernel[key_str] = (
                self.resample_kernel[key_str].to(self.dtype).to(self.device)
            )
            audio_res = self.resample_kernel[key_str](audio)

        mel = self.extract_nvstft(
            audio_res, keyshift=keyshift, train=train
        )  # B, n_frames, bins
        n_frames = int(audio.shape[1] // self.hop_size) + 1
        mel = (
            torch.cat((mel, mel[:, -1:, :]), 1) if n_frames > int(mel.shape[1]) else mel
        )
        mel = mel[:, :n_frames, :] if n_frames < int(mel.shape[1]) else mel
        return mel

    def __call__(self, audio, sample_rate, keyshift=0, train=False):
        return self.extract_mel(audio, sample_rate, keyshift=keyshift, train=train)


class DotDict(dict):
    def __getattr__(*args):
        val = dict.get(*args)
        return DotDict(val) if type(val) is dict else val

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class F0Predictor(object):
    def compute_f0(self, wav, p_len):
        pass

    def compute_f0_uv(self, wav, p_len):
        pass


class FCPEF0Predictor(F0Predictor):
    def __init__(
        self,
        model_path,
        hop_length=512,
        f0_min=50,
        f0_max=1100,
        dtype=torch.float32,
        device=None,
        sample_rate=44100,
        threshold=0.05,
    ):
        self.fcpe = FCPEInfer(model_path, device=device, dtype=dtype)
        self.hop_length = hop_length
        self.f0_min = f0_min
        self.f0_max = f0_max
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.threshold = threshold
        self.sample_rate = sample_rate
        self.dtype = dtype
        self.name = "fcpe"

    def repeat_expand(
        self,
        content: Union[torch.Tensor, np.ndarray],
        target_len: int,
        mode: str = "nearest",
    ):
        ndim = content.ndim
        content = (
            content[None, None]
            if ndim == 1
            else content[None] if ndim == 2 else content
        )
        assert content.ndim == 3
        is_np = isinstance(content, np.ndarray)
        content = torch.from_numpy(content) if is_np else content
        results = torch.nn.functional.interpolate(content, size=target_len, mode=mode)
        results = results.numpy() if is_np else results
        return results[0, 0] if ndim == 1 else results[0] if ndim == 2 else results

    def post_process(self, x, sample_rate, f0, pad_to):
        f0 = (
            torch.from_numpy(f0).float().to(x.device)
            if isinstance(f0, np.ndarray)
            else f0
        )
        f0 = self.repeat_expand(f0, pad_to) if pad_to is not None else f0

        vuv_vector = torch.zeros_like(f0)
        vuv_vector[f0 > 0.0] = 1.0
        vuv_vector[f0 <= 0.0] = 0.0

        nzindex = torch.nonzero(f0).squeeze()
        f0 = torch.index_select(f0, dim=0, index=nzindex).cpu().numpy()
        time_org = self.hop_length / sample_rate * nzindex.cpu().numpy()
        time_frame = np.arange(pad_to) * self.hop_length / sample_rate

        vuv_vector = F.interpolate(vuv_vector[None, None, :], size=pad_to)[0][0]

        if f0.shape[0] <= 0:
            return np.zeros(pad_to), vuv_vector.cpu().numpy()
        if f0.shape[0] == 1:
            return np.ones(pad_to) * f0[0], vuv_vector.cpu().numpy()

        f0 = np.interp(time_frame, time_org, f0, left=f0[0], right=f0[-1])
        return f0, vuv_vector.cpu().numpy()

    def compute_f0(self, wav, p_len=None):
        x = torch.FloatTensor(wav).to(self.dtype).to(self.device)
        p_len = x.shape[0] // self.hop_length if p_len is None else p_len
        f0 = self.fcpe(x, sr=self.sample_rate, threshold=self.threshold)[0, :, 0]
        if torch.all(f0 == 0):
            return f0.cpu().numpy() if p_len is None else np.zeros(p_len), (
                f0.cpu().numpy() if p_len is None else np.zeros(p_len)
            )
        return self.post_process(x, self.sample_rate, f0, p_len)[0]

    def compute_f0_uv(self, wav, p_len=None):
        x = torch.FloatTensor(wav).to(self.dtype).to(self.device)
        p_len = x.shape[0] // self.hop_length if p_len is None else p_len
        f0 = self.fcpe(x, sr=self.sample_rate, threshold=self.threshold)[0, :, 0]
        if torch.all(f0 == 0):
            return f0.cpu().numpy() if p_len is None else np.zeros(p_len), (
                f0.cpu().numpy() if p_len is None else np.zeros(p_len)
            )
        return self.post_process(x, self.sample_rate, f0, p_len)

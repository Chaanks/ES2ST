from collections import defaultdict
from functools import partial

from scipy.ndimage import gaussian_filter1d
import amfm_decompy.basic_tools as basic
import amfm_decompy.pYAAPT as pYAAPT
from librosa.util import normalize
import numpy as np
import parselmouth
import torch
import torch.nn.functional as F
from tqdm import tqdm
import speechbrain as sb


MAX_WAV_VALUE = 32768.0

def extract_f0(wav, sr=16000, extractor="pyaapt", interp=False):
    # wav = wav / MAX_WAV_VALUE
    # wav = normalize(wav) * 0.95

    if extractor == "pyaapt":
        frame_length = 20.0
        pad = int(frame_length / 1000 * sr) // 2
        wav = np.pad(wav.squeeze(), (pad, pad), "constant", constant_values=0)
        signal = basic.SignalObj(wav, sr)
        pitch = pYAAPT.yaapt(
                signal,
                **{
                    'frame_length': frame_length,
                    'frame_space': 5.0,
                    'nccf_thresh1': 0.25,
                    'tda_frame_length': 25.0
                })
        pitch = pitch.samp_interp[None, None, :] if interp else pitch.samp_values[None, None, :]
        pitch = pitch[0, 0]
        return pitch

    elif extractor == "parselmouth":
        frame_length = 256/sr
        pad = int(frame_length / 1000 * sr) // 2
        wav = np.pad(wav.squeeze(), (pad, pad), "constant", constant_values=0)
        x = wav.astype(np.double)
        snd = parselmouth.Sound(values=x, sampling_frequency=sr)
        pitch  = snd.to_pitch(time_step=frame_length, pitch_floor=40, pitch_ceiling=600).selected_array['frequency']
        return pitch


def quantize_f0(speaker_to_f0, nbins, normalize, log):
    f0_all = []
    for speaker, f0 in speaker_to_f0.items():
        f0 = f0.raw_data
        if log:
            f0 = f0.log()
        mean = speaker_to_f0[speaker].mean_log if log else speaker_to_f0[speaker].mean
        std = speaker_to_f0[speaker].std_log if log else speaker_to_f0[speaker].std
        if normalize == "mean":
            f0 = f0 - mean
        elif normalize == "meanstd":
            f0 = (f0 - mean) / std
        f0_all.extend(f0.tolist())

    hist, bin_x = np.histogram(f0_all, 100000)
    cum_hist = np.cumsum(hist) / len(f0_all) * 100

    bin_offset = []
    bin_size = 100 / nbins
    threshold = bin_size
    for i in range(nbins - 1):
        index = (np.abs(cum_hist - threshold)).argmin()
        bin_offset.append(bin_x[index])
        threshold += bin_size
    bins = np.array(bin_offset)
    bins = torch.FloatTensor(bins)

    return bins


def freq2bin(f0, f0_min, f0_max, bins):
    f0 = f0.clone()
    f0[f0 < f0_min] = f0_min
    f0[f0 > f0_max] = f0_max
    f0 = torch.bucketize(f0, bins)
    return f0


def bin2freq(x, f0_min, f0_max, bins, mode):
    n_bins = len(bins) + 1
    assert x.shape[-1] == n_bins
    bins = torch.cat([torch.tensor([f0_min]), bins]).to(x.device)
    if mode == "mean":
        f0 = (x * bins).sum(-1, keepdims=True) / x.sum(-1, keepdims=True)
    elif mode == "argmax":
        idx = F.one_hot(x.argmax(-1), num_classes=n_bins)
        f0 = (idx * bins).sum(-1, keepdims=True)
    else:
        raise NotImplementedError()
    return f0[..., 0]


class F0Processor():
    def __init__(
        self,
        hparams,
        ds,
    ):
        self.sr = hparams["sample_rate"]
        self.extractor = hparams["extractor"]
        self.f0_bins = hparams["f0_bins"]
        self.f0_smoothing = hparams["f0_smoothing"]
        self.f0_norm = hparams["f0_norm"]
        self.f0_log = hparams["f0_log"]
        self.f0_bin_type = hparams["f0_bin_type"]

        self.speaker_stats = self._compute_f0_stats(ds)
        self.f0_min, self.f0_max = self._compute_f0_minmax(ds)

        if self.f0_bin_type == "adaptive":
            self.f0_bins = quantize_f0(
                self.speaker_stats, self.f0_bins, self.f0_norm, self.f0_log
            )
        elif self.f0_bin_type == "uniform":
            self.f0_bins = torch.linspace(self.f0_min, self.f0_max, self.f0_bins + 1)[
                1:-1
            ]
        else:
            raise NotImplementedError
        print(f"f0 min: {self.f0_min}, f0 max: {self.f0_max}")
        print(f"bins: {self.f0_bins} (shape: {self.f0_bins.shape})")

    def _compute_f0_stats(self, ds):
        speaker_stats = defaultdict(partial(F0Stat, True))
        for row in tqdm(ds):
            spk = row['spk']
            f0 = self._load_f0(row['wav'])
            mask = f0 != 0
            f0 = f0[mask]  # compute stats only on voiced parts
            speaker_stats[spk].update(f0)
        return speaker_stats

    def _load_f0(self, filename):
        audio = sb.dataio.dataio.read_audio(filename)
        f0 = extract_f0(audio.squeeze(0).numpy(), sr=self.sr, extractor=self.extractor).astype(np.float32)
        return torch.from_numpy(f0)

    def _preprocess_f0(self, f0, spk):
        mask = f0 != -999999  # process all frames
        # mask = (f0 != 0)  # only process voiced frames
        mean = (
            self.speaker_stats[spk].mean_log
            if self.f0_log
            else self.speaker_stats[spk].mean
        )
        std = (
            self.speaker_stats[spk].std_log
            if self.f0_log
            else self.speaker_stats[spk].std
        )
        if self.f0_log:
            f0[f0 == 0] = 1e-5
            f0[mask] = f0[mask].log()
        if self.f0_norm == "mean":
            f0[mask] = f0[mask] - mean
        if self.f0_norm == "meanstd":
            f0[mask] = (f0[mask] - mean) / std
        return f0

    def _compute_f0_minmax(self, ds):
        f0_min, f0_max = float("inf"), -float("inf")
        for row in tqdm(ds):
            spk = row['spk']
            f0 = self._load_f0(row['wav'])
            f0 = self._preprocess_f0(f0, spk)
            f0_min = min(f0_min, f0.min().item())
            f0_max = max(f0_max, f0.max().item())
        return f0_min, f0_max

    def __call__(self, wav, uttid, spk, unit):
        f0_raw = self._load_f0(wav)
        f0 = self._preprocess_f0(f0_raw.clone(), spk)
        
        f0 = F.interpolate(f0.unsqueeze(0).unsqueeze(0), unit.shape[0])[0, 0]
        f0_raw = F.interpolate(f0_raw.unsqueeze(0).unsqueeze(0), unit.shape[0])[0, 0]
        
        f0 = freq2bin(f0, f0_min=self.f0_min, f0_max=self.f0_max, bins=self.f0_bins)
        f0 = F.one_hot(f0.long(), num_classes=len(self.f0_bins) + 1).float()
        
        if self.f0_smoothing > 0:
            f0 = torch.tensor(
                gaussian_filter1d(f0.float().numpy(), sigma=self.f0_smoothing)
            )
        return f0, f0_raw

class Stat:
    def __init__(self, keep_raw=False):
        self.x = 0.0
        self.x2 = 0.0
        self.z = 0.0  # z = logx
        self.z2 = 0.0
        self.n = 0.0
        self.u = 0.0
        self.keep_raw = keep_raw
        self.raw = []

    def update(self, new_x):
        new_z = new_x.log()

        self.x += new_x.sum()
        self.x2 += (new_x**2).sum()
        self.z += new_z.sum()
        self.z2 += (new_z**2).sum()
        self.n += len(new_x)
        self.u += 1

        if self.keep_raw:
            self.raw.append(new_x)

    @property
    def mean(self):
        return self.x / self.n

    @property
    def std(self):
        return (self.x2 / self.n - self.mean**2) ** 0.5

    @property
    def mean_log(self):
        return self.z / self.n

    @property
    def std_log(self):
        return (self.z2 / self.n - self.mean_log**2) ** 0.5

    @property
    def n_frms(self):
        return self.n

    @property
    def n_utts(self):
        return self.u

    @property
    def raw_data(self):
        assert self.keep_raw, "does not support storing raw data!"
        return torch.cat(self.raw)


class F0Stat(Stat):
    def update(self, new_x):
        # assume unvoiced frames are 0 and consider only voiced frames
        if new_x is not None:
            super().update(new_x[new_x != 0])
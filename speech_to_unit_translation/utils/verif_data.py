import logging
import kaldiio
import joblib
import torch
import tqdm
import pathlib as pl
import pandas as pd
import numpy as np
import speechbrain as sb

from audio import load_waveform_from_stored_zip

audio_folder = pl.Path("/local_disk/calypso/jduret/corpus/SpeechMatrix/data/audios")
json_path = pl.Path("/local_disk/calypso/jduret/git/Chaanks/ES2UT/speech_matrix/speech_to_unit_translation/data/train.json")

@sb.utils.data_pipeline.takes("src_audio")
@sb.utils.data_pipeline.provides("sig")
def audio_pipeline(path):
    """Load the audio signal. This is done on the CPU in the `collate_fn`."""
    print(path)
    filename, byte_offset, byte_size = path.split(':')
    full_path = audio_folder / filename
    sig = load_waveform_from_stored_zip(full_path.as_posix(), int(byte_offset), int(byte_size))
    return sig
    
ds = sb.dataio.dataset.DynamicItemDataset.from_json(
    json_path=json_path,
    dynamic_items=[audio_pipeline],
    output_keys=[
        "sig",
    ]
)

for i in tqdm.tqdm(range(367390, len(ds))):
    sig = ds[i]['sig']
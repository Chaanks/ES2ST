from speechbrain.lobes.models.huggingface_wav2vec import HuggingFaceWav2Vec2

model_w2v2 = HuggingFaceWav2Vec2("LeBenchmark/wav2vec2-FR-7K-base", freeze_feature_extractor=True, freeze=True, save_path='../tmpdir/pretrained/')
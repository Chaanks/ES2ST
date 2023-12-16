import sys
import json
import pathlib as pl

import torch
import torch.nn.functional as F
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
import pickle


class PitchPredictorBrain(sb.Brain):
    def compute_forward(self, batch, stage):        
        (
            units,
            f0_bin,
            f0_raw,
            spk,
            spk_emb,
            emo_emb,
            mask,
            lengths
        ) = self.batch_to_device(batch)
        

        
        yhat = self.modules.model(units, spk=None, emo=emo_emb)
        yhat_raw = self.modules.model.inference(units, spk_id=spk, spk=None, emo=emo_emb)
        
        return (yhat, yhat_raw)
    
    def compute_objectives(self, predictions, batch, stage):
        (
            units,
            f0_bin,
            f0_raw,
            spk,
            spk_emb,
            emo_emb,
            mask,
            lengths
        ) = self.batch_to_device(batch)
        
        (yhat, yhat_raw) = predictions
        
        b, t, n_bins = f0_bin.shape
        nonzero_mask = (f0_raw != 0).logical_and(mask)
        expanded_mask = mask.unsqueeze(-1).expand(-1, -1, n_bins)
        
        if self.hparams.f0_pred == "mean":
            loss = F.binary_cross_entropy(
                yhat[expanded_mask], f0_bin[expanded_mask]
            )

        l1 = F.l1_loss(yhat_raw[mask], f0_raw[mask])
        l1_voiced = F.l1_loss(yhat_raw[nonzero_mask], f0_raw[nonzero_mask])

        self.last_batch = (
            f0_bin.cpu().numpy(),
            yhat.cpu().detach().numpy(),
            f0_raw.cpu().numpy(),
            yhat_raw.cpu().detach().numpy()
        )
        self.last_loss_stats[stage] = {"loss": loss.item(), "l1": l1.item(), "l1_voiced": l1_voiced.item()}
        return loss
    
    def on_fit_start(self):
        """Gets called at the beginning of ``fit()``, on multiple processes
        if ``distributed_count > 0`` and backend is ddp and initializes statistics
        """    
        self.f0_min = self.f0_processors["train"].f0_min
        self.f0_max = self.f0_processors["train"].f0_max
        self.f0_bins = self.f0_processors["train"].f0_bins
        self.speaker_stats = self.f0_processors["train"].speaker_stats
        self.modules.model.setup_f0_stats(self.f0_min, self.f0_max, self.f0_bins, self.speaker_stats)
        
        self.last_epoch = 0
        self.last_batch = None
        self.last_loss_stats = {}
        return super().on_fit_start()
    
    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of an epoch.
        Arguments
        ---------
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, sb.Stage.TEST
        stage_loss : float
            The average loss for all of the data processed in this stage.
        epoch : int
            The currently-starting epoch. This is passed
            `None` during the test stage.
        """

        # At the end of validation...
        if stage == sb.Stage.VALID:
            (
                f0_bin,
                yhat,
                f0_raw,
                yhat_raw,
            ) = self.last_batch
            
            print(f"example    y: {f0_bin.argmax(-1)[0, 0:20].tolist()}")
            print(f"example yhat: {yhat.argmax(-1)[0, 0:20].tolist()}")
            print(f"example    y: {f0_raw[0, 0:20].round().tolist()}")
            print(f"example yhat: {yhat_raw[0, 0:20].round().tolist()}")
            
            lr = self.optimizer.param_groups[-1]["lr"]
            # The train_logger writes a summary to stdout and to the logfile.
            self.hparams.train_logger.log_stats(
                {"Epoch": epoch, "lr": lr, },
                train_stats=self.last_loss_stats[sb.Stage.TRAIN],
                valid_stats=self.last_loss_stats[sb.Stage.VALID],
            )
            
            # The tensorboard_logger writes a summary to stdout and to the logfile.
            if self.hparams.use_wandb:
                self.wandb_logger.log_stats(
                    stats_meta={"Epoch": epoch, "lr_g": lr},
                    train_stats=self.last_loss_stats[sb.Stage.TRAIN],
                    valid_stats=self.last_loss_stats[sb.Stage.VALID],
                )
                
            # Save the current checkpoint and delete previous checkpoints.
            epoch_metadata = {
                **{"epoch": epoch},
                **self.last_loss_stats[sb.Stage.VALID],
            }
            if self.checkpointer is not None:
                self.checkpointer.save_and_keep_only(
                    meta=epoch_metadata,
                    end_of_epoch=True,
                    min_keys=["l1_voiced"],
                    ckpt_predicate=(
                        lambda ckpt: (
                            ckpt.meta["epoch"]
                            % self.hparams.keep_checkpoint_interval
                            != 0
                        )
                    )
                    if self.hparams.keep_checkpoint_interval is not None
                    else None,
                )

        # We also write statistics about test data to stdout and to logfile.
        if stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                {"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=self.last_loss_stats[sb.Stage.TEST],
            )
        

    def batch_to_device(self, batch):
        """Transfers the batch to the target device

        Arguments
        ---------
        batch: tuple
            the batch to use

        Returns
        -------
        batch: tuple
            the batch on the correct device
        """
        (
            units,
            f0,
            f0_raw,
            spk,
            spk_emb,
            emo_emb,
            mask,
            lengths
        ) = batch

        units = units.to(self.device, non_blocking=True)
        f0 = f0.to(self.device, non_blocking=True)
        f0_raw = f0_raw.to(self.device, non_blocking=True)
        spk_emb = spk_emb.to(self.device, non_blocking=True)
        emo_emb = emo_emb.to(self.device, non_blocking=True)
        mask = mask.to(self.device, non_blocking=True)

        return (units, f0, f0_raw, spk, spk_emb, emo_emb, mask, lengths)


def dataio_prepare(hparams):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions.
    """

    from utils.audio import F0Processor, extract_f0, freq2bin
    f0_processors = {}
    
    from utils.embedding import EmbeddingManager
    units_loader = EmbeddingManager(hparams["units_folder"])

    if hparams["multi_speaker"]:
        speakers_loader = EmbeddingManager(hparams["speakers_folder"])

    if hparams["multi_emotion"]:
        emotions_loader = EmbeddingManager(hparams["emotions_folder"])

    @sb.utils.data_pipeline.takes("id")
    @sb.utils.data_pipeline.provides("unit")
    def unit_pipeline(clip_id):
        unit = units_loader.get_embedding_by_clip(clip_id)
        unit = torch.IntTensor(unit)
        return unit

    @sb.utils.data_pipeline.takes("id")
    @sb.utils.data_pipeline.provides("spk_emb")
    def spk_pipeline(utt_id):
        spk_emb = speakers_loader.get_embedding_by_clip(utt_id)
        yield torch.FloatTensor(spk_emb)
    
    @sb.utils.data_pipeline.takes("id")
    @sb.utils.data_pipeline.provides("emo_emb")
    def emo_pipeline(utt_id):
        emo_emb = emotions_loader.get_embedding_by_clip(utt_id)
        yield torch.FloatTensor(emo_emb)
        

    pipelines = [unit_pipeline]
    keys = ["id", "wav", "spk", "unit"]
    if hparams["multi_speaker"]:
        pipelines.append(spk_pipeline)
        keys.append("spk_emb")
    if hparams["multi_emotion"]:
        pipelines.append(emo_pipeline)
        keys.append("emo_emb")

    datasets = {}
    for split in hparams["splits"]:
        ds_dict = {}
        for ds_path in hparams[f"{split}_json"]:
            data = json.load(open(ds_path))
            for key in data: data[key]['split'] = split
            ds_dict.update(data)
        datasets[split] = sb.dataio.dataset.DynamicItemDataset(
            ds_dict,
            dynamic_items=pipelines,
            output_keys=keys,
        )
        
        save_folder = pl.Path(hparams["save_folder"])
        f0_processor_ckpt = save_folder / f"{split}.f0p"
        if f0_processor_ckpt.exists():
            print(f"Load {split} f0 processor")
            with open(f0_processor_ckpt, 'rb') as f:
                f0_processors[split] = pickle.load(f)
        else:
            f0_processors[split] = F0Processor(hparams, datasets[split])
            print(f"Save {split} f0 processor")
            with open(f0_processor_ckpt, 'wb') as f:
                pickle.dump(f0_processors[split], f)

    # Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav", "id", "spk", "split")
    @sb.utils.data_pipeline.provides("f0")
    def f0_pipeline(wav, clip_id, spk, split):
        unit = units_loader.get_embedding_by_clip(clip_id)
        unit = torch.LongTensor(unit)
        f0_processor = f0_processors[split]
        #print(f"f0 min: {f0_processor.f0_min}, f0 max: {f0_processor.f0_max}")
        f0, f0_raw  = f0_processor(wav, clip_id, spk, unit)
        return f0, f0_raw
    keys.append("f0")
    
    for split in hparams["splits"]:    
        datasets[split].add_dynamic_item(f0_pipeline)
        datasets[split].set_output_keys(keys)

    return datasets, f0_processors


if __name__ == "__main__":

    # Load hyperparameters file with command-line overrides
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    datasets, f0_processors  = dataio_prepare(hparams)

    # Initialize the Brain object to prepare for mask training.
    pitch_predictor_brain = PitchPredictorBrain(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )
    pitch_predictor_brain.f0_processors = f0_processors
    
    
    if hparams["use_wandb"]:
        from utils.logger import WandBLogger
        pitch_predictor_brain.wandb_logger = WandBLogger(**hparams["logger_opts"])

    pitch_predictor_brain.fit(
        epoch_counter=pitch_predictor_brain.hparams.epoch_counter,
        train_set=datasets["train"],
        valid_set=datasets["valid"],
        train_loader_kwargs=hparams["train_dataloader_opts"],
        valid_loader_kwargs=hparams["valid_dataloader_opts"],
    )

    # Load the best checkpoint for evaluation
    # test_stats = emo_id_brain.evaluate(
    #     test_set=datasets["test"],
    #     min_key="error_rate",
    #     test_loader_kwargs=hparams["dataloader_options"],
    # )
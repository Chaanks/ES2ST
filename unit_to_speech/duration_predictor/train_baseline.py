import sys
import json
import pathlib as pl

import torch
import torch.nn.functional as F
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from utils.metric import Accuracy


class DurationPredictorBrain(sb.Brain):
    def compute_forward(self, batch, stage):        
        (
            feats,
            targets,
            emo_emb,
            mask,
            lengths,
        ) = self.batch_to_device(batch)
        
        pred = self.modules.model(feats, emo=None)
        
        return pred
    
    def compute_objectives(self, predictions, batch, stage):
        (
            feats,
            targets,
            emo_emb,
            mask,
            lengths,
        ) = self.batch_to_device(batch)
        
        loss = F.mse_loss(
            input=predictions.float(),
            target=torch.log(targets.float() + 1),
            reduction='none'
        ) * mask

        loss = torch.mean(loss)

        self.last_batch = (
            feats.cpu().numpy(),
            targets.cpu().numpy(),
            predictions.cpu().detach().numpy(),
        )

        if stage != sb.Stage.TRAIN:
            # get normal scale loss
            predictions_scaled = torch.exp(predictions) - 1
            predictions_scaled = torch.round(predictions_scaled)
            self.acc.update(predictions_scaled[mask].view(-1).float(), targets[mask].view(-1).float())
            
            scaled_loss = torch.sum(torch.abs(predictions_scaled - targets) * mask) / mask.sum()
            self.last_loss_stats[stage] = {"loss": loss.item(), "scaled_loss": scaled_loss.item()}

        else:
            self.last_loss_stats[stage] = {"loss": loss.item()}

        return loss
    
    def on_fit_start(self):
        """Gets called at the beginning of ``fit()``, on multiple processes
        if ``distributed_count > 0`` and backend is ddp and initializes statistics
        """    
        
        self.last_epoch = 0
        self.last_batch = None
        self.last_loss_stats = {}
        return super().on_fit_start()

    def on_stage_start(self, stage, epoch=None):
        self.acc = Accuracy()
    
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
                feats,
                targets,
                predictions,
            ) = self.last_batch

            acc0 = self.acc.acc(tol=0)
            acc1 = self.acc.acc(tol=1)
            acc2 = self.acc.acc(tol=2)
            acc3 = self.acc.acc(tol=3)
            print(f"accs: {acc0,acc1,acc2,acc3}")

            
            lr = self.optimizer.param_groups[-1]["lr"]
            # The train_logger writes a summary to stdout and to the logfile.
            self.hparams.train_logger.log_stats(
                {"Epoch": epoch, "lr": lr, "acc_t0": acc0, "acc_t1": acc1},
                train_stats=self.last_loss_stats[sb.Stage.TRAIN],
                valid_stats=self.last_loss_stats[sb.Stage.VALID],
            )
            
            # The tensorboard_logger writes a summary to stdout and to the logfile.
            if self.hparams.use_wandb:
                self.wandb_logger.log_stats(
                    stats_meta={"Epoch": epoch, "lr_g": lr, "acc_t0": acc0, "acc_t1": acc1},
                    train_stats=self.last_loss_stats[sb.Stage.TRAIN],
                    valid_stats=self.last_loss_stats[sb.Stage.VALID],
                )
                
            # Save the current checkpoint and delete previous checkpoints.
            epoch_metadata = {
                **{"epoch": epoch, "acc_t0": acc0, "acc_t1": acc1},
                **self.last_loss_stats[sb.Stage.VALID],
            }
            if self.checkpointer is not None:
                self.checkpointer.save_and_keep_only(
                    meta=epoch_metadata,
                    end_of_epoch=True,
                    max_keys=["acc_t0"],
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
            feats,
            targets,
            emo_emb,
            mask,
            lengths
        ) = batch

        feats = feats.to(self.device, non_blocking=True)
        targets = targets.to(self.device, non_blocking=True)
        emo_emb = emo_emb.to(self.device, non_blocking=True)
        mask = mask.to(self.device, non_blocking=True)

        return (feats, targets, emo_emb, mask, lengths)


def dataio_prepare(hparams):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions.
    """

    from utils.embedding import EmbeddingManager
    units_loader = EmbeddingManager(hparams["units_folder"])
    emotions_loader = EmbeddingManager(hparams["emotions_folder"])

    # Define audio pipeline:
    @sb.utils.data_pipeline.takes("id")
    @sb.utils.data_pipeline.provides("feats")
    def feats_pipeline(clip_id):
        features = units_loader.get_embedding_by_clip(clip_id)
        features = torch.IntTensor(features)
        unique, count = torch.unique_consecutive(features, return_counts=True)

        return (unique, count)

    @sb.utils.data_pipeline.takes("id")
    @sb.utils.data_pipeline.provides("emo_emb")
    def emo_pipeline(utt_id):
        emo_emb = emotions_loader.get_embedding_by_clip(utt_id)
        yield torch.FloatTensor(emo_emb)
    
    datasets = {}
    for split in hparams["splits"]:
        ds_dict = {}
        for ds_path in hparams[f"{split}_json"]:
            data = json.load(open(ds_path))
            ds_dict.update(data)
        datasets[split] = sb.dataio.dataset.DynamicItemDataset(
            ds_dict,
            dynamic_items=[feats_pipeline, emo_pipeline],
            output_keys=["id", "feats", "emo_emb"],
        )

    return datasets

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

    datasets = dataio_prepare(hparams)

    # Initialize the Brain object to prepare for mask training.
    duration_predictor_brain = DurationPredictorBrain(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )


    if hparams["use_wandb"]:
        from utils.logger import WandBLogger
        duration_predictor_brain.wandb_logger = WandBLogger(**hparams["logger_opts"])

    duration_predictor_brain.fit(
        epoch_counter=duration_predictor_brain.hparams.epoch_counter,
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
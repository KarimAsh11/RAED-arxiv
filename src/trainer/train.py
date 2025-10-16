import omegaconf
import hydra
import torch
import wandb
import pytorch_lightning as pl

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor

from src.data.pl_data_modules import EmergePLDataModule
from src.models.pl_modules import EmergePLModule
from pathlib import Path
from pytorch_lightning.loggers import WandbLogger
from src.callbacks.eval_callbacks import EvalCallback
from src.models.utils import prep_raed_model_tokenizer


def init_logger(config, model):
    wandb_logger = None
    if config.logging.wandb_arg:
        wandb_config = config.logging.wandb_arg
        hydra.utils.log.info(f"Instantiating <WandbLogger>")
        wandb_logger = WandbLogger(
            name=config.logging.wandb_arg.name,
            project=config.logging.wandb_arg.project,
            entity=config.logging.wandb_arg.get("entity", None),
            log_model=True,
        )
        wandb_logger.watch(model)
    return wandb_logger

def train(conf: omegaconf.DictConfig) -> None:

    # reproducibility
    pl.seed_everything(conf.train.seed)
    torch.set_float32_matmul_precision(conf.train.float32_matmul_precision)

    if conf.train.lr_scheduler:
        if not conf.train.lr_scheduler.num_warmup_steps:
            if conf.train.lr_scheduler.warmup_steps_ratio is not None:
                conf.train.lr_scheduler.num_warmup_steps = int(
                    conf.train.lr_scheduler.num_training_steps * conf.train.lr_scheduler.warmup_steps_ratio
                )
    
    pl_tokenizer, pl_model = prep_raed_model_tokenizer(conf.model)

    # data module declaration
    pl_data_module = EmergePLDataModule(conf, tokenizer=pl_tokenizer)

    # main module declaration
    pl_module = EmergePLModule(conf, model=pl_model, tokenizer=pl_tokenizer)

    # callbacks declaration
    callbacks_store = []

    if conf.train.early_stopping_callback is not None:
        early_stopping_callback: EarlyStopping = hydra.utils.instantiate(conf.train.early_stopping_callback)
        callbacks_store.append(early_stopping_callback)

    if conf.train.model_checkpoint_callback is not None:
        model_checkpoint_callback: ModelCheckpoint = hydra.utils.instantiate(conf.train.model_checkpoint_callback)
        callbacks_store.append(model_checkpoint_callback)

    if conf.train.lr_scheduler:
        if conf.train.lr_scheduler.lr_monitoring: 
            lr_monitor = LearningRateMonitor(logging_interval='step')
            callbacks_store.append(lr_monitor)
    
    if conf.train.evaluation_print:
        callbacks_store.append(EvalCallback(output_path=conf.train.output_path))


    experiment_logger=None
    if conf.logging.log:
        experiment_logger = hydra.utils.instantiate(conf.logging.wandb_arg)
        if pl_module is not None:
            experiment_logger.watch(pl_module, **conf.logging.watch)
            # experiment_logger.watch(pl_module.model)

    # trainer
    trainer: Trainer = hydra.utils.instantiate(conf.train.pl_trainer, callbacks=callbacks_store, logger=experiment_logger)

    # module val
    # trainer.validate(pl_module, datamodule=pl_data_module, ckpt_path=None)

    # module fit
    trainer.fit(pl_module, datamodule=pl_data_module, ckpt_path=None)

    # module test
    trainer.test(pl_module, datamodule=pl_data_module)


@hydra.main(config_path="../../conf", config_name="raed")
def main(conf: omegaconf.DictConfig):
    train(conf)


if __name__ == "__main__":
    main()

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
from src.callbacks.perplex_fid_callbacks import FidPerplexCallback
from src.callbacks.perplex_callbacks import PerplexCallback
# from src.callbacks.perplex_smol_callbacks import PerplexDecoderCallback
from src.callbacks.constrained_callbacks import ConstrainedPerplexCallback
from src.callbacks.perplex_decoder_callbacks import PerplexDecoderCallback

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

def test(conf: omegaconf.DictConfig) -> None:

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

    if "best_rag_ckpt_path" in conf.train and conf.train.best_rag_ckpt_path is not None:
        pl_module.load_state_dict(torch.load("../"+conf.train_ckpt)["state_dict"])
        print("TRAINING CKPT LOADED ....")
        print("CHECKPOINT: ", conf.train.best_rag_ckpt_path)

    # callbacks declaration
    callbacks_store = []

    if conf.train.evaluation_print:
        if "smol" in conf.model.model_name.lower():
            callbacks_store.append(PerplexDecoderCallback(output_val_path=conf.train.output_val_path, output_test_path=conf.train.output_test_path, tokenizer=pl_tokenizer))
        elif conf.model.fid:
            callbacks_store.append(FidPerplexCallback(output_path=conf.train.output_path))
        else:
            callbacks_store.append(PerplexCallback(output_path=conf.train.output_path))

    # trainer
    trainer: Trainer = hydra.utils.instantiate(conf.train.pl_trainer, callbacks=callbacks_store)

    # module val
    trainer.validate(pl_module, datamodule=pl_data_module)
    
    # module test
    trainer.test(pl_module, datamodule=pl_data_module)


@hydra.main(config_path="../../conf", config_name="raed")
def main(conf: omegaconf.DictConfig):
    test(conf)


if __name__ == "__main__":
    main()

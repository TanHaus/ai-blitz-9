import nemo
import nemo.collections.asr as nemo_asr
from ruamel.yaml import YAML
from omegaconf import DictConfig
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

def main():
    model_name = "QuartzNet15x5Base-En"
    # model_name = "stt_en_jasper10x5dr"
    model = nemo_asr.models.EncDecCTCModel.from_pretrained(model_name=model_name)

    config_path = "config.yaml"
    yaml = YAML(typ="safe")

    with open(config_path) as f:
        params = yaml.load(f)

    params["model"]["train_ds"]["manifest_filepath"] = "train_manifest.json"
    params["model"]["train_ds"]["batch_size"] = 128
    params["model"]["train_ds"]["max_duration"] = 5
    params["model"]["train_ds"]["num_workers"] = 4
    params["model"]["train_ds"]["pin_memory"] = True

    params["model"]["validation_ds"]["manifest_filepath"] = "val_manifest.json"
    params["model"]["validation_ds"]["batch_size"] = 128
    params["model"]["validation_ds"]["num_workers"] = 4
    params["model"]["validation_ds"]["pin_memory"] = True

    params["model"]["optim"]["name"] = "novograd"
    params["model"]["optim"]["lr"] = 1e-3
    params["model"]["optim"]["sched"]["warmup_ratio"] = 0.1

    model.setup_optimization(optim_config=DictConfig(params["model"]["optim"]))
    model.setup_training_data(train_data_config=DictConfig(params["model"]["train_ds"]))
    model.setup_validation_data(val_data_config=DictConfig(params["model"]["validation_ds"]))

    logger = WandbLogger(project="ai-blitz-9", name="adamw-lr1e-3", log_model=True)
    logger.watch(model)
    
    trainer = pl.Trainer(
        gpus=[0], 
        max_epochs=100,
        precision=16,
        callbacks=[
            EarlyStopping(monitor="val_loss"),
            ModelCheckpoint(monitor="val_loss")
        ],
        logger=logger
    )

    trainer.fit(model)

if __name__ == "__main__":
    main()
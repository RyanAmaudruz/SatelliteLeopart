import click
import os
import torch
import pandas as pd
import sacred

from datetime import datetime
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import NeptuneLogger

import sys
sys.path.append('./')

from data.s2c.s2c_data_module import S2cDataModule
from experiments.utils import get_backbone_weights
from src.leopart import Leopart
from src.leopart_transforms import LeopartTransforms

ex = sacred.experiment.Experiment()
api_key = "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5haSIsImFwaV91cmwiOiJodHRwczovL3VpLm5lcHR1bmUuYWkiLCJhcGlfa2V5IjoiMTI3NWYzYmEtYWE4NC00NzRhLWJlZGEtNTA5ZTE4NTgxMzg0In0="


@click.command()
@click.option("--config_path", type=str, default='/gpfs/home2/ramaudruz/SatelliteLeopart/experiments/configs/train_s2c_config.yml')
@click.option("--seed", type=int, default=400)
def entry_script(config_path, seed):
    if config_path is not None:
        ex.add_config(os.path.join(os.path.abspath(os.path.dirname(__file__)), config_path))
    else:
        ex.add_config(os.path.join(os.path.abspath(os.path.dirname(__file__)), "leopart_config_dev.yml"))
    time = datetime.now().strftime("%Y%m%d-%H%M%S")
    ex_name = f"SatelliteLeopart-{time}"
    checkpoint_dir = os.path.join(ex.configurations[0]._conf["train"]["checkpoint_dir"], ex_name)
    ex.observers.append(sacred.observers.FileStorageObserver(checkpoint_dir))
    ex.run(config_updates={'seed': seed}, options={'--name': ex_name})


@ex.main
@ex.capture
def finetune_with_spatial_loss(_config, _run):
    # Setup logging
    neptune_logger = NeptuneLogger(
        api_key=api_key,
        project_name="RyanAmaudruz/SatelliteLeopart",
        experiment_name=_run.experiment_info["name"],
        params=pd.json_normalize(_config).to_dict(orient='records')[0],
        tags=_config["tags"].split(','),
    )

    # Process config
    print("Config:")
    print(_config)
    data_config = _config["data"]
    train_config = _config["train"]
    seed_everything(_config["seed"])

    # Init data modules and tranforms
    data_dir = data_config["data_dir"]
    dataset_name = data_config["dataset_name"]
    train_transforms = LeopartTransforms(size_crops=data_config["size_crops"],
                                         nmb_crops=data_config["nmb_samples"],
                                         min_intersection=data_config["min_intersection_crops"],
                                         min_scale_crops=data_config["min_scale_crops"],
                                         max_scale_crops=data_config["max_scale_crops"],
                                         jitter_strength=data_config["jitter_strength"],
                                         blur_strength=data_config["blur_strength"])

    if dataset_name == 's2c_un':
        meta_df = pd.read_csv(os.path.join(data_dir, "ssl4eo_s2_l1c_full_extract_metadata.csv"))
        temp_var = meta_df['patch_id'].astype(str)
        meta_df['patch_id'] = temp_var.map(lambda x: (7 - len(x)) * '0' + x)
        meta_df['file_name'] = meta_df['patch_id'] + '/' + meta_df['timestamp']
        num_images = meta_df.shape[0]

        train_data_module = S2cDataModule(
            train_transforms=train_transforms,
            batch_size=train_config["batch_size"],
            meta_df=meta_df,
            num_workers=_config["num_workers"],
            num_images=num_images
        )
    else:
        raise ValueError(f"Data set {dataset_name} not supported")

    # Use data module wrapper to have train_data_module provide train loader and voc data module the val loader
    data_module = train_data_module
    # data_module = TrainXVOCValDataModule(train_data_module, voc_data_module)

    # Init method
    model = Leopart(
        use_teacher=train_config["use_teacher"],
        loss_mask=train_config["loss_mask"],
        queue_length=train_config["queue_length"],
        momentum_teacher=train_config["momentum_teacher"],
        momentum_teacher_end=train_config["momentum_teacher_end"],
        num_clusters_kmeans=train_config["num_clusters_kmeans_miou"],
        weight_decay_end=train_config["weight_decay_end"],
        roi_align_kernel_size=train_config["roi_align_kernel_size"],
        val_downsample_masks=train_config["val_downsample_masks"],
        arch=train_config["arch"],
        patch_size=train_config["patch_size"],
        lr_heads=train_config["lr_heads"],
        gpus=_config["gpus"],
        num_classes=data_config["num_classes_val"],
        batch_size=train_config["batch_size"],
        num_samples=len(data_module),
        projection_feat_dim=train_config["projection_feat_dim"],
        projection_hidden_dim=train_config["projection_hidden_dim"],
        n_layers_projection_head=train_config["n_layers_projection_head"],
        max_epochs=train_config["max_epochs"],
        val_iters=train_config["val_iters"],
        nmb_prototypes=train_config["nmb_prototypes"],
        temperature=train_config["temperature"],
        sinkhorn_iterations=train_config["sinkhorn_iterations"],
        crops_for_assign=train_config["crops_for_assign"],
        nmb_crops=data_config["nmb_samples"],
        optimizer=train_config["optimizer"],
        exclude_norm_bias=train_config["exclude_norm_bias"],
        lr_backbone=train_config["lr_backbone"],
        final_lr=train_config["final_lr"],
        weight_decay=train_config["weight_decay"],
        epsilon=train_config["epsilon"],
    )

    # Optionally load weights
    if train_config["checkpoint"] is not None and train_config["only_load_weights"]:
        state_dict = torch.load(train_config["checkpoint"])["state_dict"]
        msg = model.load_state_dict(state_dict, strict=True)
        print(msg)
    elif train_config["checkpoint"] is None:
        if train_config["pretrained_weights"] is not None:
            w_student = get_backbone_weights(train_config["arch"],
                                             train_config["pretrained_weights"],
                                             patch_size=train_config.get("patch_size"),
                                             weight_prefix="model")
            w_teacher = get_backbone_weights(train_config["arch"],
                                             train_config["pretrained_weights"],
                                             patch_size=train_config.get("patch_size"),
                                             weight_prefix="teacher")

            student_sat_w = torch.load('/gpfs/work5/0/prjs0790/data/modified_checkpoints/ssl_dino_new_trans_e95_student_MODIFIED.pth', map_location="cpu")
            teacher_sat_w = torch.load('/gpfs/work5/0/prjs0790/data/modified_checkpoints/ssl_dino_new_trans_e95_teacher_MODIFIED.pth', map_location="cpu")
            for k, v in student_sat_w['state_dict'].items():
                w_student[k] = v
            for k, v in teacher_sat_w['state_dict'].items():
                w_teacher[k] = v

            w_student = {f'model.{k}' if k.startswith('head.') else k: v for k, v in w_student.items()}
            w_teacher = {f'teacher.{k}' if k.startswith('head.') else k: v for k, v in w_teacher.items()}

            msg = model.load_state_dict({**w_student, **w_teacher}, strict=False)
            print(msg)

    # Setup attention map evaluation callback
    checkpoint_dir = os.path.join(train_config["checkpoint_dir"], _run.experiment_info["name"])
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename='ckp-{epoch:02d}',
        save_top_k=-1,
        verbose=True,
        period=train_config["save_checkpoint_every_n_epochs"]
    )
    callbacks = [checkpoint_callback]

    # Used if train data is small as for pvoc
    val_every_n_epochs = train_config.get("val_every_n_epochs")
    if val_every_n_epochs is None:
        val_every_n_epochs = 1

    from pytorch_lightning.loggers import WandbLogger
    # from pytorch_lightning import Trainer
    wandb_logger = WandbLogger(log_model="all")
    # Setup trainer and start training

    trainer = Trainer(
        check_val_every_n_epoch=val_every_n_epochs,
        logger=wandb_logger,
        max_epochs=train_config["max_epochs"],
        gpus=_config["gpus"],
        accelerator='ddp' if _config["gpus"] > 1 else None,
        fast_dev_run=train_config["fast_dev_run"],
        log_every_n_steps=50,
        benchmark=True,
        deterministic=False,
        amp_backend='native',
        num_sanity_val_steps=train_config['val_iters'],
        resume_from_checkpoint=train_config['checkpoint'] if not train_config["only_load_weights"] else None,
        terminate_on_nan=True,
        callbacks=callbacks,
    )
    trainer.fit(model, datamodule=data_module)


if __name__ == '__main__':
    entry_script()

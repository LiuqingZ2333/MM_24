from share import *
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from tutorial_dataset import MyDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
from pytorch_lightning.strategies import DeepSpeedStrategy
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

SAVE_PATH = "./output/control_MG"
log0 = CSVLogger(save_dir=SAVE_PATH, name="control_MG")

# Configs
batch_size = 1
logger_freq = 600
learning_rate = 1e-5
sd_locked = True
only_mid_control = False

model_name = "control_MG"
# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model(
    f"./models/{model_name}.yaml"
).cpu()
model.load_state_dict(
    load_state_dict(
        './v2-1_512-ema-pruned.ckpt', location="cpu"
    ),
    strict=False,
)
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control

# Misc

strategy = DeepSpeedStrategy()
dataset = MyDataset("./train_control_MG/")
dataloader = DataLoader(dataset, num_workers=0, batch_size=batch_size, shuffle=True)
logger = ImageLogger(batch_frequency=logger_freq)
trainer = pl.Trainer(
    strategy=strategy,
    gpus = 1,
    precision=32,
    logger=log0,
    callbacks=[logger, ModelCheckpoint(save_top_k=-1)],
    default_root_dir=SAVE_PATH,
)

# Train!
trainer.fit(model, dataloader)

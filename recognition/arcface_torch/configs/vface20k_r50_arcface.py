import torch
from easydict import EasyDict as edict
from torchvision.transforms import v2 as transforms_v2


# make training faster
# our RAM is 256G
# mount -t tmpfs -o size=140G  tmpfs /train_tmp

config = edict()
config.margin_list = (1.0, 0.5, 0.0)
config.network = "r50"
config.resume = False
config.output = 'outputs/vf20k_r50_arcface'
config.embedding_size = 512
config.sample_rate = 1.0
config.fp16 = True
config.momentum = 0.9
config.weight_decay = 5e-4
config.batch_size = 128
config.lr = 0.1
config.dali = False

config.rec = "/path/to/vface20k"
config.test_root = "/path/to//test"
config.num_classes = 20000
config.num_image = 905972
config.num_epoch = 40
config.warmup_epoch = 0
config.num_workers = 2
config.lr_scheduler = "step"
config.lr_step_epochs = [18, 28, 35]
config.val_targets = ['lfw', 'calfw', 'cplfw', 'cfp_fp', "agedb_30", 'eclipse', 'hadrian']
config.machine_name = "stuxmet4"
config.transform = transforms_v2.Compose([
    transforms_v2.ToImage(),
    transforms_v2.Resize(size=(112, 112)),
    # transforms_v2.RandomResizedCrop(size=112, scale=(0.2, 1.0), ratio=(0.75, 1.3333333333333333)),
    transforms_v2.RandomHorizontalFlip(),
    transforms_v2.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0),
    transforms_v2.ToDtype(torch.float32, scale=True),
    transforms_v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    transforms_v2.RandomErasing(),
])

import torch
from easydict import EasyDict as edict
from torchvision.transforms import v2 as transforms_v2
from .vec2face_transform import AdaFaceAugmentTransformV2
# make training faster
# our RAM is 256G
# mount -t tmpfs -o size=140G  tmpfs /train_tmp

config = edict()
config.margin_list = (1.0, 0.5, 0.0)
config.network = "r50"
config.resume = False
config.output = 'outputs/vface10k_r50'
config.embedding_size = 512
config.sample_rate = 1.0
config.fp16 = True
config.momentum = 0.9
config.weight_decay = 5e-4
config.batch_size = 128
config.lr = 0.1
config.dali = False

config.rec = "/workspace/dataset/vface10k/train"
config.test_root = "/workspace/dataset/test"
config.num_classes = 10000
config.num_image = 452454
config.num_epoch = 40
config.warmup_epoch = 0
config.num_workers = 4
config.lr_scheduler = "step"
config.lr_step_epochs = [18, 28, 35]
config.val_targets = ['lfw',  'calfw', 'cplfw', 'cfp_fp', 'agedb_30']

config.transform = transforms_v2.Compose([
    transforms_v2.ToImage(),
    transforms_v2.RandomHorizontalFlip(),
    AdaFaceAugmentTransformV2(p=1.0, crop_augmentation_prob=0.3, photometric_augmentation_prob=0.3, low_res_augmentation_prob=0.3),
    transforms_v2.ToDtype(torch.float32, scale=True),
    transforms_v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    transforms_v2.RandomErasing(),
])

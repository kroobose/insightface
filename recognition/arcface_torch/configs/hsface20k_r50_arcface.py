from easydict import EasyDict as edict

# make training faster
# our RAM is 256G
# mount -t tmpfs -o size=140G  tmpfs /train_tmp

config = edict()
config.margin_list = (1.0, 0.5, 0.0)
config.network = "r50"
config.resume = False
config.output = 'outputs/hs20k_r50_arcface'
config.embedding_size = 512
config.sample_rate = 1.0
config.fp16 = True
config.momentum = 0.9
config.weight_decay = 5e-4
config.batch_size = 128
config.lr = 0.1
config.dali = False

config.rec = "/path/to/hsface20k/train"
config.test_root = "/path/to/hsface20k/test"
config.num_classes = 20000
config.num_image = 1000000
config.num_epoch = 26
config.warmup_epoch = 0
config.num_workers = 2
config.lr_scheduler = "step"
config.lr_step_epochs = [12, 20, 24]
config.val_targets = ['lfw', 'calfw', 'cplfw', 'cfp_fp', "agedb_30", 'eclipse', 'hadrian']
config.machine_name = "xx"

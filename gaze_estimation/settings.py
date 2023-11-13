class Settings(object):
    pass


settings = Settings()

settings.dataset = Settings()
settings.dataset.processed_dataset_path = 'datasets/MPIIGaze.h5'
settings.dataset.processed_dataset_output_dir = 'datasets/'
settings.dataset.dataset_dir = 'datasets/MPIIGaze'

settings.device = 'cpu'

settings.model = Settings()
settings.model.name = 'lenet'
settings.model.backbone = Settings()
settings.model.backbone.name = 'resnet_simple'
settings.model.backbone.pretrained = 'resnet18'
settings.model.backbone.resnet_block = 'basic'
settings.model.backbone.resnet_layers = [2, 2, 2]

settings.train = Settings()
settings.train.batch_size = 64
# optimizer (options: sgd, adam, amsgrad)
settings.train.optimizer = 'sgd'
settings.train.base_lr = 0.01
settings.train.momentum = 0.9
settings.train.nesterov = True
settings.train.weight_decay = 1e-4
settings.train.no_weight_decay_on_bn = False
# options: L1, L2, SmoothL1
settings.train.loss = 'L2'
settings.train.seed = 0
settings.train.val_first = True
settings.train.val_period = 1

settings.train.test_id = 0
settings.train.val_ratio = 0.1

settings.train.output_dir = 'experiments/mpiigaze/exp00'
settings.train.log_period = 100
settings.train.checkpoint_period = 10

# optimizer
settings.optim = Settings()

# scheduler
settings.scheduler = Settings()
settings.scheduler.epochs = 40
# scheduler (options: multistep, cosine)
settings.scheduler.type = 'multistep'
settings.scheduler.milestones = [20, 30]
settings.scheduler.lr_decay = 0.1
settings.scheduler.lr_min_factor = 0.001

# train data loader
settings.train.train_dataloader = Settings()
settings.train.train_dataloader.num_workers = 2
settings.train.train_dataloader.drop_last = True
settings.train.train_dataloader.pin_memory = False
settings.train.val_dataloader = Settings()
settings.train.val_dataloader.num_workers = 1
settings.train.val_dataloader.pin_memory = False

# test config
settings.test = Settings()
settings.test.test_id = 0
settings.test.checkpoint = 'experiments/mpiigaze/exp00/00/checkpoint_0040.pth'
settings.test.output_dir = 'out'
settings.test.batch_size = 256
# test data loader
settings.test.dataloader = Settings()
settings.test.dataloader.num_workers = 2
settings.test.dataloader.pin_memory = False

# cuDNN
settings.cudnn = Settings()
settings.cudnn.benchmark = True
settings.cudnn.deterministic = False


def get_settings():
    return settings

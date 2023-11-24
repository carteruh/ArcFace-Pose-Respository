from easydict import EasyDict as edict

def get_config():
    config = edict()

    # Training configurations
    config.lr = 1e-1
    config.weight_decay = 5e-4
    config.backbone = "ir50"
    config.embedding_size= 512
    config.weight_path = ''
    config.input_size = (112, 112)
    config.train_folder= ''
    config.test_folder= ''
    config.batch_size = 128
    config.milestone= [40, 80, 120, 200]
    config.momentum= 0.9
    config.scale= 64.0
    config.margin= 0.5
    config.pose_bin = 'all_pose_HELEN'
 
    return config

from lib.config import cfg, args
import numpy as np
import os
from training_loop import TrainLoop
from lib.networks.make_network import create_gaussian_diffusion
from lib.utils.parser_util import *
### SCRIPTS BEGINING ###
def run_dataset():
    from lib.datasets.make_dataset import make_data_loader
    from lib.utils.data_utils import to_cuda
    import tqdm

    cfg.train.num_workers = 0
    data_loader = make_data_loader(cfg)
    total_time = 0
    import time
    for batch in tqdm.tqdm(data_loader):
        start = time.time()
        #batch = to_cuda(batch)
        total_time += time.time() - start
    print(total_time / len(data_loader))

def run_network():
    from lib.networks import make_network
    from lib.datasets import make_data_loader
    from lib.utils.net_utils import load_network
    from lib.utils.data_utils import to_cuda
    from lib.utils.fixseed import fixseed
    from lib.train.train_platforms import ClearmlPlatform, TensorboardPlatform, NoPlatform  # required for the eval operation

    import tqdm
    import torch
    import time

    args = train_args()
    fixseed(args.seed)
    network = make_network(cfg)
    diffusion = create_gaussian_diffusion(args)
    
    #load_network(network, cfg.trained_model_dir, epoch=cfg.test.epoch)

    train_platform_type = eval(args.train_platform_type)
    train_platform = train_platform_type(args.save_dir)
    train_platform.report_args(args, name='Args')

    data_loader = make_data_loader(cfg)

    TrainLoop(args, train_platform, network, diffusion, data_loader).run_loop()

    network.eval()


    
    # total_time = 0
    # for batch in tqdm.tqdm(data_loader):
    #     batch = to_cuda(batch)
    #     with torch.no_grad():
    #         torch.cuda.synchronize()
    #         start = time.time()
    #         network(x=batch,timesteps=32)
    #         torch.cuda.synchronize()
    #         total_time += time.time() - start
    # print(total_time / len(data_loader))


def run_evaluate():
    from lib.datasets import make_data_loader
    from lib.evaluators import make_evaluator
    import tqdm
    import torch
    from lib.networks import make_network
    from lib.utils import net_utils
    import time

    network = make_network(cfg)
    
    net_utils.load_network(network,
                           cfg.trained_model_dir,
                           resume=cfg.resume,
                           epoch=cfg.test.epoch)
    network.eval()

    data_loader = make_data_loader(cfg)
    evaluator = make_evaluator(cfg)
    net_time = []
    for batch in tqdm.tqdm(data_loader):
        # for k in batch:
        #     if k != 'meta':
        #         batch[k] = batch[k].cuda()
        with torch.no_grad():
            #torch.cuda.synchronize()
            start_time = time.time()
            #output = network(batch)
            #torch.cuda.synchronize()
            end_time = time.time()
        net_time.append(end_time - start_time)
        #evaluator.evaluate(output, batch)
    #evaluator.summarize()

    

    if len(net_time) > 1:
        print('net_time: ', np.mean(net_time[1:]))
        print('fps: ', 1./np.mean(net_time[1:]))
    else:
        print('net_time: ', np.mean(net_time))
        print('fps: ', 1./np.mean(net_time))

def run_visualize():
    from lib.networks import make_network
    from lib.datasets import make_data_loader
    from lib.utils.net_utils import load_network
    from lib.utils import net_utils
    import tqdm
    import torch
    from lib.visualizers import make_visualizer
    from lib.utils.data_utils import to_cuda

    network = make_network(cfg)
    load_network(network,
                 cfg.trained_model_dir,
                 resume=cfg.resume,
                 epoch=cfg.test.epoch)
    network.eval()

    data_loader = make_data_loader(cfg, is_train=False)
    visualizer = make_visualizer(cfg)
    for batch in tqdm.tqdm(data_loader):
        #batch = to_cuda(batch)
        with torch.no_grad():
            output = network(batch)
        visualizer.visualize(output, batch)
    if visualizer.write_video:
        visualizer.summarize()

if __name__ == '__main__':
    globals()['run_' + args.type]()

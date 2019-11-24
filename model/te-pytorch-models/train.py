import argparse
import collections
import torch
import numpy as np
import transformers
import model.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from model.trainer import Trainer
from parse_config import ConfigParser
import os

# hack to fix OMP error on Mac
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

def main(config):
    logger = config.get_logger('train')

    # setup data_loader instances
    data_loader = config.init_obj('data_loader', module_data, split="train")
    valid_data_loader = config.init_obj('data_loader', module_data, split="validation")

    # build model architecture, then print to console
    model = config.init_obj('arch', module_arch)
    #logger.info(model)

    # get function handles of loss and metrics
    criterion = getattr(module_loss, config['loss'])
    sentence_metrics = [getattr(module_metric, met) for met in config['sentence_metrics']]
    term_metrics = [getattr(module_metric, met) for met in config['term_metrics']]

    # build optimizer, learning rate scheduler. 
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    #optimizer = config.init_obj('optimizer', torch.optim, trainable_params)
    optimizer = config.init_obj('optimizer', transformers.optimization, trainable_params)

    lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)

    trainer = Trainer(model, criterion, sentence_metrics, term_metrics, optimizer,
                      config=config,
                      data_loader=data_loader,
                      valid_data_loader=valid_data_loader)
                      #lr_scheduler=lr_scheduler)

    trainer.train()


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size')
    ]
    config = ConfigParser.from_args(args, options)
    main(config)

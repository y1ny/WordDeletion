import logging
import argparse
from train import train
from test import test
from utils.load_config import init_logging, read_config
from dataset.preprocess_data import PreprocessData
init_logging()
logger = logging.getLogger(__name__)


def preprocess(config_path):
    logger.info('------------Preprocess dataset--------------')
    logger.info('loading config file...')
    global_config = read_config(config_path)

    logger.info('preprocess data...')
    pdata = PreprocessData(global_config)
    pdata.run()


parser = argparse.ArgumentParser(description="preprocess/train/test the model")
parser.add_argument('--mode', '-m', required=False, help='preprocess or train or test')
parser.add_argument('--config', '-c', required=False, dest='config_path', default='config/global_config.yaml')
parser.add_argument('--output', '-o', required=False, dest='out_path')
parser.add_argument('--id', '-i', required=False, dest='id', default='0')
parser.add_argument('--n_sample', '-n', required=False, default=800, help='training sample size')

args = parser.parse_args()

if args.mode == 'preprocess':
    preprocess(args.config_path)
elif args.mode == 'train':
    train(args.config_path, args.id, args.n_sample)
elif args.mode == 'test':
    test(config_path=args.config_path, out_path=args.out_path, id=args.id, n_sample=args.n_sample)
else:
    raise ValueError('Unrecognized mode selected.')


import os
import argparse
import yaml

# define arguements
parser = argparse.ArgumentParser(description='command setting')
parser.add_argument('--phase', type=str, default='train')
parser.add_argument('--data', type=str, default='plate_stress_DG')
parser.add_argument('--model', type=str, default='GANO')
parser.add_argument('--geo_node', type=str, default='vary_bound', choices=['vary_bound', 'all_bound', 'all_domain'])
parser.add_argument('--debug-cuda', action='store_true', help='Enable CUDA debug flags (slow).')

args = parser.parse_args()

if args.debug_cuda:
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    os.environ['TORCH_USE_CUDA_DSA'] = "1"

import torch

from lib.model_plate import GANO
from lib.utils_plate_train import train
from lib.utils_data import generate_plate_stress_data_loader

print('Model forward phase: {}'.format(args.phase))
print('Using dataset: {}'.format(args.data))
print('Using model: {}'.format(args.model))
print('Using geo_node: {}'.format(args.geo_node))

# extract configuration
with open(r'./configs/{}_{}.yaml'.format(args.model, args.data), 'r') as stream:
    config = yaml.load(stream, yaml.FullLoader)

# define device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# define model
# if args.model == 'DCON':
#     model = DCON(config)
if args.model == 'GANO':
    model = GANO(config)
# if args.model == 'self_defined':
#     model = New_model_plate(config)

# extract the data
train_loader, val_loader, test_loader, node_counts = generate_plate_stress_data_loader(args, config)

# train solution function
train(args, config, model, device, (train_loader, val_loader, test_loader), node_counts)

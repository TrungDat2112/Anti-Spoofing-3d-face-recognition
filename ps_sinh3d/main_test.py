from __future__ import print_function, division
from modules.model.model_utils import *
from modules.builder.builder import builder
from modules.io.dataio import dataio
import argparse
import time
import torch
import sys

sys.argv = [sys.argv[0]]
sys.argv += ['--session_name', 'hello_cosmo', '--target', 'normal', '--checkpoint', '/content/drive/MyDrive/SDM-UniPS/checkpoint',
             '--max_image_res', '512', '--max_image_num', '3', '--test_ext', '.data', '--train_dir', 'test_data',
             '--test_dir', 'test_data', '--test_prefix', 'L*', '--mask_margin', '8',
             '--canonical_resolution', '256', '--pixel_samples', '2048']

parser = argparse.ArgumentParser()

# Properties
parser.add_argument('--session_name', default='sdm_unips')
parser.add_argument('--target', default='normal', choices=['normal', 'brdf', 'normal_and_brdf'])
parser.add_argument('--checkpoint', default='')

# Data Configuration
parser.add_argument('--max_image_res', type=int, default=2048)
parser.add_argument('--max_image_num', type=int, default=3)
parser.add_argument('--test_ext', default='.data')
parser.add_argument('--train_dir', default='')
parser.add_argument('--train_dir2', default='')
parser.add_argument('--test_dir', default='')
parser.add_argument('--test_prefix', default='L*')
parser.add_argument('--mask_margin', type=int, default=8)

# Network Configuration
parser.add_argument('--canonical_resolution', type=int, default=256)
parser.add_argument('--pixel_samples', type=int, default=10000)
parser.add_argument('--scalable', action='store_true')

def main():
    args = parser.parse_args()
    print(f'\nStarting a session: {args.session_name}')
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    # import torch_xla.core.xla_model as xm
    # device = xm.xla_device()
    
    print("device:", device)
    sdf_unips = builder(args, device)
    train_data = dataio('Train', args)
    test_data = dataio('Test', args)
    model_save_path = 'model_weights'

    start_time = time.time()
    for epoch in range(1, 11):
        sdf_unips.run(train_data=train_data, test_data=test_data, epoch=epoch, model_save_path=model_save_path)
    end_time = time.time()

    print(f"Prediction finished (Elapsed time is {end_time - start_time:.3f} sec)")

if __name__ == '__main__':
    main()
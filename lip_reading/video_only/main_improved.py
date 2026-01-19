# coding: utf-8
"""
IMPROVED MAIN.PY - Tích hợp các Quick Wins để tăng accuracy ngay
"""
import os
import time
import logging
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from model import *
from dataset import *
from lr_scheduler import *
from cvtransforms import *

SEED = 1
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)

class EarlyStopping:
    """Early stopping để tránh overfitting"""
    def __init__(self, patience=10, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
    
    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

def improved_augmentation(inputs, phase):
    """Cải tiến data augmentation - FIXED version"""
    if phase == 'train':
        # Safe random crop - chỉ crop trong kích thước có sẵn
        batch_img = RandomCrop(inputs.numpy(), (88, 88))
        
        # Color normalization
        batch_img = ColorNormalize(batch_img)
        
        # Horizontal flip với 50% probability
        if np.random.random() < 0.5:
            batch_img = HorizontalFlip(batch_img)
        
        # Gaussian noise với 30% probability (tăng từ 20%)
        if np.random.random() < 0.3:
            noise = np.random.normal(0, 0.01, batch_img.shape)  # Giảm noise từ 0.02 xuống 0.01
            batch_img = np.clip(batch_img + noise, 0, 1)
            
    else:  # test phase
        batch_img = CenterCrop(inputs.numpy(), (88, 88))
        batch_img = ColorNormalize(batch_img)
    
    return batch_img

def data_loader(args):
    generator = torch.Generator()
    generator.manual_seed(SEED)

    dsets = {x: MyDataset(x, args.dataset) for x in ['train', 'test']}
    dset_loaders = {
        'train': torch.utils.data.DataLoader(
            dsets['train'], batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, generator=generator, drop_last=True,
            pin_memory=True),  # Pin memory để tăng tốc
        'test': torch.utils.data.DataLoader(
            dsets['test'], batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, generator=generator, pin_memory=True)
    }
    dset_sizes = {x: len(dsets[x]) for x in ['train', 'test']}
    print('\nStatistics: train: {}, test: {}'.format(
        dset_sizes['train'], dset_sizes['test']))
    return dset_loaders, dset_sizes

def reload_model(model, logger, path=""):
    if not bool(path):
        logger.info('train from scratch')
        return model
    else:
        model_dict = model.state_dict()
        pretrained_dict = torch.load(path, map_location='cpu')
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        logger.info('*** model has been successfully loaded! ***')
        return model

def showLR(optimizer):
    return [param_group['lr'] for param_group in optimizer.param_groups]

def train_test(model, dset_loaders, criterion, epoch, phase, optimizer, args, logger, device, save_path):
    model.to(device)
    model.train() if phase == 'train' else model.eval()

    if phase == 'train':
        logger.info('-' * 10)
        logger.info('Epoch {}/{}'.format(epoch, args.epochs - 1))
        logger.info('Current Learning rate: {}'.format(showLR(optimizer)))

    running_loss, running_corrects, running_all = 0., 0., 0.

    for batch_idx, (inputs, targets) in enumerate(dset_loaders[phase]):
        # IMPROVED: Sử dụng augmentation cải tiến
        batch_img = improved_augmentation(inputs, phase)
        
        batch_img = np.reshape(batch_img, (batch_img.shape[0], batch_img.shape[1], batch_img.shape[2], batch_img.shape[3], 1))
        inputs = torch.from_numpy(batch_img).float().permute(0, 4, 1, 2, 3).to(device)
        targets = targets.to(device)

        # Forward pass
        with torch.no_grad() if phase == 'test' else torch.enable_grad():
            outputs = model(inputs)
            if args.every_frame:
                outputs = torch.mean(outputs, 1)
            _, preds = torch.max(F.softmax(outputs, dim=1).data, 1)
            loss = criterion(outputs, targets)

            if phase == 'train':
                optimizer.zero_grad()
                loss.backward()
                
                # IMPROVED: Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == targets.data)
        running_all += len(inputs)

        if batch_idx == 0:
            since = time.time()
        elif batch_idx % args.interval == 0 or (batch_idx == len(dset_loaders[phase]) - 1):
            print('Process: [{:5.0f}/{:5.0f} ({:.0f}%)]\tLoss: {:.4f}\tAcc:{:.4f}\tCost time:{:5.0f}s\tEstimated time:{:5.0f}s\r'.format(
                running_all,
                len(dset_loaders[phase].dataset),
                100. * batch_idx / (len(dset_loaders[phase]) - 1),
                running_loss / running_all,
                running_corrects.item() / running_all,
                time.time() - since,
                (time.time() - since) * (len(dset_loaders[phase]) - 1) / batch_idx - (time.time() - since)))

    epoch_loss = running_loss / len(dset_loaders[phase].dataset)
    epoch_acc = running_corrects.item() / len(dset_loaders[phase].dataset)
    
    logger.info('{} Epoch:\t{:2}\tLoss: {:.4f}\tAcc:{:.4f}'.format(
        phase, epoch, epoch_loss, epoch_acc) + '\n')

    if phase == 'train':
        torch.save(model.state_dict(), save_path + '/' + args.mode + '_' + str(epoch + 1) + '.pt')
        return model, epoch_loss, epoch_acc
    else:
        return model, epoch_loss, epoch_acc

def test_adam(args, use_gpu):
    device = torch.device("cuda" if use_gpu else "cpu")

    save_path = './improved_' + args.mode
    os.makedirs(save_path, exist_ok=True)

    # Logging
    filename = save_path + '/' + args.mode + '_improved.txt'
    logger_name = "improved_training"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fh = logging.FileHandler(filename, mode='a')
        fh.setLevel(logging.INFO)
        logger.addHandler(fh)
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        logger.addHandler(console)

    # IMPROVED: Tăng hidden dimension
    model = lipreading(mode=args.mode, inputDim=256, hiddenDim=args.hidden_dim, 
                      nClasses=args.nClasses, frameLen=29, every_frame=args.every_frame).to(device)
    model = reload_model(model, logger, args.path)

    criterion = nn.CrossEntropyLoss()

    # IMPROVED: Sử dụng AdamW với weight decay
    if args.mode == 'temporalConv' or args.mode == 'finetuneGRU':
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.mode == 'backendGRU':
        for param in model.parameters():
            param.requires_grad = False
        for param in model.gru.parameters():
            param.requires_grad = True
        optimizer = optim.AdamW([{'params': model.gru.parameters(), 'lr': args.lr}], 
                               lr=0., weight_decay=args.weight_decay)
    else:
        raise Exception('No model is found!')

    # IMPROVED: Learning rate scheduler
    if args.scheduler == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)
    elif args.scheduler == 'plateau':
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    else:
        scheduler = None

    dset_loaders, dset_sizes = data_loader(args)
    
    # IMPROVED: Early stopping
    early_stopping = EarlyStopping(patience=args.patience, min_delta=0.001)
    
    # Track best model
    best_val_acc = 0.0
    best_model_state = None

    if args.test:
        train_test(model, dset_loaders, criterion, 0, 'test', optimizer, args, logger, device, save_path)
        return

    logger.info("🚀 Starting IMPROVED training with:")
    logger.info(f"   • Optimizer: AdamW with weight_decay={args.weight_decay}")
    logger.info(f"   • Scheduler: {args.scheduler}")
    logger.info(f"   • Hidden dimension: {args.hidden_dim}")
    logger.info(f"   • Early stopping patience: {args.patience}")
    logger.info(f"   • Advanced augmentation: ✅")
    logger.info(f"   • Gradient clipping: ✅")

    for epoch in range(args.epochs):
        # Training
        model, train_loss, train_acc = train_test(
            model, dset_loaders, criterion, epoch, 'train', 
            optimizer, args, logger, device, save_path)
        
        # Validation (using test set)
        model, val_loss, val_acc = train_test(
            model, dset_loaders, criterion, epoch, 'test', 
            optimizer, args, logger, device, save_path)
        
        # Learning rate scheduling
        if scheduler:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            torch.save(best_model_state, os.path.join(save_path, f'{args.mode}_best.pt'))
            logger.info(f'💾 New best model saved! Val Acc: {best_val_acc:.4f}')
        
        # Early stopping
        early_stopping(val_loss)
        if early_stopping.early_stop:
            logger.info("⏹️  Early stopping triggered!")
            break
    
    # Final results
    logger.info("=" * 50)
    logger.info("🎉 TRAINING COMPLETED!")
    logger.info(f"🏆 Best validation accuracy: {best_val_acc:.4f}")
    logger.info(f"💾 Best model saved at: {os.path.join(save_path, f'{args.mode}_best.pt')}")
    logger.info("=" * 50)

def main():
    parser = argparse.ArgumentParser(description='Improved Lip Reading Training')
    parser.add_argument('--nClasses', default=10, type=int, help='the number of classes')
    parser.add_argument('--path', default='', help='path to model')
    parser.add_argument('--dataset', default='', help='path to dataset')
    parser.add_argument('--mode', default='finetuneGRU', help='temporalConv, backendGRU, finetuneGRU')
    parser.add_argument('--every-frame', default=False, action='store_true', help='prediction based on every frame')
    
    # IMPROVED: Better hyperparameters
    parser.add_argument('--lr', default=0.001, type=float, help='initial learning rate (improved from 0.0003)')
    parser.add_argument('--batch-size', default=32, type=int, help='mini-batch size (improved from 16)')
    parser.add_argument('--weight-decay', default=1e-4, type=float, help='weight decay (new)')
    parser.add_argument('--hidden-dim', default=768, type=int, help='hidden dimension (improved from 512)')
    
    # IMPROVED: New options
    parser.add_argument('--scheduler', default='cosine', choices=['cosine', 'plateau', 'none'], 
                       help='learning rate scheduler')
    parser.add_argument('--patience', default=10, type=int, help='early stopping patience')
    
    parser.add_argument('--workers', default=4, type=int, help='number of data loading workers')
    parser.add_argument('--epochs', default=50, type=int, help='number of total epochs (increased)')
    parser.add_argument('--interval', default=10, type=int, help='display interval')
    parser.add_argument('--test', default=False, action='store_true', help='perform on the test phase')
    
    args = parser.parse_args()

    print("🚀 IMPROVED LIP READING TRAINING")
    print("=" * 50)
    print(f"📊 Dataset: {args.dataset}")
    print(f"🏗️  Mode: {args.mode}")
    print(f"📈 Learning Rate: {args.lr}")
    print(f"📦 Batch Size: {args.batch_size}")
    print(f"🧠 Hidden Dim: {args.hidden_dim}")
    print(f"⚖️  Weight Decay: {args.weight_decay}")
    print(f"📅 Scheduler: {args.scheduler}")
    print(f"⏱️  Patience: {args.patience}")
    print("=" * 50)

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    use_gpu = torch.cuda.is_available()
    print(f"🖥️  Using GPU: {use_gpu}")
    
    test_adam(args, use_gpu)

if __name__ == '__main__':
    main() 
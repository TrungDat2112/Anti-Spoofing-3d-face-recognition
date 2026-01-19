#!/usr/bin/env python3
"""
Script training cải tiến để tăng độ chính xác model lip reading
Bao gồm các techniques hiện đại: data augmentation, regularization, optimization
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
from model import lipreading
from dataset import MyDataset
from cvtransforms import *

# Improved data augmentation
class ImprovedAugmentation:
    def __init__(self, training=True):
        self.training = training
    
    def __call__(self, batch_img):
        if self.training:
            # Random crop với multiple scales
            crop_size = np.random.choice([84, 88, 92])
            batch_img = RandomCrop(batch_img, (crop_size, crop_size))
            if crop_size != 88:
                # Resize về 88x88
                resized_batch = []
                for i in range(batch_img.shape[0]):
                    resized_frames = []
                    for j in range(batch_img.shape[1]):
                        import cv2
                        frame = cv2.resize(batch_img[i, j], (88, 88))
                        resized_frames.append(frame)
                    resized_batch.append(np.stack(resized_frames, axis=0))
                batch_img = np.stack(resized_batch, axis=0)
            
            # Color normalization
            batch_img = ColorNormalize(batch_img)
            
            # Horizontal flip với probability 0.5
            if np.random.random() < 0.5:
                batch_img = HorizontalFlip(batch_img)
            
            # Temporal augmentation - random frame dropping
            if np.random.random() < 0.3:
                batch_img = self.temporal_dropout(batch_img)
            
            # Noise injection
            if np.random.random() < 0.2:
                batch_img = self.add_gaussian_noise(batch_img)
                
        else:
            # Test time: chỉ center crop và normalize
            batch_img = CenterCrop(batch_img, (88, 88))
            batch_img = ColorNormalize(batch_img)
        
        return batch_img
    
    def temporal_dropout(self, batch_img, drop_rate=0.1):
        """Randomly drop some frames and interpolate"""
        T = batch_img.shape[1]
        num_drop = int(T * drop_rate)
        
        for i in range(batch_img.shape[0]):
            drop_indices = np.random.choice(T, num_drop, replace=False)
            for idx in drop_indices:
                if idx == 0:
                    batch_img[i, idx] = batch_img[i, idx + 1]
                elif idx == T - 1:
                    batch_img[i, idx] = batch_img[i, idx - 1]
                else:
                    # Linear interpolation
                    batch_img[i, idx] = (batch_img[i, idx - 1] + batch_img[i, idx + 1]) / 2
        
        return batch_img
    
    def add_gaussian_noise(self, batch_img, noise_factor=0.05):
        """Add small amount of Gaussian noise"""
        noise = np.random.normal(0, noise_factor, batch_img.shape)
        return np.clip(batch_img + noise, 0, 1)

class FocalLoss(nn.Module):
    """Focal Loss để xử lý class imbalance"""
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class LabelSmoothingLoss(nn.Module):
    """Label Smoothing để tránh overconfident predictions"""
    def __init__(self, num_classes, smoothing=0.1):
        super(LabelSmoothingLoss, self).__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
    
    def forward(self, pred, target):
        pred = F.log_softmax(pred, dim=1)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.num_classes - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=1))

class EarlyStopping:
    """Early stopping để tránh overfitting"""
    def __init__(self, patience=7, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

def improved_data_loader(args):
    """Data loader với cải tiến"""
    generator = torch.Generator()
    generator.manual_seed(42)

    dsets = {x: MyDataset(x, args.dataset) for x in ['train', 'test']}
    
    # Sử dụng stratified sampling nếu có thể
    dset_loaders = {
        'train': torch.utils.data.DataLoader(
            dsets['train'], 
            batch_size=args.batch_size, 
            shuffle=True,
            num_workers=args.workers, 
            generator=generator, 
            drop_last=True,
            pin_memory=True  # Tăng tốc transfer GPU
        ),
        'test': torch.utils.data.DataLoader(
            dsets['test'], 
            batch_size=args.batch_size, 
            shuffle=False,
            num_workers=args.workers, 
            generator=generator,
            pin_memory=True
        )
    }
    
    dset_sizes = {x: len(dsets[x]) for x in ['train', 'test']}
    print(f'Dataset sizes: train: {dset_sizes["train"]}, test: {dset_sizes["test"]}')
    return dset_loaders, dset_sizes

def improved_train_test(model, dset_loaders, criterion, epoch, phase, optimizer, args, logger, device, save_path, scheduler=None):
    """Training/testing function với cải tiến"""
    model.train() if phase == 'train' else model.eval()
    
    augmentation = ImprovedAugmentation(training=(phase == 'train'))
    
    running_loss = 0.0
    running_corrects = 0
    running_all = 0
    
    # Mixup parameters
    mixup_alpha = 0.2 if phase == 'train' and args.mixup else 0
    
    for batch_idx, (inputs, targets) in enumerate(dset_loaders[phase]):
        # Improved augmentation
        batch_img = augmentation(inputs.numpy())
        
        # Convert to tensor
        batch_img = np.reshape(batch_img, (batch_img.shape[0], batch_img.shape[1], 
                                         batch_img.shape[2], batch_img.shape[3], 1))
        inputs = torch.from_numpy(batch_img).float().permute(0, 4, 1, 2, 3).to(device)
        targets = targets.to(device)
        
        # Mixup augmentation (chỉ cho training)
        if mixup_alpha > 0:
            inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, mixup_alpha)
        
        # Forward pass
        with torch.no_grad() if phase == 'test' else torch.enable_grad():
            outputs = model(inputs)
            if args.every_frame:
                outputs = torch.mean(outputs, 1)
            
            # Calculate loss
            if mixup_alpha > 0 and phase == 'train':
                loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
            else:
                loss = criterion(outputs, targets)
            
            # Backward pass (chỉ cho training)
            if phase == 'train':
                optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping để tránh exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
        
        # Statistics
        running_loss += loss.item() * inputs.size(0)
        if mixup_alpha > 0 and phase == 'train':
            # Với mixup, tính accuracy khó hơn
            _, preds = torch.max(outputs, 1)
            running_corrects += (lam * preds.eq(targets_a).cpu().sum().float() + 
                               (1 - lam) * preds.eq(targets_b).cpu().sum().float())
        else:
            _, preds = torch.max(F.softmax(outputs, dim=1), 1)
            running_corrects += torch.sum(preds == targets.data)
        
        running_all += len(inputs)
        
        # Progress reporting
        if batch_idx % args.interval == 0:
            print(f'Process: [{running_all:5.0f}/{len(dset_loaders[phase].dataset):5.0f} '
                  f'({100. * batch_idx / (len(dset_loaders[phase]) - 1):.0f}%)]\t'
                  f'Loss: {running_loss / running_all:.4f}\t'
                  f'Acc: {running_corrects.item() / running_all:.4f}')
    
    # Epoch statistics
    epoch_loss = running_loss / len(dset_loaders[phase].dataset)
    epoch_acc = running_corrects.item() / len(dset_loaders[phase].dataset)
    
    logger.info(f'{phase} Epoch: {epoch:2}\tLoss: {epoch_loss:.4f}\tAcc: {epoch_acc:.4f}')
    
    # Learning rate scheduling
    if phase == 'test' and scheduler:
        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(epoch_loss)
        else:
            scheduler.step()
    
    # Save model (chỉ cho training)
    if phase == 'train':
        torch.save(model.state_dict(), os.path.join(save_path, f'{args.mode}_epoch_{epoch + 1}.pt'))
    
    return model, epoch_loss, epoch_acc

def mixup_data(x, y, alpha=1.0):
    """Mixup augmentation"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Mixup loss calculation"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def improved_training(args, use_gpu):
    """Main training function với cải tiến"""
    device = torch.device("cuda" if use_gpu else "cpu")
    
    # Create save directory
    save_path = f'./improved_{args.mode}'
    os.makedirs(save_path, exist_ok=True)
    
    # Setup logging
    logger = setup_logger(save_path, args.mode)
    
    # Model initialization
    model = lipreading(
        mode=args.mode, 
        inputDim=256, 
        hiddenDim=args.hidden_dim,  # Tăng hidden dimension
        nClasses=args.nClasses, 
        frameLen=29, 
        every_frame=args.every_frame
    ).to(device)
    
    # Load pretrained weights if available
    if args.pretrained:
        model = load_pretrained(model, args.pretrained, logger)
    
    # Loss function với cải tiến
    if args.focal_loss:
        criterion = FocalLoss(alpha=1, gamma=2)
    elif args.label_smoothing:
        criterion = LabelSmoothingLoss(args.nClasses, smoothing=0.1)
    else:
        criterion = nn.CrossEntropyLoss()
    
    # Optimizer với cải tiến
    if args.optimizer == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(), 
            lr=args.lr, 
            weight_decay=args.weight_decay,
            betas=(0.9, 0.999)
        )
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=0.9,
            weight_decay=args.weight_decay,
            nesterov=True
        )
    else:
        optimizer = optim.Adam(
            model.parameters(), 
            lr=args.lr, 
            weight_decay=args.weight_decay
        )
    
    # Learning rate scheduler
    if args.scheduler == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)
    elif args.scheduler == 'plateau':
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    else:
        scheduler = None
    
    # Data loaders
    dset_loaders, dset_sizes = improved_data_loader(args)
    
    # Early stopping
    early_stopping = EarlyStopping(patience=args.patience, min_delta=0.001)
    
    # Training loop
    best_acc = 0.0
    best_model_state = None
    
    for epoch in range(args.epochs):
        logger.info(f'Epoch {epoch}/{args.epochs - 1}')
        logger.info('-' * 20)
        
        # Training phase
        model, train_loss, train_acc = improved_train_test(
            model, dset_loaders, criterion, epoch, 'train', 
            optimizer, args, logger, device, save_path, scheduler
        )
        
        # Validation phase (test được dùng làm validation)
        model, val_loss, val_acc = improved_train_test(
            model, dset_loaders, criterion, epoch, 'test', 
            optimizer, args, logger, device, save_path, scheduler
        )
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_state = model.state_dict().copy()
            torch.save(best_model_state, os.path.join(save_path, f'{args.mode}_best.pt'))
            logger.info(f'New best model saved with accuracy: {best_acc:.4f}')
        
        # Early stopping check
        early_stopping(val_loss)
        if early_stopping.early_stop:
            logger.info("Early stopping triggered")
            break
    
    # Load best model for final evaluation
    if best_model_state:
        model.load_state_dict(best_model_state)
        logger.info(f'Best validation accuracy: {best_acc:.4f}')
    
    return model

def setup_logger(save_path, mode):
    """Setup logger"""
    filename = os.path.join(save_path, f'{mode}_improved.log')
    logger = logging.getLogger("improved_training")
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        fh = logging.FileHandler(filename, mode='a')
        fh.setLevel(logging.INFO)
        
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        console.setFormatter(formatter)
        
        logger.addHandler(fh)
        logger.addHandler(console)
    
    return logger

def load_pretrained(model, pretrained_path, logger):
    """Load pretrained weights"""
    try:
        pretrained_dict = torch.load(pretrained_path, map_location='cpu')
        model_dict = model.state_dict()
        
        # Filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        
        logger.info(f'Successfully loaded pretrained weights from {pretrained_path}')
    except Exception as e:
        logger.warning(f'Failed to load pretrained weights: {e}')
    
    return model

def main():
    parser = argparse.ArgumentParser(description='Improved Lip Reading Training')
    
    # Basic parameters
    parser.add_argument('--nClasses', default=10, type=int, help='Number of classes')
    parser.add_argument('--dataset', required=True, help='Path to dataset')
    parser.add_argument('--mode', default='finetuneGRU', help='Model mode')
    parser.add_argument('--every-frame', default=False, action='store_true')
    
    # Training parameters
    parser.add_argument('--epochs', default=50, type=int, help='Number of epochs')
    parser.add_argument('--batch-size', default=16, type=int, help='Batch size')
    parser.add_argument('--lr', default=0.001, type=float, help='Learning rate')
    parser.add_argument('--weight-decay', default=1e-4, type=float, help='Weight decay')
    parser.add_argument('--hidden-dim', default=512, type=int, help='Hidden dimension')
    
    # Optimization
    parser.add_argument('--optimizer', default='adamw', choices=['adam', 'adamw', 'sgd'])
    parser.add_argument('--scheduler', default='cosine', choices=['cosine', 'plateau', 'none'])
    parser.add_argument('--patience', default=10, type=int, help='Early stopping patience')
    
    # Augmentation
    parser.add_argument('--mixup', action='store_true', help='Use mixup augmentation')
    parser.add_argument('--focal-loss', action='store_true', help='Use focal loss')
    parser.add_argument('--label-smoothing', action='store_true', help='Use label smoothing')
    
    # Other
    parser.add_argument('--pretrained', default='', help='Path to pretrained model')
    parser.add_argument('--workers', default=4, type=int, help='Number of workers')
    parser.add_argument('--interval', default=10, type=int, help='Print interval')
    
    args = parser.parse_args()
    
    # Set environment
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    use_gpu = torch.cuda.is_available()
    
    print("🚀 IMPROVED TRAINING STARTED")
    print("=" * 50)
    print(f"Dataset: {args.dataset}")
    print(f"Mode: {args.mode}")
    print(f"Optimizer: {args.optimizer}")
    print(f"Scheduler: {args.scheduler}")
    print(f"Mixup: {args.mixup}")
    print(f"Focal Loss: {args.focal_loss}")
    print(f"Label Smoothing: {args.label_smoothing}")
    print("=" * 50)
    
    model = improved_training(args, use_gpu)
    print("✅ Training completed!")

if __name__ == '__main__':
    main() 
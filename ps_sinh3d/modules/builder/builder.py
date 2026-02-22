import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
import numpy as np
from matplotlib import cm
import math
import os
import shutil
import time
import matplotlib.pyplot as plt
from modules.net.net import Net
from modules.utils.util import *


def colorMap(diff):
    thres = 90
    diff_norm = np.clip(diff, 0, thres) / thres
    diff_cm = torch.from_numpy(cm.jet(diff_norm.numpy()))[:,:,:, :3]
    return diff_cm.permute(0,3,1,2).clone().float()

def calNormalAcc(gt_n, pred_n, mask=None):
    """Tensor Dim: NxCxHxW"""
    dot_product = (gt_n * pred_n).sum(1).clamp(-1,1)
    error_map   = torch.acos(dot_product) # [-pi, pi]
    angular_map = error_map * 180.0 / math.pi
    angular_map = angular_map * mask.narrow(1, 0, 1).squeeze(1)

    valid = mask.narrow(1, 0, 1).sum()
    ang_valid  = angular_map[mask.narrow(1, 0, 1).squeeze(1).bool()]
    n_err_mean = ang_valid.sum() / valid
    n_err_med  = ang_valid.median()
    n_acc_11   = (ang_valid < 11.25).sum().float() / valid
    n_acc_30   = (ang_valid < 30).sum().float() / valid
    n_acc_45   = (ang_valid < 45).sum().float() / valid

    angular_map = colorMap(angular_map.cpu().squeeze(1))
    value = {'n_err_mean': n_err_mean.item(),
            'n_acc_11': n_acc_11.item(), 'n_acc_30': n_acc_30.item(), 'n_acc_45': n_acc_45.item()}
    angular_error_map = {'angular_map': angular_map}
    return value, angular_error_map


class builder():
    def __init__(self, args, device):
        self.device = device
        print('DEVICE-BUILDER: self.device:', self.device)
        self.args = args
        self.mode = 'Train'
        if 'normal' in args.target:
            model_dir = f'{args.checkpoint}/normal'
            self.net_nml = Net(args.pixel_samples, 'normal', device).to(self.device)
            self.net_nml = torch.nn.DataParallel(self.net_nml)
            self.net_nml = loadmodel(self.net_nml, 'checkpoint/final1.pth', strict=False)
            total_params = sum(p.numel() for p in self.net_nml.parameters())
            print(total_params)

    def separate_batch(self, batch):
        I = batch[0].to(self.device) # [B, 3, H, W]
        N = batch[1].to(self.device) # [B, 1, H, W]
        M = batch[2].to(self.device) # [B, 1, H, W]
        nImgArray = batch[3].to(self.device)
        roi = batch[4]
        return I, N, M, nImgArray, roi
    
    def run(self, train_data=None, test_data=None, epoch=9, model_save_path=None, canonical_resolution=256):

        # train_data_loader = DataLoader(train_data, batch_size = 1, shuffle=False, num_workers=0, pin_memory=False)
        test_data_loader = DataLoader(test_data, batch_size=1, shuffle=True, num_workers=0, pin_memory=False)

        total_samples = len(train_data)
        indices = list(range(total_samples))
        np.random.shuffle(indices)

        train_steps = 38
        test_steps = 9600
        batch_size = 1

        total_train_loss = 0.0
        total_train_mae = 0.0
        total_train_batches = 0
        batch_count = 0
        remaining_samples = 0 # Số lượng mẫu đã train
        used_samples = 0

        print(f"--- Starting Training Epoch {epoch} ---")
        while used_samples < total_samples:
            start_time = time.time()
            remaining_samples = total_samples - used_samples
            if remaining_samples < train_steps:
                subset_indices = indices[used_samples:used_samples + remaining_samples]
                test_steps = remaining_samples // 2

            else:
                subset_indices = indices[used_samples:used_samples + train_steps]
            sampler = SubsetRandomSampler(subset_indices)
            train_data_loader = DataLoader(train_data, batch_size = 1, num_workers=0, pin_memory=False, sampler=sampler)
            
            self.mode = 'Train'
            train_res = []
            losses = 0.0
            cnt = 0
            
            for i, batch_train in enumerate(train_data_loader):
                I, N, M, nImgArray, roi = self.separate_batch(batch_train)
                roi = roi[0].cpu().numpy()
                h_ = roi[0]; w_ = roi[1]; r_s = roi[2]; r_e = roi[3]; c_s = roi[4]; c_e = roi[5]
                B, C, H, W, Nimg = I.shape

                # Forward pass
                nout, loss, n_use, m_use  = self.net_nml(I, M, N, nImgArray.reshape(-1,1), decoder_resolution= H * torch.ones(I.shape[0],1), canonical_resolution=canonical_resolution* torch.ones(I.shape[0],1),
                                           mode_n = 'Train')
                # Calculate accuracy and loss
                train_acc, _ = calNormalAcc(n_use.cpu().detach(), nout.cpu().detach(), m_use.cpu().detach())
                iter_res = train_acc['n_err_mean']
                train_res.append(iter_res)
                losses += loss.detach().item()
                cnt += 1
                
                nout = nout.permute(0, 2, 3, 1).squeeze().cpu().numpy()
                n_use = n_use.permute(0, 2, 3, 1).squeeze().cpu().numpy()
                 # nout[:, :, 0] = -nout[:, :, 0]; nout[:, :, 1] = -nout[:, :, 1]; 
                nout = (nout+1)/2
                n_use[:, :, 0] = -n_use[:, :, 0]; n_use[:, :, 1] = -n_use[:, :, 1]; 
                n_use = (n_use+1)/2
                fig, axes = plt.subplots(1, 2, figsize=(10, 5))  # 1 hàng, 2 cột
                axes[0].imshow(nout)
                axes[0].axis('off')
                axes[0].set_title('predict')
                axes[1].imshow(n_use)
                axes[1].axis('off')
                axes[1].set_title(' ')
                plt.tight_layout()
                plt.show()
                print(nout.shape)
            
            # Tổng hợp kết quả huấn luyện
            train_res = np.array(train_res).mean()
            total_train_loss += losses / cnt
            total_train_mae += train_res
            total_train_batches += cnt
            print(f"Epoch {epoch} - Step {batch_count+1} -- Train Loss: {losses / cnt:.4f}, Train MAE: {train_res:.4f}",end=" ")
            used_samples += train_steps
            

            # Kiểm tra điều kiện dừng
            batch_count += 1
            # save_exr(nout, f"local_normal_{batch_count}.exr", channels=['R', 'G', 'B'])
            # nml = (nout * 255).astype(np.uint8)
            # imageio.imwrite(os.path.join('/kaggle/working/', f"local_normal_{batch_count}.png"), nml)

            # Save model sau mỗi vòng train
            if epoch%5 == 0 :
                if not os.path.exists(model_save_path):
                    os.makedirs(model_save_path)
                model_filename = f"{model_save_path}/model_v5dung4_epoch{epoch}.pth"
                torch.save(self.net_nml.state_dict(), model_filename)

                # Sao chép tệp từ working sang output
                output_save_path = 'model_weights'
                if not os.path.exists(output_save_path):
                    os.makedirs(output_save_path)
                
                # Đường dẫn tệp trong thư mục output
                output_model_filename = f"{output_save_path}/model_step_{batch_count}_epoch{epoch}.pth"
                shutil.copy(model_filename, output_model_filename)
            
            end_time = time.time()
            print(f" , Time: {end_time - start_time:.3f} sec, {train_steps} batch")

        # Tổng hợp toàn bộ quá trình
        print(f"--- Training Completed ---")
        print(f"FINAL TRAIN Loss: {total_train_loss / batch_count:.4f}, Final Train MAE: {total_train_mae / batch_count:.4f}")
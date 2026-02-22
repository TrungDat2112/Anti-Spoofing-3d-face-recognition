from torch.cuda.amp import autocast, GradScaler
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import kaiming_normal_, trunc_normal_
from modules.utils.util import *
from modules.model import model
from modules.utils.ind2sub import *
from modules.utils import gauss_filter

class Net(nn.Module):
    def __init__(self, pixel_samples, output, device):
        super().__init__()
        self.device = device
        self.target = output
        self.pixel_samples = pixel_samples
        self.glc_smoothing = False
        self.mode = 'Train'
        self.lr = 2e-5
        self.stype = 'step'
        self.init_encoder = True
        self.init_decoder = True

        self.input_dim = 4 # RGB + mask
        self.image_encoder = model.ScaleInvariantSpatialLightImageEncoder(self.input_dim, use_efficient_attention=False).to(self.device)

        self.input_dim = 3 # RGB only
        self.glc_upsample = model.GLC_Upsample(256+self.input_dim, num_enc_sab=1, dim_hidden=256, dim_feedforward=1024, use_efficient_attention=True).to(self.device)

        self.glc_aggregation = model.GLC_Aggregation(256+self.input_dim, num_agg_transformer=2, dim_aggout=384, dim_feedforward=1024, use_efficient_attention=False).to(self.device)

        self.regressor = model.Regressor(384, num_enc_sab=1, use_efficient_attention=True, dim_feedforward=1024, output=self.target).to(self.device)

        self.set_optimizer()
        self.criteronL2 = nn.MSELoss(reduction = 'sum').to(self.device)

    def set_optimizer(self):
        if self.mode == 'Train':
            [self.image_encoder, self.optimizer_encoder, self.scheduler_encoder] = optimizer_setup_AdamW(self.image_encoder, lr = self.lr, init=True, stype=self.stype)

        if self.mode == 'Train':
            [self.glc_upsample, self.optimizer_glc_upsample, self.scheduler_glc_upsample] = optimizer_setup_AdamW(self.glc_upsample, lr = self.lr, init=True, stype=self.stype)

        if self.mode == 'Train':
            [self.glc_aggregation, self.optimizer_glc_aggregation, self.scheduler_glc_aggregation] = optimizer_setup_AdamW(self.glc_aggregation, lr = self.lr, init=True, stype=self.stype)

        if self.mode == 'Train':
            [self.regressor, self.optimizer_regressor, self.scheduler_regressor] = optimizer_setup_AdamW(self.regressor, lr = self.lr, init=True, stype=self.stype)

    def no_grad(self):
        mode_change(self.image_encoder, False)
        mode_change(self.glc_upsample, False)
        mode_change(self.glc_aggregation, False)
        mode_change(self.regressor, False)

    def set_grad(self):
        mode_change(self.image_encoder, True)
        mode_change(self.glc_upsample, True)
        mode_change(self.glc_aggregation, True)
        mode_change(self.regressor, True)

    def forward(self, I, M, N, nImgArray, decoder_resolution, canonical_resolution, mode_n='Train'):
        decoder_resolution = decoder_resolution[0,0].cpu().numpy().astype(np.int32).item()
        canonical_resolution = canonical_resolution[0,0].cpu().numpy().astype(np.int32).item()

        self.no_grad()

        """init"""
        B, C, H, W, Nmax = I.shape

        """ Image Encoder at Canonical Resolution """
        I_enc = I.permute(0, 4, 1, 2, 3)# B Nmax C H W
        M_enc = M # B 1 H W
        img_index = make_index_list(Nmax, nImgArray) # Extract objects > 0
        I_enc = I_enc.reshape(-1, I_enc.shape[2], I_enc.shape[3], I_enc.shape[4])
        M_enc = M_enc.unsqueeze(1).expand(-1, Nmax, -1, -1, -1).reshape(-1, 1, H, W)
        data = torch.cat([I_enc * M_enc, M_enc], dim=1)
        data = data[img_index==1,:,:,:].to(self.device)       # torch.sze([B, N, 4, H, W])d
        glc = self.image_encoder(data, nImgArray, canonical_resolution) # torch.Size([B, N, 256, H/4, W/4]) [img, mask]
        # print("---model.encoder.shape", glc.shape)

        """ Sample Decoder at Original Resokution"""
        I_dec = []
        M_dec = []
        N_dec = []

        img = I.permute(0, 4, 1, 2, 3).to(self.device)
        mask = M

        decoder_imgsize = (decoder_resolution, decoder_resolution)
        img = img.reshape(-1, img.shape[2], img.shape[3], img.shape[4])
        img = img[img_index==1, :, :, :]
        I_dec = F.interpolate(img, size=decoder_imgsize, mode='bilinear', align_corners=False)
        M_dec = F.interpolate(mask, size=decoder_imgsize, mode='nearest')

        C = img.shape[1]
        H = decoder_imgsize[0]
        W = decoder_imgsize[1]

        nout = torch.zeros(B, H * W, 3).to(self.device)
        n_use = torch.zeros(B, H * W, 3).to(self.device)
        m_use = torch.zeros(B, H * W, 1).to(self.device)

        if self.glc_smoothing:
            f_scale = decoder_resolution//canonical_resolution # (2048/256)
            smoothing = gauss_filter.gauss_filter(glc.shape[1], 10 * f_scale+1, 1).to(glc.device) # channels, kernel_size, sigma
            glc = smoothing(glc)
        p = 0
        losses = 0.
        for b in range(B):
            target = range(p, p+nImgArray[b])
            p = p+nImgArray[b]
            m_ = M_dec[b, :, :, :].reshape(-1, H * W).permute(1,0)
            n_ = N[b, :, :, :].reshape(-1, H * W).permute(1,0)

            ids = np.nonzero(m_>0)[:,0]
            ids = ids[np.random.permutation(len(ids))]
            if len(ids) > self.pixel_samples:
                num_split = len(ids) // self.pixel_samples + 1
                # print("---model.decoder.num_split:",num_split)
                idset = np.array_split(ids, num_split)
            else:
                idset = [ids]

            o_ = I_dec[target, :, :, :].reshape(nImgArray[b], C, H * W).permute(2,0,1)  # [N, c, h, w]]

            loss_num_split = 0.
            num_split_use = 0
            for i,ids in enumerate(idset):
                # if i > 5:
                #     break
                
                o_ids = o_[ids, :, :]
                n_ids = n_[ids, :]
                coords = ind2coords(np.array((H, W)), ids).expand(nImgArray[b],-1,-1,-1)
                glc_ids = F.grid_sample(glc[target, :, :, :], coords.to(self.device), mode='bilinear', align_corners=False).reshape(len(target), -1, len(ids)).permute(2,0,1) # [m, N, f]


                """ glc_ids """
                x = torch.cat([o_ids, glc_ids], dim=2).to(self.device) # [len(ids), N, 256+3]
                glc_ids = self.glc_upsample(x)

                x = torch.cat([o_ids, glc_ids], dim=2).to(self.device) # [len(ids), N, 256+3]
                x = self.glc_aggregation(x)  #[len(ids), 384]

                x_n = self.regressor(x.to(self.device), len(ids)) # [len(ids), 3]

                X_n = F.normalize(x_n.to(self.device), dim=1, p=2)

                loss_num_split += self.criteronL2(X_n.to(self.device), n_ids.to(self.device)) / len(ids)
                # print('--model.loss.num_split:', loss_num_split)

                if self.target == 'normal':
                    nout[b, ids, :] = X_n.detach()
                    n_use[b, ids, :] = n_ids.detach()
                    m_use[b, ids, :] = 1
                
                num_split_use += 1

            losses = loss_num_split / num_split_use

        nout = nout.permute(0, 2, 1).reshape(B, 3, H, W)
        n_use = n_use.permute(0, 2, 1).reshape(B, 3, H, W)
        m_use = m_use.permute(0, 2, 1).reshape(B, 1, H, W)

        # if mode_n == 'Train':
        #     self.optimizer_encoder.zero_grad()
        #     self.optimizer_glc_upsample.zero_grad()
        #     self.optimizer_glc_aggregation.zero_grad()
        #     self.optimizer_regressor.zero_grad()
            
        #     losses.backward()
            
        #     self.optimizer_encoder.step()
        #     self.optimizer_glc_upsample.step()
        #     self.optimizer_glc_aggregation.step()
        #     self.optimizer_regressor.step()

        return nout, losses , n_use, m_use
import os
import torch
import numpy as np
import einops
import Imath
import OpenEXR

def loadmodel(model, filename, strict=True):
    if os.path.exists(filename):
        params = torch.load('%s' % filename, weights_only=True)
        model.load_state_dict(params,strict=strict)
        print('Loading pretrained model... %s ' % filename)
    else:
        print('Pretrained model not Found')
    return model

def mode_change(net, Training=True):
    if Training == True:
        for param in net.parameters():
            param.requires_grad = True
        net.train()
    if Training == False:
        for param in net.parameters():
            param.requires_grad = False
        net.eval()

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

def masking(img, mask):
    # img [B, C, H, W]
    # mask [B, 1, H, W] [0,1]
    img_masked = img * mask.expand((-1, img.shape[1], -1, -1))
    return img_masked

def print_model_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print('# parameters: %d' % params)

def make_index_list(maxNumImages, numImageList):
    index = np.zeros((len(numImageList) * maxNumImages), np.int32)
    for k in range(len(numImageList)):
        index[maxNumImages*k:maxNumImages*k+numImageList[k]] = 1
    return index

def optimizer_setup_AdamW(net, lr = 0.001, init=True, stype='step'):
    print(f'optimizer (AdamW) lr={lr}')
    if init==True:
        net.init_weights()
    optim_params = [{'params': net.parameters(), 'lr': lr},] # confirmed
    optimizer = torch.optim.AdamW(optim_params, betas=(0.9, 0.999), weight_decay=0.01)
    if stype == 'cos':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 30, eta_min=1e-7, last_epoch=-1)
        print('cosine aneealing learning late scheduler')
    if stype == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.8)
        print('step late scheduler x0.8 decay')
    return net, optimizer, scheduler
    
def save_exr(data, path, channels=['R', 'G', 'B']):

    height, width = data.shape[:2]
    header = OpenEXR.Header(width, height)
    header['channels'] = {c: Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)) for c in channels}

    data = (data - np.min(data)) / (np.max(data) - np.min(data))

    exr = OpenEXR.OutputFile(path, header)
    if len(data.shape) == 3 and data.shape[2] == 3:  # 3 kênh (RGB)
        exr.writePixels({c: data[..., i].astype(np.float32).tobytes() for i, c in enumerate(channels)})
    elif len(data.shape) == 2:  # Grayscale
        exr.writePixels({channels[0]: data.astype(np.float32).tobytes()})
    else:
        raise ValueError(f"Unexpected data shape: {data.shape}")
    exr.close()
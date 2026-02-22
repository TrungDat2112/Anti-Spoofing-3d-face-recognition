import glob
import os
import numpy as np
import cv2
import re
import OpenEXR
import Imath
from scipy.ndimage import rotate
import random

class dataloader():
    def __init__(self, numberOfImages = None, outdir = '.', mask_margin=16, ctype='ORTHO'):
        self.mask_margin=mask_margin
        self.numberOfImages = numberOfImages
        self.outdir = outdir
        self.use_mask = True
        self.ctype = ctype

    def read_exr_openexr(self, file_path):
        # Mở file EXR
        exr_file = OpenEXR.InputFile(file_path)

        # Lấy thông tin tiêu đề (header) và kích thước ảnh
        header = exr_file.header()
        dw = header['dataWindow']
        size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
        # Đọc các kênh R, G, B của ảnh EXR
        r = exr_file.channel('R', Imath.PixelType(Imath.PixelType.FLOAT))
        g = exr_file.channel('G', Imath.PixelType(Imath.PixelType.FLOAT))
        b = exr_file.channel('B', Imath.PixelType(Imath.PixelType.FLOAT))

        # Chuyển đổi dữ liệu từ byte thành mảng numpy
        r = np.frombuffer(r, dtype=np.float32).reshape(size)
        g = np.frombuffer(g, dtype=np.float32).reshape(size)
        b = np.frombuffer(b, dtype=np.float32).reshape(size)

        # Kết hợp các kênh lại thành ảnh RGB
        img = np.stack([r, g, b], axis=-1)
        return img

    def read_exr_grayscale(self, file_path):
        exr_file = OpenEXR.InputFile(file_path)
        header = exr_file.header()
        dw = header['dataWindow']
        width = dw.max.x - dw.min.x + 1
        height = dw.max.y - dw.min.y + 1

        # Đọc kênh 'R' hoặc kênh đầu tiên
        channel = 'R' if 'R' in header['channels'] else list(header['channels'].keys())[0]
        data = exr_file.channel(channel, Imath.PixelType(Imath.PixelType.FLOAT))

        img = np.frombuffer(data, dtype=np.float32).reshape(height, width)

        return img
    
    def load(self, objdir, prefix, margin = 0, max_image_resolution = 2048, aug=[]):

        self.objname = re.split(r'\\|/',objdir)[-1]
        # self.data_workspace = f'{self.outdir}/results/{self.objname}'
        # os.makedirs(self.data_workspace, exist_ok=True)

        # print(f'Testing on {self.objname}')

        directlist = []
        [directlist.append(p) for p in glob.glob(objdir + '/%s[!.txt]' % prefix, recursive=True) if os.path.isfile(p)]
        directlist = sorted(directlist)

        if len(directlist) == 0:
            return False
        if os.name == 'posix':
            temp = directlist[0].split("/")
        if os.name == 'nt':
            temp = directlist[0].split("\\")
        img_dir = "/".join(temp[:-1])

        if self.numberOfImages is not None:
            indexset = np.random.permutation(len(directlist))[:self.numberOfImages]
        else:
            indexset = range(len(directlist))
        numberOfImages = np.min([len(indexset), self.numberOfImages])

        flip_type = random.randint(1, 3)
        angle = random.uniform(0, 360)

        for i, indexofimage in enumerate(indexset):
            img_path = directlist[indexofimage]
            mask_path = img_dir + '/mask.png'

            file_extension = os.path.splitext(img_path)[-1].lower()
            if file_extension == ".png":
                img = cv2.imread(img_path, flags=cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
                if len(img.shape) == 2:  # Ảnh chỉ có 1 kênh
                    img = np.expand_dims(img, axis=-1)
                    # img = np.stack([img] * 3, axis=-1)
                elif img.shape[2] == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = img[:, :, i]
                    # img = np.stack([img] * 3, axis=-1)
                    img = np.expand_dims(img, axis=-1)
                    
            if file_extension == ".exr": 
                img = self.read_exr_grayscale(img_path)
                # img = np.stack([img] * 3, axis=-1)
                img = np.expand_dims(img, axis=-1)

            if i == 0:
                h0 = img.shape[0]
                w0 = img.shape[1]
                margin = self.mask_margin

                # nml_path_diligent = img_dir + '/local_normal.png'
                # nml_path_others = img_dir + '/normal.tif'
                nml_path_exr = img_dir + '/local_normal.exr'

                # if os.path.isfile(nml_path_diligent):
                #     nml_path = nml_path_diligent
                # elif os.path.isfile(nml_path_others):
                #     nml_path = nml_path_others
                if os.path.isfile(nml_path_exr):
                    nml_path = nml_path_exr
                else:
                    nml_path = "no_normal"

                # if ground truth normal map is avelable, generate normal-based mask
                mask_flag = False
                if os.path.isfile(nml_path):
                    if nml_path.endswith('.exr'):
                        N = self.read_exr_openexr(nml_path)
                    else:
                        N = cv2.cvtColor(cv2.imread(nml_path, flags=cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH), cv2.COLOR_BGR2RGB)
                    if N.dtype == 'uint8':
                        bit_depth = 255.0
                    if N.dtype == 'uint16':
                        bit_depth = 65535.0
                    else:
                        bit_depth = 1.0
                    # N = self.rotate_and_resize(N, angle, flip_type)
                    N = np.float32(N)/bit_depth
                    if np.min(N) < 0 or np.max(N) > 1:
                        N = (N - np.min(N)) / (np.max(N) - np.min(N))
                    N = 2 * N - 1
                    N[:, :, 0] = -N[:, :, 0]; N[:, :, 1] = -N[:, :, 1]
                    mask = np.float32(np.abs(1 - np.sqrt(np.sum(N * N, axis=2))) < 0.5)
                    N = N/(np.sqrt(np.sum(N * N, axis=2, keepdims=True)) + 1.0e-6)
                    N = N * mask[:, :, np.newaxis]
                    mask_flag = True
                else:
                    N = np.zeros((h0, w0, 3), np.float32)

                if os.path.isfile(mask_path) and i == 0:
                    mask = (cv2.imread(mask_path, flags = cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH) > 0).astype(np.float32)
                    if len(mask.shape) == 3:
                        mask = mask[:, :, 0]
                    mask_flag = True

                # If no mask was created from normal map or binary mask file, create a default one
                if not mask_flag:
                    mask = np.ones((h0, w0), np.float32)

                # Keep mask and normal of the original resolution
                n_true = N

                # Based on object mask, crop boudning rectangular area
                if  mask_flag == True:
                    rows, cols = np.nonzero(mask)
                    rowmin = np.min(rows)
                    rowmax = np.max(rows)
                    row = rowmax - rowmin
                    colmin = np.min(cols)
                    colmax = np.max(cols)
                    col = colmax - colmin
                    if rowmin - margin <= 0 or rowmax + margin > img.shape[0] or colmin - margin <= 0 or colmax + margin > img.shape[1]:
                        flag = False

                    else:
                        flag = True

                    if row > col and flag:
                        r_s = rowmin-margin
                        r_e = rowmax+margin
                        c_s = np.max([colmin- int(0.5 * (row-col))-margin,0])
                        c_e = np.min([colmax+int(0.5 * (row-col))+margin,img.shape[1]])
                    elif col >= row and flag:
                        r_s = np.max([rowmin-int(0.5*(col-row))-margin,0])
                        r_e = np.min([rowmax+int(0.5*(col-row))+margin, img.shape[0]])
                        c_s = colmin-margin
                        c_e = colmax+margin
                    # if flag == True:
                    #     mask = mask[r_s:r_e,c_s:c_e]
                    else:
                        r_s = 0
                        r_e = h0
                        c_s = 0
                        c_e = w0
                else:
                    mask = np.ones((h0, w0), np.float32)
                    rows, cols = np.nonzero(mask)
                    rowmin = np.min(rows)
                    rowmax = np.max(rows)
                    row = rowmax - rowmin
                    colmin = np.min(cols)
                    colmax = np.max(cols)
                    col = colmax - colmin
                    margin = 0

                    flag = True
                    if row <= col and flag:
                        r_s = rowmin-margin
                        r_e = rowmax+margin
                        c_s = int(0.5 * col) - int(0.5 * row)
                        c_e = int(0.5 * col) + int(0.5 * row)
                    elif row > col and flag:
                        r_s = int(0.5 * row) - int(0.5 * col)
                        r_e = int(0.5 * row) + int(0.5 * col)
                        c_s = colmin-margin
                        c_e = colmax+margin
            #         mask = mask[r_s:r_e,c_s:c_e]

            # if flag:
            #     img  = img[r_s:r_e, c_s:c_e, :]
            #     if i == 0:
            #         N = N[r_s:r_e, c_s:c_e, :]


            h = int(np.floor(np.max([img.shape[0], img.shape[1]]) / 512) * 512)
            if h > max_image_resolution:
                h = max_image_resolution
            if h < 256:
                h = 256

            w = h 
            img = cv2.resize(img, dsize=(w, h), interpolation=cv2.INTER_CUBIC)
            img = np.expand_dims(img, axis=-1)
            if i == 0:
                N = cv2.resize(N, dsize=(w, h), interpolation=cv2.INTER_CUBIC)
                mask = np.float32(cv2.resize(mask, dsize=(w, h), interpolation=cv2.INTER_CUBIC) > 0.5)

            if img.dtype == 'uint8':
                bit_depth = 255.0
            elif img.dtype == 'uint16':
                bit_depth = 65535.0
            else:
                bit_depth = 1.0

            img = np.float32(img) / bit_depth
            img = img*mask[:, :, np.newaxis]
            if np.max(img) > 1 or np.min(img) < 0:
                img = (img - np.min(img)) / (np.max(img) - np.min(img))
            if i == 0:
                I = np.zeros((len(indexset), h, w, 1), np.float32) # [N, h, w, c]
            I[i, :, :, :] = img

        # self.img_tile(I, 3, 3, self.data_workspace)
        I = np.reshape(I, (-1, h * w, 1))

        """Data Normalization"""
        temp = np.mean(I[:, mask.flatten()==1,:], axis=2)
        mean = np.mean(temp, axis=1)
        mx = np.max(temp, axis=1)
        scale = np.random.rand(I.shape[0],)
        temp = (1-scale) * mean + scale * mx
        temp = mx
        I /= (temp.reshape(-1,1,1) + 1.0e-6)

        I = np.transpose(I, (1, 2, 0))
        I = I.reshape(h, w, 1, numberOfImages)
        mask = (mask.reshape(h, w, 1)).astype(np.float32) # 1, h, w

        h = h0
        w = w0
        h = mask.shape[0]
        w = mask.shape[1]

        self.h = h
        self.w = w
        self.I = I
        # self.N = n_true
        # N = N/(np.sqrt(np.sum(N * N, axis=2, keepdims=True)) + 1.0e-6)
        self.N = N

        self.roi = np.array([h0, w0, r_s, r_e, c_s, c_e])
        if self.use_mask == True:
            self.mask = mask
        else:
            self.mask = np.ones(mask.shape, np.float32)
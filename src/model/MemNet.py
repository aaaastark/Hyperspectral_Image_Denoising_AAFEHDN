import numpy as np
import pandas as pd
import random, os, torch, shutil, time, scipy
from datetime import datetime
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
import torch.nn.init as init
import torch.nn.functional as F
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

seed = 123789 # Internation fixed number of seed...
torch.manual_seed(seed)
np.random.seed(seed)  # Numpy module.
random.seed(seed)  # Python random module.
os.environ['PYTHONHASHSEED'] = str(seed)

def SAM_Metric(x_true, x_pred):
    assert x_true.ndim ==3 and x_true.shape == x_pred.shape
    h,w,c=x_pred.shape
    sam_rad = []
    for x in range(x_true.shape[0]):
        for y in range(x_true.shape[1]):
            tmp_pred = x_pred[x, y].ravel()
            tmp_true = x_true[x, y].ravel()
            s = np.sum(np.dot(tmp_pred, tmp_true))
            t = (np.sqrt(np.sum(tmp_pred ** 2))) * (np.sqrt(np.sum(tmp_true ** 2)))
            th = np.arccos(s/t)
            sam_rad.append(th)
    sam_deg = np.mean(sam_rad)
    return sam_deg

def quantitative_assess(data_clean,test_out):
    psnrs=[]
    ssims=[]
    height,width,band =data_clean.shape
    for b in range(band):
        psnr1 = peak_signal_noise_ratio(data_clean[:, :, b], test_out[:, :, b],data_range=1)
        ssim1 = structural_similarity(data_clean[:, :, b], test_out[:, :, b],win_size=11,data_range=1,gaussian_weights=1)
        psnrs.append(psnr1)
        ssims.append(ssim1)
    avg_psnr = np.mean(psnrs)
    avg_ssim = np.mean(ssims)
    Sam=SAM_Metric(data_clean,test_out)
    return [avg_psnr,avg_ssim,Sam]


class BNReLUConv(nn.Sequential):
    def __init__(self, in_channels, channels, k=3, s=1, p=1, inplace=True):
        super(BNReLUConv, self).__init__()
        self.add_module('bn', nn.BatchNorm2d(in_channels))
        self.add_module('relu', nn.ReLU(inplace=inplace))
        self.add_module('conv', nn.Conv2d(in_channels, channels, k, s, p, bias=False))

class ResidualBlock(torch.nn.Module):
    def __init__(self, channels, k=3, s=1, p=1):
        super(ResidualBlock, self).__init__()
        self.relu_conv1 = BNReLUConv(channels, channels, k, s, p)
        self.relu_conv2 = BNReLUConv(channels, channels, k, s, p)
    def forward(self, x):
        residual = x
        out = self.relu_conv1(x)
        out = self.relu_conv2(out)
        out = out + residual
        return out

class MemoryBlock(nn.Module):
    def __init__(self, channels, num_resblock, num_memblock):
        super(MemoryBlock, self).__init__()
        self.recursive_unit = nn.ModuleList(
            [ResidualBlock(channels) for i in range(num_resblock)]
        )
        self.gate_unit = BNReLUConv((num_resblock+num_memblock) * channels, channels, 1, 1, 0)
    def forward(self, x, ys):
        xs = []
        residual = x
        for layer in self.recursive_unit:
            x = layer(x)
            xs.append(x)
        gate_out = self.gate_unit(torch.cat(xs+ys, 1))
        ys.append(gate_out)
        return gate_out

class MemNet(nn.Module):
    def __init__(self, in_channels, channels, num_memblock, num_resblock):
        super(MemNet, self).__init__()
        out_channnels = 1
        self.feature_extractor = BNReLUConv(in_channels, channels)
        self.reconstructor = BNReLUConv(channels, out_channnels)
        self.dense_memory = nn.ModuleList(
            [MemoryBlock(channels, num_resblock, i+1) for i in range(num_memblock)]
        )
        self.freeze_bn = True
        self.freeze_bn_affine = True
    def forward(self, x):
        residual = x
        out = self.feature_extractor(x)
        ys = [out]
        for memory_block in self.dense_memory:
            out = memory_block(out, ys)
        out = self.reconstructor(out)
        return out


def create_folder(StreamLit = None, noise_type = None, name = None):
    if os.path.isdir(f"MemNet_Denoising_{noise_type}_{name}"):
        shutil.rmtree(f"MemNet_Denoising_{noise_type}_{name}")
        StreamLit.write(f"Delete MemNet_Denoising_{noise_type}_{name} Folder")
    if not os.path.exists(f'MemNet_Denoising_{noise_type}_{name}'):
        os.makedirs(f'MemNet_Denoising_{noise_type}_{name}')
        StreamLit.write(f"Create MemNet_Denoising_{noise_type}_{name} Folder")
    else:
        pass

    save_dir = os.path.join(f'MemNet_Model_{noise_type}')
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
        StreamLit.write(f"Delete MemNet_Model_{noise_type} Folder")
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
        StreamLit.write(f"Create MemNet_Model_{noise_type} Folder")
    else:
        pass
    return save_dir, name

def MemNet_SimulationCode(StreamLit = None, data = None, test_data_clean_DC = None, test_data_noise_DC = None, batch_size = None, K_Adjacent = None, epochs = None, noise_type = None, band = None):
    save_dir, name_dir = create_folder(StreamLit = StreamLit, noise_type = noise_type, name = "Simulation")

    argsK = K_Adjacent
    k = int(argsK/2)
    milestone=[180]
    lr = 1e-5 # 0.00005
    n_epoch = epochs

    DLoader = DataLoader(dataset=data, num_workers=0, drop_last=False, batch_size=batch_size, shuffle=True)
    model = MemNet(1, 64, 6, 6)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = MultiStepLR(optimizer, milestones=milestone, gamma=0.2)

    psnrs_matric, ssims_matric, sams_matric, time_train, time_test = list(), list(), list(), list(), list()
    for epoch in range(0, n_epoch):
        current_time = datetime.now()
        formatted_time = current_time.strftime("%Y-%m-%d %I:%M:%S %p")
        with StreamLit.status(f"Epoch : {epoch+1}/{n_epoch} and Current Time: {formatted_time}"):
            model.train()
            lr = optimizer.state_dict()['param_groups'][0]['lr']
            epoch_loss = 0
            start_time = time.time()

            for _, batch_yx in enumerate(DLoader):
                optimizer.zero_grad()
                batch_x, batch_y = batch_yx[:,0,:,:], batch_yx[:,1,:,:]

                iter_band = np.arange(band)
                np.random.shuffle(iter_band)

                for b in iter_band:
                    x = batch_y[:,b, :, :]
                    noise_free = batch_x[:,b, :, :]

                    x = torch.unsqueeze(x, dim=1).type(torch.FloatTensor)
                    noise_free = torch.unsqueeze(noise_free, dim=1).type(torch.FloatTensor)

                    learned_image=model(x)
                    loss = criterion(learned_image, noise_free)
                    epoch_loss += loss.item()
                    loss.backward()
                    optimizer.step()

            scheduler.step()
            batch_number = data.size(0) // batch_size
            final_time = time.time() - start_time
            time_train.append(final_time)

            if(epoch<n_epoch):
                model.eval()
                data_noise = torch.from_numpy(test_data_noise_DC)
                hight,width,cs=data_noise.shape
                data_noise = data_noise.permute(2, 1, 0)
                test_out = torch.zeros(data_noise.shape).type(torch.FloatTensor)

                start_time_test = time.time()
                for channel_i in range(cs):
                    x_data = data_noise[channel_i, :, :]
                    x_data = torch.unsqueeze(x_data, dim=0).type(torch.FloatTensor)
                    x_data = torch.unsqueeze(x_data, dim=0).type(torch.FloatTensor)

                    with torch.no_grad():
                        out = model(x_data)

                    out = out.squeeze()
                    test_out[channel_i,:,:] = out

                end_time_test = time.time() - start_time_test
                time_test.append(end_time_test)

                test_out = test_out.permute(2,1,0)
                denoise_image_out = test_out.cpu().numpy()

                PSNR,SSIM,SAM = quantitative_assess(test_data_clean_DC, denoise_image_out)
                psnrs_matric.append(PSNR)
                ssims_matric.append(SSIM)
                sams_matric.append(SAM)

            total_epoch_loss = epoch_loss / batch_number
            StreamLit.write(f"Epoch : {epoch+1}/{n_epoch} and Loss : ({total_epoch_loss:.5f}) and (PSNR : {PSNR} & SSIM : {SSIM} & SAM : {SAM}) and Train Time : ({final_time:.1f}) Sec and Test Time : ({end_time_test:.1f}) Sec")

            torch.save(model, os.path.join(save_dir, 'model_%03d.pth' % (epoch + 1)))
            checkpoint = {
                            'optimizer': optimizer.state_dict(),
                            "epoch": epoch,
                            'lr_schedule': scheduler.state_dict()
                        }
            torch.save(checkpoint, os.path.join(save_dir, 'checkpoint_%03d.pth' % (epoch + 1)))

    end_time = datetime.now()
    formatted_time = end_time.strftime("%Y-%m-%d %I:%M:%S %p")
    with StreamLit.success(f'Denoise (Mat) and CSV (PSNR-SSIM-SAM-TrainTime-TestTime) files have been saved. End Time : {formatted_time}', icon="✅"):
        scipy.io.savemat(f"./MemNet_Denoising_{noise_type}_{name_dir}/WashingtonDCMallUniversity_Denoise_{noise_type}_Dataset_Image.mat",{'Image': denoise_image_out})
        data = {"PSNR": psnrs_matric, "SSIM": ssims_matric, "SAM": sams_matric, "TrainTime": time_train, "TestTime": time_test}
        data = pd.DataFrame().from_dict(data)
        data.to_csv(f'./MemNet_Denoising_{noise_type}_{name_dir}/MemNet_Denoising_PSNR_SSIM_SAM_BANDS_CSVFILE_{noise_type}.csv', index=False)


def MemNet_RealCode(StreamLit = None, data = None, test_data_clean_DC = None, test_data_noise_DC = None, batch_size = None, K_Adjacent = None, epochs = None, noise_type = None, band = None):
    save_dir, name_dir = create_folder(StreamLit = StreamLit, noise_type = noise_type, name = "Real")
    
    argsK = K_Adjacent
    k = int(argsK/2)
    milestone=[180]
    lr = 1e-5 # 0.00005
    n_epoch = epochs

    DLoader = DataLoader(dataset=data, num_workers=0, drop_last=False, batch_size=batch_size, shuffle=True)
    model = MemNet(1, 64, 6, 6)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = MultiStepLR(optimizer, milestones=milestone, gamma=0.2)

    psnrs_matric, ssims_matric, sams_matric, time_train, time_test = list(), list(), list(), list(), list()
    for epoch in range(0, n_epoch):
        current_time = datetime.now()
        formatted_time = current_time.strftime("%Y-%m-%d %I:%M:%S %p")
        with StreamLit.status(f"Epoch : {epoch+1}/{n_epoch} and Current Time: {formatted_time}"):
            model.train()
            lr = optimizer.state_dict()['param_groups'][0]['lr']
            epoch_loss = 0
            start_time = time.time()

            for _, batch_yx in enumerate(DLoader):
                optimizer.zero_grad()
                batch_x, batch_y = batch_yx[:,0,:,:], batch_yx[:,1,:,:]

                iter_band = np.arange(band)
                np.random.shuffle(iter_band)

                for b in iter_band:
                    x = batch_y[:,b, :, :]
                    noise_free = batch_x[:,b, :, :]

                    x = torch.unsqueeze(x, dim=1).type(torch.FloatTensor)
                    noise_free = torch.unsqueeze(noise_free, dim=1).type(torch.FloatTensor)

                    learned_image=model(x)
                    loss = criterion(learned_image, noise_free)
                    epoch_loss += loss.item()
                    loss.backward()
                    optimizer.step()

            scheduler.step()
            batch_number = data.size(0) // batch_size
            final_time = time.time() - start_time
            time_train.append(final_time)

            if(epoch<n_epoch):
                model.eval()
                data_noise = torch.from_numpy(test_data_noise_DC)
                hight,width,cs=data_noise.shape
                data_noise = data_noise.permute(2, 1, 0)
                test_out = torch.zeros(data_noise.shape).type(torch.FloatTensor)

                start_time_test = time.time()
                for channel_i in range(cs):
                    x_data = data_noise[channel_i, :, :]
                    x_data = torch.unsqueeze(x_data, dim=0).type(torch.FloatTensor)
                    x_data = torch.unsqueeze(x_data, dim=0).type(torch.FloatTensor)

                    with torch.no_grad():
                        out = model(x_data)

                    out = out.squeeze()
                    test_out[channel_i,:,:] = out

                end_time_test = time.time() - start_time_test
                time_test.append(end_time_test)

                test_out = test_out.permute(2,1,0)
                denoise_image_out = test_out.cpu().numpy()

                PSNR,SSIM,SAM = quantitative_assess(test_data_clean_DC, denoise_image_out)
                psnrs_matric.append(PSNR)
                ssims_matric.append(SSIM)
                sams_matric.append(SAM)

            total_epoch_loss = epoch_loss / batch_number
            StreamLit.write(f"Epoch : {epoch+1}/{n_epoch} and Loss : ({total_epoch_loss:.5f}) and (PSNR : {PSNR} & SSIM : {SSIM} & SAM : {SAM}) and Train Time : ({final_time:.1f}) Sec and Test Time : ({end_time_test:.1f}) Sec")

            torch.save(model, os.path.join(save_dir, 'model_%03d.pth' % (epoch + 1)))
            checkpoint = {
                            'optimizer': optimizer.state_dict(),
                            "epoch": epoch,
                            'lr_schedule': scheduler.state_dict()
                        }
            torch.save(checkpoint, os.path.join(save_dir, 'checkpoint_%03d.pth' % (epoch + 1)))

    end_time = datetime.now()
    formatted_time = end_time.strftime("%Y-%m-%d %I:%M:%S %p")
    with StreamLit.success(f'Denoise (Mat) and CSV (PSNR-SSIM-SAM-TrainTime-TestTime) files have been saved. End Time : {formatted_time}', icon="✅"):
        scipy.io.savemat(f"./MemNet_Denoising_{noise_type}_{name_dir}/Indian_Pines_Denoise_{noise_type}_Dataset_Image.mat",{'Image': denoise_image_out})
        data = {"PSNR": psnrs_matric, "SSIM": ssims_matric, "SAM": sams_matric, "TrainTime": time_train, "TestTime": time_test}
        data = pd.DataFrame().from_dict(data)
        data.to_csv(f'./MemNet_Denoising_{noise_type}_{name_dir}/MemNet_Denoising_PSNR_SSIM_SAM_BANDS_CSVFILE_{noise_type}.csv', index=False)
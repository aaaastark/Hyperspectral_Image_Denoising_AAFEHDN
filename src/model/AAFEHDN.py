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

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid() # real number is reduced to a value between 0 and 1.

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ):
        super(ChannelAttention, self).__init__()
        # input signal composed of several input planes. The h and w dimensions of the output tensor are determined by the parameter output_size.
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU() # A rectified linear unit (ReLU). Manage non-linearity to a deep learning model and solves the vanishing gradients issue.
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class Geometrical_Characteristics(nn.Module):
  def __init__(self, input_Channel=None, feature_Channel=None, K=None):
    super(Geometrical_Characteristics, self).__init__()

    self.Spatial_Feature_3 = nn.Sequential(
        nn.Conv2d(input_Channel, feature_Channel, 3, padding=1),
    )
    self.Spatial_Feature_5 = nn.Sequential(
        nn.Conv2d(input_Channel, feature_Channel, 5, padding=2),
    )
    self.Spatial_Feature_7 = nn.Sequential(
        nn.Conv2d(input_Channel, feature_Channel, 7, padding=3),
    )

    self.Spectral_Feature_3 = nn.Sequential(
        nn.Conv3d(1, feature_Channel, (K, 1, 1), 1, (0, 0, 0)),
        nn.Conv3d(feature_Channel, feature_Channel, (1, 3, 3), 1, (0, 1, 1))
    )
    self.Spectral_Feature_5 = nn.Sequential(
        nn.Conv3d(1, feature_Channel, (K, 1, 1), 1, (0, 0, 0)),
        nn.Conv3d(feature_Channel, feature_Channel, (1, 5, 5), 1, (0, 2, 2))
    )
    self.Spectral_Feature_7 = nn.Sequential(
        nn.Conv3d(1, feature_Channel, (K, 1, 1), 1, (0, 0, 0)),
        nn.Conv3d(feature_Channel, feature_Channel, (1, 7, 7), 1, (0, 3, 3))
    )

  def forward(self, Spatial, Spectral):
    Spatial_3 = self.Spatial_Feature_3(Spatial)
    Spatial_5 = self.Spatial_Feature_5(Spatial)
    Spatial_7 = self.Spatial_Feature_7(Spatial)

    Spectral_3 = self.Spectral_Feature_3(Spectral)
    Spectral_5 = self.Spectral_Feature_5(Spectral)
    Spectral_7 = self.Spectral_Feature_7(Spectral)

    spatial = F.leaky_relu(torch.cat((Spatial_3, Spatial_5, Spatial_7), dim=1))
    spectral = F.leaky_relu(torch.cat((Spectral_3, Spectral_5, Spectral_7), dim=1)).squeeze(2)
    spatial_spectral = torch.cat((spatial, spectral), dim=1)

    return spatial_spectral
  

# Attentive Skip Connection (High and Low Frequency Features)
class ASC(nn.Module):
  def __init__(self, channel):
    super().__init__()
    self.weight = nn.Sequential(
        nn.Conv2d(channel * 2, channel, 1),
        nn.LeakyReLU(), # Similar to Relu. Small slope for negative values instead of a flat slope. The slope coefficient is determined before training.
        nn.Conv2d(channel, channel, 3, 1, 1),
        nn.Sigmoid()
    )

  def forward(self, x, y):
    w = self.weight(torch.cat([x, y], dim=1))
    out = (1 - w) * x + w * y
    return out

# Progressive Spectral Channel Attention (PSCA)
class PSCA(nn.Module):
  def __init__(self, channel, channel_half):
    super().__init__()
    self.w_3 = nn.Conv2d(channel, channel, 1, bias=False)
    self.w_1 = nn.Conv2d(channel, channel_half, 1, bias=False)
    self.w_2 = nn.Conv2d(channel_half, channel, 1, bias=False)
    nn.init.zeros_(self.w_3.weight)

  def forward(self, x):
    x = self.w_3(x) * x + x
    x = self.w_1(x)
    x = F.gelu(x) # Gaussian cumulative distribution function. The GELU nonlinearity weights inputs by their percentile, rather than gates inputs by their sign as in ReLUs.
    x = self.w_2(x)
    return x
  
class AAFEHDN_block(nn.Module):
    def __init__(self,block_num=None, channel=None):
        super(AAFEHDN_block,self).__init__()
        self.group_list_asc_model = []
        self.group_list_psca_model = []
        channel_half = int(channel//2)
        self.block_number = block_num

        for i in range(0, block_num-1):
          asc_model = ASC(channel)
          self.add_module(name='asc_model_%d' % i, module=asc_model)
          self.group_list_asc_model.append(asc_model)

          psca_model = PSCA(channel, channel_half)
          self.add_module(name='psca_model_%d' % i, module=psca_model)
          self.group_list_psca_model.append(psca_model)

        self.channel_attention = ChannelAttention(channel)
        self.spatial_attention = SpatialAttention(7)

    def forward(self, input):
      output_feature = [input]
      updated_feature = input
      for index in range(0, self.block_number-1):
        asc_model_group1 = self.group_list_asc_model[index](updated_feature, updated_feature)
        channel_attention_asc = self.channel_attention(asc_model_group1) * asc_model_group1
        spatial_attention_asc = self.spatial_attention(channel_attention_asc) * channel_attention_asc + input

        psca_model_group2 = self.group_list_psca_model[index](spatial_attention_asc)
        channel_attention_psca = self.channel_attention(psca_model_group2) * psca_model_group2
        spatial_attention_psca = self.spatial_attention(channel_attention_psca) * channel_attention_psca + input

        output_feature.append(spatial_attention_psca)
        updated_feature = spatial_attention_psca + input

      concat = torch.cat(output_feature, dim=1)
      return concat
    
class AAFEHDN_Simulation(nn.Module):
    def __init__(self, input_Channel=1, block_num=3, K=24, feature_Channel=20, layer_output=80):
        super(AAFEHDN_Simulation, self).__init__()
        self.geometricalCharacter = Geometrical_Characteristics(input_Channel, feature_Channel, K)
        self.ca_in = ChannelAttention(feature_Channel * 6)
        self.sa_in = SpatialAttention(7)
        self.concat_in = nn.Conv2d(feature_Channel * 6, layer_output, 3, 1, padding=1)
        self.AAFEHDN_block = AAFEHDN_block(block_num=block_num, channel=layer_output)
        self.FR = nn.Conv2d(layer_output*block_num, input_Channel, 3, 1, padding=1)

    def forward(self, Spatial, Spectral):
        spatial_spectral = self.geometricalCharacter(Spatial, Spectral)
        ca_in = self.ca_in(spatial_spectral) * spatial_spectral
        sa_in = self.sa_in(ca_in) * ca_in
        output_Geometrical =  self.concat_in(sa_in)
        AAFEHDN_output = self.AAFEHDN_block(output_Geometrical)
        Residual = self.FR(AAFEHDN_output)
        out = Spatial - Residual
        return out

    def _initialize_weights(self):
      for m in self.modules():
        if isinstance(m, nn.Conv2d):
          init.orthogonal_(m.weight)
          if m.bias is not None:
              init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv3d):
          init.orthogonal_(m.weight)
          print('init weight')
          if m.bias is not None:
              init.constant_(m.bias, 0)


def create_folder(StreamLit = None, noise_type = None, name = None):
    if os.path.isdir(f"AAFEHDN_Denoising_{noise_type}_{name}"):
        shutil.rmtree(f"AAFEHDN_Denoising_{noise_type}_{name}")
        StreamLit.write(f"Delete AAFEHDN_Denoising_{noise_type}_{name} Folder")
    if not os.path.exists(f'AAFEHDN_Denoising_{noise_type}_{name}'):
        os.makedirs(f'AAFEHDN_Denoising_{noise_type}_{name}')
        StreamLit.write(f"Create AAFEHDN_Denoising_{noise_type}_{name} Folder")
    else:
        pass

    save_dir = os.path.join(f'AAFEHDN_Model_{noise_type}')
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
        StreamLit.write(f"Delete AAFEHDN_Model_{noise_type} Folder")
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
        StreamLit.write(f"Create AAFEHDN_Model_{noise_type} Folder")
    else:
        pass
    return save_dir, name

def AAFEHDN_SimulationCode(StreamLit = None, data = None, test_data_clean_DC = None, test_data_noise_DC = None, batch_size = None, K_Adjacent = None, epochs = None, noise_type = None, band = None):
    save_dir, name_dir = create_folder(StreamLit = StreamLit, noise_type = noise_type, name = "Simulation")

    argsK = K_Adjacent
    k = int(argsK/2)
    milestone=[180]
    lr = 1e-5 # 0.00005
    n_epoch = epochs

    DLoader = DataLoader(dataset=data, num_workers=0, drop_last=False, batch_size=batch_size, shuffle=True)
    model = AAFEHDN_Simulation(K = K_Adjacent)
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

                    if b < k: # first
                        y = batch_y[:,0:argsK, :, :]
                    elif b < band - k: # last
                        y = torch.cat((batch_y[:,b - k:b, :, :],batch_y[:,b + 1:b + k + 1, :, :]),1)
                    else:
                        y = batch_y[:,band - argsK:band, :, :]

                    y = torch.unsqueeze(y, dim=1).type(torch.FloatTensor)

                    learned_image=model(x, y)
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

                    if channel_i < k:
                        y_data = data_noise[0:argsK, :, :]
                    elif channel_i < cs - k:
                        y_data = torch.cat((data_noise[channel_i - k:channel_i, :, :],data_noise[channel_i + 1:channel_i + k + 1, :, :]))
                    else:
                        y_data = data_noise[cs - argsK:cs, :, :]

                    y_data = torch.unsqueeze(y_data, dim=0).type(torch.FloatTensor)
                    y_data = torch.unsqueeze(y_data, dim=0).type(torch.FloatTensor)

                    with torch.no_grad():
                        out = model(x_data, y_data)

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
            # print(f"Epoch : {epoch+1}/{n_epoch} and Loss : ({total_epoch_loss:.5f}) and (PSNR : {PSNR} & SSIM : {SSIM} & SAM : {SAM}) and Train Time : ({final_time:.1f}) Sec and Test Time : ({end_time_test:.1f}) Sec")
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
        scipy.io.savemat(f"./AAFEHDN_Denoising_{noise_type}_{name_dir}/WashingtonDCMallUniversity_Denoise_{noise_type}_Dataset_Image.mat",{'Image': denoise_image_out})
        data = {"PSNR": psnrs_matric, "SSIM": ssims_matric, "SAM": sams_matric, "TrainTime": time_train, "TestTime": time_test}
        data = pd.DataFrame().from_dict(data)
        data.to_csv(f'./AAFEHDN_Denoising_{noise_type}_{name_dir}/AAFEHDN_Denoising_PSNR_SSIM_SAM_BANDS_CSVFILE_{noise_type}.csv', index=False)


def AAFEHDN_RealCode(StreamLit = None, data = None, test_data_clean_DC = None, test_data_noise_DC = None, batch_size = None, K_Adjacent = None, epochs = None, noise_type = None, band = None):
    save_dir, name_dir = create_folder(noise_type = noise_type, name = "Real")
    
    argsK = K_Adjacent
    k = int(argsK/2)
    milestone=[180]
    lr = 1e-5 # 0.00005
    n_epoch = epochs

    DLoader = DataLoader(dataset=data, num_workers=0, drop_last=False, batch_size=batch_size, shuffle=True)
    model = AAFEHDN_Simulation(K = K_Adjacent)
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

                    if b < k: # first
                        y = batch_y[:,0:argsK, :, :]
                    elif b < band - k: # last
                        y = torch.cat((batch_y[:,b - k:b, :, :],batch_y[:,b + 1:b + k + 1, :, :]),1)
                    else:
                        y = batch_y[:,band - argsK:band, :, :]

                    y = torch.unsqueeze(y, dim=1).type(torch.FloatTensor)

                    learned_image=model(x, y)
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

                    if channel_i < k:
                        y_data = data_noise[0:argsK, :, :]
                    elif channel_i < cs - k:
                        y_data = torch.cat((data_noise[channel_i - k:channel_i, :, :],data_noise[channel_i + 1:channel_i + k + 1, :, :]))
                    else:
                        y_data = data_noise[cs - argsK:cs, :, :]

                    y_data = torch.unsqueeze(y_data, dim=0).type(torch.FloatTensor)
                    y_data = torch.unsqueeze(y_data, dim=0).type(torch.FloatTensor)

                    with torch.no_grad():
                        out = model(x_data, y_data)

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
            print(f"Epoch : {epoch+1}/{n_epoch} and Loss : ({total_epoch_loss:.5f}) and (PSNR : {PSNR} & SSIM : {SSIM} & SAM : {SAM}) and Train Time : ({final_time:.1f}) Sec and Test Time : ({end_time_test:.1f}) Sec")
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
        scipy.io.savemat(f"./AAFEHDN_Denoising_{noise_type}_{name_dir}/Indian_Pines_Denoise_{noise_type}_Dataset_Image.mat",{'Image': denoise_image_out})
        data = {"PSNR": psnrs_matric, "SSIM": ssims_matric, "SAM": sams_matric, "TrainTime": time_train, "TestTime": time_test}
        data = pd.DataFrame().from_dict(data)
        data.to_csv(f'./AAFEHDN_Denoising_{noise_type}_{name_dir}/AAFEHDN_Denoising_PSNR_SSIM_SAM_BANDS_CSVFILE_{noise_type}.csv', index=False)
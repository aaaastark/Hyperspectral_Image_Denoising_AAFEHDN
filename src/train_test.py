import numpy as np
import matplotlib.pyplot as plt
import scipy
import random

class TrainTest:
    def __init__(self, st = None) -> None:
        self.StreamLit = st
        self.aug_times = 1
        self.count = 0
        np.random.seed(123789)
        random.seed(123789) 
    
    def inputs_features(self):
        noise_type = self.StreamLit.selectbox("Choose the Single Noise Type for Simulation Process.", ("Sigma", "Random", "Gaussain"), index=None, placeholder="Select Noise method...",)
        scale = self.StreamLit.number_input("Scale", value=1, placeholder="Type a number. By default 1.", format = "%i")
        stride = self.StreamLit.number_input("Stride", value=20, placeholder="Type a number. By default 20.", format = "%i")
        pathsize = self.StreamLit.number_input("Patch Size", value=20, placeholder="Type a number. By default 20.", format = "%i")
        batchsize = self.StreamLit.number_input("Batch Size", value=25, placeholder="Type a number. By default 25.", format = "%i")
        return noise_type, scale, pathsize, stride, batchsize

    def calculation_gaussian(self, value_i_b,B=200,N=30):
        Upper_left = np.negative(np.power(np.divide(np.subtract(value_i_b,B),2),2))
        Upper_right = np.multiply(2, np.power(N, 2))
        value = np.exp(np.divide(Upper_left,Upper_right))
        return value

    def Gaussian_Noise(self, Channel = None):
        B, N = 200, 30
        Upper = self.calculation_gaussian(Channel, B, N)
        Lower = np.sum(self.calculation_gaussian(np.arange(1,B), B, N))
        Final = np.multiply(B,np.sqrt(np.divide(Upper,Lower)))
        return Final

    def noise_add_by_bands(self, data = None, noise_type = None):
        h, w, c = data.shape
        noise = []
        for channel in range(c):
            # add alpha noise
            if (noise_type == "Sigma"):
                sigma = np.random.randint(1, 50)
                noise_sigma_band = np.random.normal(scale=(sigma / 255), size=[h, w])
                noise.append(noise_sigma_band)
            # add random noise
            if (noise_type == "Random"):
                random_value = np.random.randint(1, 25)
                random_band = ((random_value + np.random.randn(h, w))/255)
                noise.append(random_band)
            # add gaussian noise
            if ((noise_type == "Gaussain")):
                value = self.Gaussian_Noise(channel)
                gaussian_band = np.random.normal(scale=(value / 255), size=[h, w])
                noise.append(gaussian_band)

        noise = np.array(noise)
        noise = noise.transpose(1, 2, 0)
        noise_image = np.clip(data + noise, 0, 1).astype(np.float32)
        return noise_image
    
    def data_rotation(self, img, rot_time, filp_mode):
        if filp_mode == -1:
            return np.rot90(img, k=rot_time)
        else:
            return np.flip(np.rot90(img, k=rot_time), axis=filp_mode)
        
    def train_augmentation(self, data_train, scale, stride, patch_size, noise_type, batch_size):
        clean_data = []
        noise_data = []
        scales = [scale]
        for s in scales:
            data_scaled = scipy.ndimage.zoom(data_train, (s, s, 1)).astype(np.float32)
            data_scaled[data_scaled < 0] = 0
            data_scaled[data_scaled > 1] = 1

            h_scaled, w_scaled, band_scaled = data_scaled.shape
            for i in range(0, h_scaled - patch_size + 1, stride):
                for j in range(0, w_scaled - patch_size + 1, stride):
                    for k in range(0, self.aug_times):
                        self.count += 1
                        x = data_scaled[i:i + patch_size, j:j + patch_size, :]

                        rot_time = np.random.randint(0, 4) # Generate: 0, 1, 2, 3
                        filp_mode = np.random.randint(-1, 2) # Generate: -1, 0, 1
                        x_aug = self.data_rotation(x, rot_time, filp_mode) # Data Augmented

                        y_aug = self.noise_add_by_bands(data = x_aug, noise_type = noise_type) # Add the Simulated Noise
                        x_np = np.array(x_aug, dtype='float32')
                        y_np = np.array(y_aug, dtype='float32')
                        clean_data.append(x_np)
                        noise_data.append(y_np)
            
            original_image = np.array(clean_data)
            noise_image = np.array(noise_data)
            original_image_batchsize = original_image[:batch_size]
            noise_image_batchsize = noise_image[:batch_size]
            return original_image_batchsize, noise_image_batchsize

    def train_procedure(self, dc_image, noise_type, scale, pathsize, stride, batchsize):
        data_train1 = dc_image[0:600, :, :]
        data_train2 = dc_image[800:1280, :, :]
        data_train_DC = np.concatenate((data_train1, data_train2), axis=0)
        original_image_batchsize, noise_image_batchsize = self.train_augmentation(data_train = data_train_DC, scale = scale, stride = stride, patch_size = pathsize, 
                                                                                  noise_type = noise_type, batch_size = batchsize)
        return original_image_batchsize, noise_image_batchsize

    def train_mat_file_save(self, original_image = None, noise_image = None, file_name = None):
        scipy.io.savemat(f"./mat/DC_Mall_TrainDataset_Original_{str(file_name)}_Noise_Augmentation_Data.mat",{'original': original_image, 'noise': noise_image})
        self.StreamLit.success(f"Image saved to ./mat/DC_Mall_TrainDataset_Original_{str(file_name)}_Noise_Augmentation_Data.mat")

    def test_procedure(self, dc_image, noise_type):
        original_image = dc_image[600:800, 50:250, :].astype(np.float32)
        noise_image = self.noise_add_by_bands(data = original_image, noise_type = noise_type)
        return original_image, noise_image
    
    def test_mat_file_save(self, original_image = None, noise_image = None, file_name = None):
        scipy.io.savemat(f"./mat/DC_Mall_TestDataset_Original_{str(file_name)}_Noise_Data.mat",{'original': original_image, 'noise': noise_image})
        self.StreamLit.success(f"Image saved to ./mat/DC_Mall_TestDataset_Original_{str(file_name)}_Noise_Data.mat")

    def real_mat_file_save(self, original_image = None, file_name = None):
        scipy.io.savemat(f"./mat/Indian_Pines_Dataset_{str(file_name)}_Original_Data.mat",{'original': original_image})
        self.StreamLit.success(f"Image saved to ./mat/Indian_Pines_Dataset_{str(file_name)}_Original_Data.mat")


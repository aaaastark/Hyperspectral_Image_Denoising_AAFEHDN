import tifffile
import numpy as np
from PIL import Image
import random
import scipy
import os, shutil

class DataProcess:
    def __init__(self, st = None, ) -> None:
        self.StreamLit = st
        np.random.seed(123789)
        random.seed(123789) 

    def trim_image_in_linear_stretch(self, image, q_sequence_of_percentiles=2, maxout=1, min_out=0):
        def image_ojbect_process(image_ojbect, maxout=maxout, minout=min_out):
            # Truncate the HSI Image. # 2% of pixels to be truncated from both ends
            trim_down = np.percentile(a=image_ojbect, q=q_sequence_of_percentiles)
            trim_up = np.percentile(a=image_ojbect, q=100 - q_sequence_of_percentiles)
            # q = between 0 and 100 inclusive. If q is a single percentile and axis=None,
            # then the result is a scalar. The return output is specified in array.
            # Rescale the pixel values. Just like Numpy Clip image function. In ranage of b/w 0 and 1.
            image_ojbect_new = (image_ojbect - trim_down) / ((trim_up - trim_down) / (maxout - minout))
            # Formula: ((Old Image - Low Percentile) / ((Max Percentile - Low Percentile) / (Max Edge - Min Edge))
            image_ojbect_new[image_ojbect_new < minout] = minout # Replace 0.
            image_ojbect_new[image_ojbect_new > maxout] = maxout # Replace 1a.
            return np.float32(image_ojbect_new)
        image = np.float32(image)
        height, width, band = image.shape
        new_image = np.zeros((height, width, band))
        for b in range(band):
            new_image[:, :, b] = image_ojbect_process(image[:, :, b])
        return new_image

    def load_data(self, dc_mall_dataset = None):
        data_all = tifffile.imread(dc_mall_dataset)
        data_all = data_all.transpose(1, 2, 0)
        data_all = data_all.astype(np.float32)
        data_all = self.trim_image_in_linear_stretch(data_all) # used to convert colorwise image... not to be black image.
        return data_all
    
    def load_real_data(self, indian_pines_dataset = None):
        data_all = scipy.io.loadmat(indian_pines_dataset)['indian_pines']
        data_all = self.trim_image_in_linear_stretch(data_all) # used to convert colorwise image... not to be black image.
        return data_all
    
    def image_correct_shape(self, image_file):
        # Ensure the correct data type and shape
        if image_file.dtype != np.uint8:
            image_file = (image_file * 255).astype(np.uint8)

        if len(image_file.shape) == 2:
            # Convert grayscale to RGB
            image_file = np.stack((image_file,) * 3, axis=-1)
        elif image_file.shape[2] == 1:
            # Expand single-channel image_files to RGB
            image_file = np.concatenate((image_file,) * 3, axis=-1)
        return image_file
    
    def dc_mall_plot(self, image = None, name_file = None, save = False):
        self.StreamLit.image(image, caption=f"DC Mall Hyperspectral Image {name_file}")
        if save:
            filename = f"./img/WashingtonDCMall{name_file}.png"
            image_array = self.image_correct_shape(image_file = image)
            Image.fromarray(image_array).save(filename)
            self.StreamLit.success(f"Image saved to {filename}")
    
    def indian_pines_plot(self, image = None, name_file = None, save = False):
        self.StreamLit.image(image, caption=f"Indian Pines Hyperspectral Image {name_file}")
        if save:
            filename = f"./img/IndianPines{name_file}.png"
            image_array = self.image_correct_shape(image_file = image)
            Image.fromarray(image_array).save(filename)
            self.StreamLit.success(f"Image saved to {filename}")

    def img_mat_folder_make(self):
        # if os.path.isdir(f"img"):
        #     shutil.rmtree(f"img")
        #     self.StreamLit.write(f"Delete Image (img) Folder")
        if not os.path.exists(f'img'):
            with self.StreamLit.expander("See the Storage Folder (img) of Hyperspectral Image"):
                os.makedirs(f'img')
                self.StreamLit.write(f"Create Image (img) Folder")
        else:
            pass

        # if os.path.isdir(f"mat"):
        #     shutil.rmtree(f"mat")
        #     self.StreamLit.write(f"Delete Mat (mat) Folder")
        if not os.path.exists(f'mat'):
            with self.StreamLit.expander("See the Storage Folder (mat) of Hyperspectral Image"):
                os.makedirs(f'mat')
                self.StreamLit.write(f"Create Mat (mat) Folder")
        else:
            pass

from model.AAFEHDN import AAFEHDN_SimulationCode, AAFEHDN_RealCode
from model.DeNet import DeNet_SimulationCode, DeNet_RealCode
from model.MemNet import MemNet_SimulationCode, MemNet_RealCode

import numpy as np
import random, os, torch, scipy

seed = 123789 # Internation fixed number of seed...
torch.manual_seed(seed)
np.random.seed(seed)  # Numpy module.
random.seed(seed)  # Python random module.
os.environ['PYTHONHASHSEED'] = str(seed)

def train_test_data_process(TestData = None, TrainData = None, experiment_type = None):
    # TEST DATA
    test_data_DC = scipy.io.loadmat(TestData)
    test_data_clean_DC = test_data_DC['original']
    if experiment_type == "Simulation":
        test_data_noise_DC = test_data_DC['noise']
    elif experiment_type == "Real":
        test_data_noise_DC = test_data_DC['original']
        
    # TRAIN DATA
    data = scipy.io.loadmat(TrainData)
    data_clean = data['original']
    data_noise = data['noise']

    number, heigth, width, band = data_clean.shape
    data = np.zeros((2, number, heigth, width, band))
    for n in range(number):
        data[0, n, :, :, :] = data_clean[n, :, :, :]
        data[1, n, :, :, :] = data_noise[n, :, :, :]
    data = torch.from_numpy(data.transpose((1, 0, 4, 2, 3)))
    return data, test_data_clean_DC, test_data_noise_DC, band


def Model_Procedure(StreamLit = None, trainfile = None, testfile = None, model_type = None, noise_type = None, experiment_type = None, batch_size = None, K_Adjacent = None, epochs = None):
    data, test_data_clean_DC, test_data_noise_DC, band = train_test_data_process(TestData = testfile, TrainData = trainfile, experiment_type = experiment_type)

    if (model_type == "AAFEHDN") and (experiment_type == "Simulation"):
        with StreamLit.spinner('Start the Simulation Process of AAFEHDN for HSI Denoising.'): 
            AAFEHDN_SimulationCode(StreamLit = StreamLit, data = data, test_data_clean_DC = test_data_clean_DC, test_data_noise_DC = test_data_noise_DC, 
                                batch_size = batch_size, K_Adjacent = K_Adjacent, epochs = epochs, noise_type = noise_type, band = band)
    elif (model_type == "AAFEHDN") and (noise_type == "Sigma") and (experiment_type == "Real"):
        with StreamLit.spinner('Start the Real Process of AAFEHDN for HSI Denoising.'): 
            AAFEHDN_RealCode(StreamLit = StreamLit, data = data, test_data_clean_DC = test_data_clean_DC, test_data_noise_DC = test_data_noise_DC, 
                                batch_size = batch_size, K_Adjacent = K_Adjacent, epochs = epochs, noise_type = noise_type, band = band)
            
    if (model_type == "DeNet") and (experiment_type == "Simulation"):
        with StreamLit.spinner('Start the Simulation Process of DeNet for HSI Denoising.'): 
            DeNet_SimulationCode(StreamLit = StreamLit, data = data, test_data_clean_DC = test_data_clean_DC, test_data_noise_DC = test_data_noise_DC, 
                                batch_size = batch_size, K_Adjacent = K_Adjacent, epochs = epochs, noise_type = noise_type, band = band)
    elif (model_type == "DeNet") and (noise_type == "Sigma") and (experiment_type == "Real"):
        with StreamLit.spinner('Start the Real Process of DeNet for HSI Denoising.'): 
            DeNet_RealCode(StreamLit = StreamLit, data = data, test_data_clean_DC = test_data_clean_DC, test_data_noise_DC = test_data_noise_DC, 
                                batch_size = batch_size, K_Adjacent = K_Adjacent, epochs = epochs, noise_type = noise_type, band = band)
    
    if (model_type == "MemNet") and (experiment_type == "Simulation"):
        with StreamLit.spinner('Start the Simulation Process of MemNet for HSI Denoising.'): 
            MemNet_SimulationCode(StreamLit = StreamLit, data = data, test_data_clean_DC = test_data_clean_DC, test_data_noise_DC = test_data_noise_DC, 
                                batch_size = batch_size, K_Adjacent = K_Adjacent, epochs = epochs, noise_type = noise_type, band = band)
    elif (model_type == "MemNet") and (noise_type == "Sigma") and (experiment_type == "Real"):
        with StreamLit.spinner('Start the Real Process of MemNet for HSI Denoising.'): 
            MemNet_RealCode(StreamLit = StreamLit, data = data, test_data_clean_DC = test_data_clean_DC, test_data_noise_DC = test_data_noise_DC, 
                                batch_size = batch_size, K_Adjacent = K_Adjacent, epochs = epochs, noise_type = noise_type, band = band)
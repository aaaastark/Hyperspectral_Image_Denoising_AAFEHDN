import numpy as np
import random

from models import Model_Procedure

class ModelConfig:
    def __init__(self, st = None, ) -> None:
        self.StreamLit = st
        np.random.seed(123789)
        random.seed(123789) 
    
    def input_types_config(self):
        model_type = self.StreamLit.selectbox("Choose the Single Model Type.", ("AAFEHDN", "MemNet", "DeNet"), index=None, placeholder="Select Model method...",)
        noise_type = self.StreamLit.selectbox("Choose the Single Noise Type.", ("Sigma", "Random", "Gaussain"), index=None, placeholder="Select Noise method...",)
        experiment_type = self.StreamLit.selectbox("Choose the Single Experiment Type.", ("Simulation", "Real"), index=None, placeholder="Select Experiment method...",)
        batch_size = self.StreamLit.number_input("Batch Size", value=4, placeholder="Type a number. By default 4.", format = "%i")
        K_Adjacent = self.StreamLit.number_input("K Adjacent", value=4, placeholder="Type a number. By default 4.", format = "%i")
        epochs = self.StreamLit.number_input("Epoch", value=1, placeholder="Type a number between 1 and 100. By default 1.", format = "%i")
        return model_type, noise_type, experiment_type, batch_size, K_Adjacent, epochs
    
    def input_data_config(self, model_type = None, noise_type = None, experiment_type = None):
        uploaded_file_train = self.StreamLit.file_uploader(f"Choose the {experiment_type} file of Hyperspectral Image for {model_type} Training of {noise_type} method.", type=["mat"])
        uploaded_file_test = self.StreamLit.file_uploader(f"Choose the {experiment_type} file of Hyperspectral Image for {model_type} Testing of {noise_type} method.", type=["mat"])
        return uploaded_file_train, uploaded_file_test
    
    def setup_config(self):
        with self.StreamLit.container(border=True):
            model_type, noise_type, experiment_type, batch_size, K_Adjacent, epochs = self.input_types_config()
            if model_type is not None and noise_type is not None and experiment_type is not None and batch_size is not None and K_Adjacent is not None and epochs is not None:
                uploaded_file_train, uploaded_file_test = self.input_data_config(model_type = model_type, noise_type = noise_type, experiment_type = experiment_type)
                if uploaded_file_train is not None and uploaded_file_test is not None:
                    self.StreamLit.success('Train and Test files of Hyperspectral Image are Upload', icon="✅")
                    Model_Procedure(StreamLit = self.StreamLit, trainfile = uploaded_file_train, testfile = uploaded_file_test, model_type = model_type, 
                                    noise_type = noise_type, experiment_type = experiment_type, batch_size = batch_size, K_Adjacent = K_Adjacent, epochs = epochs)
                else:
                    self.StreamLit.info("Please upload the Train and Test files of Hyperspectral Image.")
            else:
                self.StreamLit.warning("Please enter type for all fields.")

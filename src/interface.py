import numpy as np
import pandas as pd
import scipy, random
from prettytable import PrettyTable
from matplotlib import pyplot as plt


class InterfaceConfig:
    def __init__(self, st = None, ) -> None:
        self.StreamLit = st
        np.random.seed(123789)
        random.seed(123789) 
    
    def input_types_config(self):
        model_type = self.StreamLit.selectbox("Choose the Single Model Type.", ("AAFEHDN", "MemNet", "DeNet"), index=None, placeholder="Select Model method...", key="model_type")
        noise_type = self.StreamLit.selectbox("Choose the Single Noise Type.", ("Sigma", "Random", "Gaussain"), index=None, placeholder="Select Noise method...", key="noise_type")
        experiment_type = self.StreamLit.selectbox("Choose the Single Experiment Type.", ("Simulation", "Real"), index=None, placeholder="Select Experiment method...", key="experiment_type")
        return model_type, noise_type, experiment_type
    
    def input_data_config(self, model_type = None, noise_type = None, experiment_type = None):
        uploaded_file_mat = self.StreamLit.file_uploader(f"Choose the Denoise {experiment_type} data file of Hyperspectral Image for {model_type} of {noise_type} method.", type=["mat"])
        uploaded_file_csv = self.StreamLit.file_uploader(f"Choose the Result {experiment_type} data file of Hyperspectral Image for {model_type} of {noise_type} method.", type=["csv"])
        return uploaded_file_mat, uploaded_file_csv
    
    def csv_table_file_headel(self, uploaded_file_csv = None, model_type = None, noise_type = None, experiment_type = None):
        with self.StreamLit.status(f"Table of {model_type}_{noise_type}_{experiment_type}."):
            data_file = pd.read_csv(uploaded_file_csv)
            psnrs_matric = data_file['PSNR']
            ssims_matric = data_file['SSIM']
            sams_matric = data_file['SAM']
            train_time = data_file['TrainTime'] 
            time_test = data_file['TestTime']

            psnr_max, psnr_std = np.amax(psnrs_matric), np.std(psnrs_matric) # ddof=1
            ssim_max, ssim_std = np.amax(ssims_matric), np.std(ssims_matric)
            sam_max, sam_std = np.amin(sams_matric), np.std(sams_matric)
            traintime_average, testtime_average = np.average(train_time), np.average(time_test)

            pt = PrettyTable()
            pt.field_names = ["Model", "MPSNR", "MSSIM", "MSAM", "TRAIN_TIME", "TESTTIME"]
            pt.add_row([f"{model_type}_{noise_type}_{experiment_type} ","{:.4f} ± {:.4f}".format(psnr_max, psnr_std),"{:.4f} ± {:.4f}".format(ssim_max,ssim_std),"{:.4f} ± {:.4f}".format(sam_max,sam_std), "Average {:.4f}".format(traintime_average), "Average {:.4f}".format(testtime_average)])
            self.StreamLit.write(pt)
            return psnrs_matric, ssims_matric, sams_matric, len(data_file)
    
    def plots_file_handel(self, model_type = None, psnrs_matric = None, ssims_matric = None, sams_matric = None, df_size = None):
        with self.StreamLit.status(f"PSNR - SSIM - SAM graph of {model_type}."):
            epochs = range(0, df_size)
            MPSNR, MSSIM, MSAM = self.StreamLit.columns(3)
            with MPSNR:
                plt.plot(epochs, psnrs_matric, 'b', label='MPSNR')
                plt.title(f'MPSNR with {model_type}')
                plt.xlabel('Bands')
                plt.ylabel('MPSNR')
                plt.legend()
                plt.xlim(0,df_size)
                self.StreamLit.pyplot()
            with MSSIM:
                plt.plot(epochs, ssims_matric, 'b', label='MSSIM')
                plt.title(f'MSSIM with {model_type}')
                plt.xlabel('Bands')
                plt.ylabel('MSSIM')
                plt.legend()
                plt.xlim(0,df_size)
                self.StreamLit.pyplot()
            with MSAM:
                plt.plot(epochs, sams_matric, 'b', label='MSAM')
                plt.title(f'MSAM with {model_type}')
                plt.xlabel('Bands')
                plt.ylabel('MSAM')
                plt.legend()
                plt.xlim(0,df_size)
                self.StreamLit.pyplot()

    def mat_plot_handel(self, uploaded_file_mat = None, model_type = None, noise_type = None, experiment_type = None):
        with self.StreamLit.status(f"HSI {model_type} Denoise of {experiment_type} with {noise_type} noise."):
            data_file = scipy.io.loadmat(uploaded_file_mat)['Image']
            if experiment_type == "Simulation":
                data_file = data_file[:, :, (56,25,16)]
            elif experiment_type == "Real":
                data_file = data_file[:, :, (6,8,203)]
            fig = plt.figure(frameon=False, figsize = (5, 5))
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)
            fig.add_axes(ax)
            plt.imshow(data_file.clip(0, 1))
            plt.axis('off')
            self.StreamLit.pyplot()
    
    def model_selection_procedure(self,uploaded_file_mat = None, uploaded_file_csv = None, model_type = None, noise_type = None, experiment_type = None):
        if (model_type == "AAFEHDN") and (experiment_type == "Simulation"):
            with self.StreamLit.spinner('Start the Simulation Process of AAFEHDN for HSI Denoising Results.'): 
                psnrs_matric, ssims_matric, sams_matric, df_size = self.csv_table_file_headel(uploaded_file_csv = uploaded_file_csv, model_type = model_type, 
                                                                                     noise_type = noise_type, experiment_type = experiment_type)
                self.plots_file_handel(model_type = model_type, psnrs_matric = psnrs_matric, ssims_matric = ssims_matric, sams_matric = sams_matric, df_size = df_size)
                self.mat_plot_handel(uploaded_file_mat = uploaded_file_mat, model_type = model_type, noise_type = noise_type, experiment_type = experiment_type)

        elif (model_type == "AAFEHDN") and (noise_type == "Sigma") and (experiment_type == "Real"):
            with self.StreamLit.spinner('Start the Real Process of AAFEHDN for HSI Denoising Results.'): 
                psnrs_matric, ssims_matric, sams_matric, df_size = self.csv_table_file_headel(uploaded_file_csv = uploaded_file_csv, model_type = model_type, 
                                                                                     noise_type = noise_type, experiment_type = experiment_type)
                self.plots_file_handel(model_type = model_type, psnrs_matric = psnrs_matric, ssims_matric = ssims_matric, sams_matric = sams_matric, df_size = df_size)
                self.mat_plot_handel(uploaded_file_mat = uploaded_file_mat, model_type = model_type, noise_type = noise_type, experiment_type = experiment_type)

        if (model_type == "DeNet") and (experiment_type == "Simulation"):
            with self.StreamLit.spinner('Start the Simulation Process of DeNet for HSI Denoising Results.'): 
                psnrs_matric, ssims_matric, sams_matric, df_size = self.csv_table_file_headel(uploaded_file_csv = uploaded_file_csv, model_type = model_type, 
                                                                                     noise_type = noise_type, experiment_type = experiment_type)
                self.plots_file_handel(model_type = model_type, psnrs_matric = psnrs_matric, ssims_matric = ssims_matric, sams_matric = sams_matric, df_size = df_size)
                self.mat_plot_handel(uploaded_file_mat = uploaded_file_mat, model_type = model_type, noise_type = noise_type, experiment_type = experiment_type)

        elif (model_type == "DeNet") and (noise_type == "Sigma") and (experiment_type == "Real"):
            with self.StreamLit.spinner('Start the Real Process of DeNet for HSI Denoising Results.'): 
                psnrs_matric, ssims_matric, sams_matric, df_size = self.csv_table_file_headel(uploaded_file_csv = uploaded_file_csv, model_type = model_type, 
                                                                                     noise_type = noise_type, experiment_type = experiment_type)
                self.plots_file_handel(model_type = model_type, psnrs_matric = psnrs_matric, ssims_matric = ssims_matric, sams_matric = sams_matric, df_size = df_size)
                self.mat_plot_handel(uploaded_file_mat = uploaded_file_mat, model_type = model_type, noise_type = noise_type, experiment_type = experiment_type)

        if (model_type == "MemNet") and (experiment_type == "Simulation"):
            with self.StreamLit.spinner('Start the Simulation Process of MemNet for HSI Denoising Results.'): 
                psnrs_matric, ssims_matric, sams_matric, df_size = self.csv_table_file_headel(uploaded_file_csv = uploaded_file_csv, model_type = model_type, 
                                                                                     noise_type = noise_type, experiment_type = experiment_type)
                self.plots_file_handel(model_type = model_type, psnrs_matric = psnrs_matric, ssims_matric = ssims_matric, sams_matric = sams_matric, df_size = df_size)
                self.mat_plot_handel(uploaded_file_mat = uploaded_file_mat, model_type = model_type, noise_type = noise_type, experiment_type = experiment_type)

        elif (model_type == "MemNet") and (noise_type == "Sigma") and (experiment_type == "Real"):
            with self.StreamLit.spinner('Start the Real Process of MemNet for HSI Denoising Results.'): 
                psnrs_matric, ssims_matric, sams_matric, df_size = self.csv_table_file_headel(uploaded_file_csv = uploaded_file_csv, model_type = model_type, 
                                                                                     noise_type = noise_type, experiment_type = experiment_type)
                self.plots_file_handel(model_type = model_type, psnrs_matric = psnrs_matric, ssims_matric = ssims_matric, sams_matric = sams_matric, df_size = df_size)
                self.mat_plot_handel(uploaded_file_mat = uploaded_file_mat, model_type = model_type, noise_type = noise_type, experiment_type = experiment_type)
    
    def setup_config(self):
        with self.StreamLit.container(border=True):
            model_type, noise_type, experiment_type = self.input_types_config()
            if model_type is not None and noise_type is not None and experiment_type is not None:
                uploaded_file_mat, uploaded_file_csv = self.input_data_config(model_type = model_type, noise_type = noise_type, experiment_type = experiment_type)
                if uploaded_file_mat is not None and uploaded_file_csv is not None:
                    self.StreamLit.success('Mat and CSV files of Hyperspectral Image are Upload', icon="✅")
                    self.model_selection_procedure(uploaded_file_mat = uploaded_file_mat, uploaded_file_csv = uploaded_file_csv, model_type = model_type, 
                                    noise_type = noise_type, experiment_type = experiment_type)
                else:
                    self.StreamLit.info("Please upload the Mat and CSV files of Hyperspectral Image.")
            else:
                self.StreamLit.warning("Please enter type for all fields.")
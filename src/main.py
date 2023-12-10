import streamlit as st
import numpy as np

from data_process import DataProcess
from train_test import TrainTest
from model_config import ModelConfig
from interface import InterfaceConfig


st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_page_config(
    page_title="Hyperspectral Image Denoising",
    page_icon="👻",
    layout="wide",
    initial_sidebar_state="collapsed"
)
st.markdown("""
    <style>
        .reportview-container {
            margin-top: -2em;
        }
        .stDeployButton {display:none;}
        footer {visibility: hidden;}
        #stDecoration {display:none;}
    </style>
""", unsafe_allow_html=True)

st.markdown("""
            ---
            ### Hyperspectral Image Denoising using Attention and Adjacent Features Extraction Hybrid Dense Network
            ---
            """)

########################################### Pre Processing ###########################################

def Simulation_Section(dataprocess = None, traintest = None):
    st.header("Washington DC Mall University (Simulation Dataset Evaluation)")
    uploaded_file = st.file_uploader("Choose an DC Mall Hyperspectral Image.", type=["tif"])
    train_column, test_column = st.columns(2)

    if uploaded_file is not None:
        dc_image = dataprocess.load_data(dc_mall_dataset = uploaded_file)
        with st.expander("See the DC Mall Hyperspectral Image"):
            dc_image_mall = np.stack((dc_image[:,:,56],dc_image[:,:,26],dc_image[:,:,16]),2)
            dataprocess.dc_mall_plot(image = dc_image_mall, name_file = "University", save = True)

        with st.container(border=True):
            noise_type, scale, pathsize, stride, batchsize = traintest.inputs_features()
            if noise_type is not None and scale is not None and pathsize is not None and stride is not None and batchsize is not None:
                    with train_column:
                        with st.container(border=True):
                            if st.button(label="Start Training Procedure"):          
                                # st.write(noise_type, scale, pathsize, stride, batchsize)
                                with st.spinner('HSI Data In Training Process'): 
                                    originalImage, noiseImage = traintest.train_procedure(dc_image = dc_image, noise_type = noise_type, scale = scale, pathsize = pathsize, stride = stride, batchsize = batchsize)  
                                    if originalImage is not None:
                                        with st.expander("See the DC Mall Hyperspectral of Tain Dataset Images"):
                                            original_dc_image_mall = np.stack((originalImage[20,:,:,56],originalImage[20,:,:,26],originalImage[20,:,:,16]),2)
                                            dataprocess.dc_mall_plot(image = original_dc_image_mall, name_file = "TrainOriginalUniversity", save = True)
                                            noise_dc_image_mall = np.stack((noiseImage[10,:,:,56],noiseImage[10,:,:,26],noiseImage[10,:,:,16]),2)
                                            dataprocess.dc_mall_plot(image = noise_dc_image_mall, name_file = "TrainNoiseUniversity", save = True)
                                            traintest.train_mat_file_save(original_image = originalImage, noise_image = noiseImage, file_name = str(noise_type))
                    with test_column:
                        with st.container(border=True):
                            if st.button(label="Start Testing Procedure"):
                                with st.spinner('HSI Data In Testing Process'): 
                                    originalImage, noiseImage = traintest.test_procedure(dc_image = dc_image, noise_type = noise_type)
                                    if originalImage is not None:
                                        with st.expander("See the DC Mall Hyperspectral of Test Dataset Images"):
                                            original_dc_image_mall = np.stack((originalImage[:,:,56],originalImage[:,:,26],originalImage[:,:,16]),2)
                                            dataprocess.dc_mall_plot(image = original_dc_image_mall, name_file = "TestOriginalUniversity", save = True)
                                            noise_dc_image_mall = np.stack((noiseImage[:,:,56],noiseImage[:,:,26],noiseImage[:,:,16]),2)
                                            dataprocess.dc_mall_plot(image = noise_dc_image_mall, name_file = "TestNoiseUniversity", save = True)
                                            traintest.test_mat_file_save(original_image = originalImage, noise_image = noiseImage, file_name = str(noise_type))
            else:
                st.warning("Please enter values for all fields.")              
    else:
        st.info("Please upload the Simulation Hyperspectral Image.")

def Real_Section(dataprocess = None, traintest = None):
    st.header("Indian Pines (Real Dataset Evaluation)")
    uploaded_file = st.file_uploader("Choose an Indian Pines Hyperspectral Image.", type=["mat"])

    if uploaded_file is not None:
        indian_pines_image = dataprocess.load_real_data(indian_pines_dataset = uploaded_file)
        with st.expander("See the Indian Pines Hyperspectral Image"):
            indian_pines_image_mall = np.stack((indian_pines_image[:,:,6],indian_pines_image[:,:,8],indian_pines_image[:,:,203]),2)
            dataprocess.indian_pines_plot(image = indian_pines_image_mall, name_file = "Real_Predefined_Noise", save = True)
            traintest.real_mat_file_save(original_image = indian_pines_image, file_name = "Real_Predefined_Noise")
    else:
        st.info("Please upload the Real Hyperspectral Image.")


########################################### HSI Model ###########################################

def Model_Setup():
    modelconfig = ModelConfig(st = st)
    modelconfig.setup_config()

########################################### Post Process ###########################################


def InterFace_Section():
    interfaceconfig = InterfaceConfig(st = st)
    interfaceconfig.setup_config()


########################################### ["PreProcess", "HSI Model", "PostProcess"] ###########################################

PreProcess, HSIModel, PostProcess = st.tabs(["Pre Process", "HSI Model", "Post Process"])

with PreProcess:
    dataprocess = DataProcess(st = st)
    traintest = TrainTest(st = st)
    dataprocess.img_mat_folder_make()
    Simulation_Section(dataprocess = dataprocess, traintest = traintest)
    st.markdown("---")
    Real_Section(dataprocess = dataprocess, traintest = traintest)

with HSIModel:
    Model_Setup()

with PostProcess:
    InterFace_Section()
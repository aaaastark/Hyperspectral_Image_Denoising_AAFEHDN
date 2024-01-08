
# **Hyperspectral Image Denoising using Attention and Adjacent Features Extraction Hybrid Dense Network**
The efficacy of addressing high-level semantic tasks in hyperspectral images (HSIs) is often hindered by the intricate nature of the imaging environment, as well as issues related to corruption and degeneration. These factors contribute to various types of noise in HSIs. While conventional natural image denoising methods have seen success, current convolutional neural network (CNN)-based HSIs denoising approaches grapple with the challenge of insufficient noise suppression and inadequate feature extraction. Consequently, there remains ample room for enhancement in this domain. In response to these challenges, a pioneering HSIs denoising algorithm has been introduced, leveraging Attention and Adjacent Features - Hybrid Dense Network (AAFHDN). This algorithm excels in decomposing high-frequency features, maintaining geometric characteristics as structural priors, and extracting band correlation from adjacent spatial and multiscale separable spectral features. The efficacy of the proposed method has been assessed through experiments conducted on both simulated and real-world noisy images. The results underscore that the AAFHDN algorithm surpasses established traditional methods in both quantitative assessments and visual impact. The heightened denoising performance of this approach holds promise for improving subsequent classification and target detection tasks in the realm of HSIs.

<p align="center" width="100%">
    <img width="100%" hight="100%" src="https://github.com/aaaastark/Hyperspectral_Image_Denoising_AAFEHDN/blob/master/Poster/Poster%20FYP%20Standee.png"> 
</p>

# **Revolutionizing Hyperspectral Image Denoising: Unveiling Our AAFHDN HSI Denoising App**
Introducing our cutting-edge application, developed on the revolutionary AAFHDN HSI Denoising algorithm. In addressing the complexities of hyperspectral images (HSIs), this app leverages Attention and Adjacent Features - Hybrid Dense Network (AAFHDN) to overcome challenges related to noise, corruption, and feature extraction. To gain a deeper understanding of the app's structure and functionality, we invite you to watch our informative video, where we explain how the AAFHDN algorithm excels in decomposing high-frequency features, preserving geometric characteristics, and extracting valuable band correlations from adjacent spatial and multiscale separable spectral features. Discover the future of hyperspectral image processing with our innovative App.

## **Video Section**
[![HSI App](https://github.com/aaaastark/Hyperspectral_Image_Denoising_AAFEHDN/blob/master/data/AppView.png)](https://drive.google.com/file/d/1T_8KnEldcaglV6O6XlDaNTFCYHe_dl3Z/view?usp=sharing)

## **Code Section**
> To begin, verify the Python IDE version on your local system. For this project, we specifically utilize Python version `3.11.5`. If any problems arise during the installation of packages, we suggest using the designated Python version `3.11.5`. Alternatively, you can opt for the default Python IDE version installed on your system.

1. Clone the `Hyperspectral_Image_Denoising_AAFEHDN` repository to your local system by fetching it from the `www.github.com/aaaastark/` page.
```bash
git clone https://github.com/aaaastark/Hyperspectral_Image_Denoising_AAFEHDN.git
```
2. To create a virtual environment, go to your project’s directory and run venv. This will create a new virtual environment in a local folder .venv
```bash
py -m venv .environment
```
3. Before you can start installing or using packages in your virtual environment you’ll need to activate it. Activating a virtual environment will put the virtual environment-specific python and pip executables into your shell’s PATH
```bash
.\environment\Scripts\activate
```
4. To confirm the virtual environment is activated, check the location of your Python interpreter: `Return Output None`
```bash
where python
```
5. Instead of installing packages individually, pip allows you to declare all dependencies in a `Requirements File`. For example you have a `requirements.txt` file. And tell pip to install all of the packages in this file using the -r flag:
```bash
py -m pip install -r requirements.txt
```
6. Retrieve the dataset files named `DC Mall` and `Indian Pines` from the provided Drive URL and subsequently place them into the designated `data` folder for seamless access.
```bash
DC Mall: https://drive.google.com/file/d/14h-1GeB9ILb2WkwNOBhnEKOoBPPBwiKh/view?usp=sharing
Indian Pines: https://drive.google.com/file/d/183QvCbrVTIR_KlIgpr-RzEu3_vXl68OH/view?usp=sharing
```
7. Navigate to the `src` directory and execute the `main.py` file to initiate the AAFEHDN App designed for Hyperspectral Image Denoising. Subsequently, open the Streamlit interface in a browser tab and proceed to launch various task processes, including Pre Processing, HSI Model, and Post Processing.
```bash
cd src
streamlit run main.py
```

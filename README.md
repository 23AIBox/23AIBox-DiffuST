# 23AIBox-DiffuST
DiffuST is a powerful method for spatial transcriptomics denoising, which enables enhanced downstream analysis and provides valuable insights into complex biological systems.
## DiffuST: denoising spatial transcriptomics with a latent diffusion model
DiffuST is a powerful method for spatial transcriptomics denoising, which enables enhanced downstream analysis and provides valuable insights into complex biological systems. Firstly, we leverage the spatial information of the spatial transcriptomics dataset to construct an undirected neighborhood graph where spots close to each other are connected. Next, the encoder of a graph autoencoder[1] is built to effectively compress the gene expression profiles and spatial similarity into a latent representation by iteratively aggregating gene expressions from neighboring spots. Following this, a latent diffusion model[2] is designed to generate a denoising latent representation. During the training process, starting with latent representation, Gaussian noise is iteratively added over T discrete steps until z_T. This noising procedure is performed via the Markov process q(z_t|z_{t-1}). The diffusion model is trained to predict the noise added at each step, learning a time-conditional Fully Convolutional Network (U-Net)[3] that performs the reverse denoising process. After training, to generate a denoising latent representation, the diffusion model initiates the process with z_T, a highly noisy state of the original data. It then employs T steps of iterative denoising. During each step, the model uses the output from the previous denoising step as the input for the subsequent one. This iterative refinement continues until it reaches the final denoised representation. Furthermore, to effectively use mapped imaging data to guide the denoising process, DiffuST employs a pre-trained ViT[4] model to obtain image features and utilize a cross-attention layer to incorporate them into the reverse diffusion process. Ultimately, the denoising latent representation is passed through the decoder of the graph autoencoder to obtain denoising spatial gene expression. Further details about the DiffuST can be found in the Methods section.
## Set up environment
Set up the required environment using `requirements.txt` with Python. While in the project directory run:
pip install -r requirements.txt
```python
pip install -r requirements.txt
```
## Running RL-GenRisk
1. Training script for graph convolutional autoencoder: ./train_gae.ipynb

2. Training script for employs a pre-trained ViT model to obtain image features and utilize a cross-attention layer to incorporate them into the reverse diffusion process: train_img.ipynb

3. Training script for the diffusion model and denoising spatial transcriptomics: main.ipynb
## 

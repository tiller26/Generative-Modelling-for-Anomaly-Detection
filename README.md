# Generative-Modelling-for-Anomaly-Detection
The Generative Spectrum: A Comparative Benchmark of Autoencoders, Variational  Autoencoders, and Denoising Diffusion Models for Industrial Anomaly Detection 

Unsupervised visual anomaly detection identifies defects without labeled anomaly 
data by training models exclusively on normal samples to learn pristine appearance 
distributions. At inference time, query images are reconstructed or denoised; 
subtracting the output from the input yields a residual error map flagging anomalies 
that the model cannot reconstruct. Three major generative paradigms address this 
task deterministic Autoencoders (AEs), probabilistic Variational Autoencoders 
(VAEs), and Denoising Diffusion Implicit Models (DDIM). However, existing literature 
rarely compares them side-by-side under unified, controlled conditions. This project 
establishes a benchmark across AE, VAE, and DDIM architectures on the MVTec AD 
dataset to quantify trade-offs between reconstruction quality, localization accuracy, 
and inference latency 

**f-AnoGAN
- [Paper](https://www.sciencedirect.com/science/article/abs/pii/S1361841518302640)
- Similar to AnoGAN but added Encoder to enable fast mapping to latent space
	- allows fast evaluation whether or not novel images/images segments fall into trained manifold
- Training Process consists of two steps
	- GAN training
	- Encoder training based on the trained GAN model
		- Image-To-Image Autoencoder
		- Trained to map images to latent space
	- "In the case of a normal input image (and under the assumption of a perfect generator and a perfect encoder), mapping from image space to the latent space via the encoder and subsequent mapping from latent space back to image space via the generator should closely resemble the identity transform"

___

**GANomaly
- [Paper](https://arxiv.org/abs/1805.06725)
- [Article](https://towardsdatascience.com/ganomaly-paper-review-semi-supervised-anomaly-detection-via-adversarial-training-a6f7a64a265f#cc2c)
- generator is an autoencoder
- **Based on Conditional Generative Adversarial Network
	- Is "conditioned" with labels of the training samples
		- in this case: jointly learns the generation of higher dimensional image space and the inference of latent space
	- see [Article](https://medium.com/analytics-vidhya/anomaly-detection-using-generative-adversarial-networks-gan-ca433f2ac287) and [Article](https://towardsdatascience.com/cgan-conditional-generative-adversarial-network-how-to-gain-control-over-gan-outputs-b30620bd0cc8)
- Incorporates encoder-decoder-encoder sub-networks in the generator network to map the image to a lower dimensional vector which is then used to reconstruct the generated output image
	- Similar to Adversarial Autoencoder 
	- Autoencoder with an additional Encoder after the image is generated
		- Maps the generated image to its latent space representation thus the reconstruction into the latent space point z'
	- Autoencoder works as the Generator of the model
	- Discriminator (network at the bottom) learns to tell if image is real or fake
- Learning performed via three loss-/objective-function that consist of
	- encoder loss
		- Generator learns how to encode features of the generated images for normal samples
	- contextual loss
		- distance between generated and original images
	- adversarial loss
		- utilizes intermediate layer of discriminator
![[Pasted image 20230410134820.png]]

___

**Efficient GAN-Based Anomaly Detection (EGBAD)
- [Paper](https://arxiv.org/abs/1605.09782v7)
- Bi-Directional GAN [Source](https://paperswithcode.com/method/bigan)
	- In addition to the generator $G$ from the standard GAN, BiGAN includes an encoder $E$ which maps data $x$ to latent representations $z$
- "...avoid the computationally expensive step of recovering a latent representation at test time."
	- thus no need to do the expensive iterative backpropagation at the end of AnoGAN

___

**Improving Unsupervised Defect Segmentation by Applying Structural Similarity to Autoencoders
- [Paper](https://arxiv.org/abs/1807.02011)
- Defect segmentation
- "We propose to use a perceptual loss function based on structural similarity that examines inter-dependencies between local image regions, taking into account luminance, contrast, and structural information, instead of simply comparing single pixel values"

---

**Uninformed Students: Student-Teacher Anomaly Detection with Discriminative Latent Embeddings
- [Paper](https://arxiv.org/abs/1911.02357)
- Student-Teacher framework
	- "Student networks are trained to regress the output of a descriptive teacher network that was pretrained on a large dataset of patches from natural images"
	- Anomalies are detected when the outputs of the student networks differ from that of the teacher network
	- Adresseing GANs and VAEs
		- "These detect anomalies using per-pixel reconstruction errors or by evaluating the density obtained from the model’s probability distribution. This has been shown to be problematic due to inaccurate reconstructions or poorly calibrated likelihoods"
		- "frames anomaly detection as a feature regression problem"
		- Teacher NN learns a variety of patches of anomalous samples
		- Student NNs are trained on anomaly free data to mimic teachers output
		- Uncertainty of Student Nets is indicator for anomalous regions

![[Pasted image 20230410160333.png]]

----

**MSG-GAN: Multi-Scale Gradients for Generative Adversarial Networks
- [Paper](https://openaccess.thecvf.com/content_CVPR_2020/html/Karnewar_MSG-GAN_Multi-Scale_Gradients_for_Generative_Adversarial_Networks_CVPR_2020_paper.html)
- Not explicitly about AN
- Addresses training instability
	- Gradients at multiple scales can be used to generate high resolution images

![[Pasted image 20230410163405.png]]

- "MSG-GAN allows the discriminator to look at not only the final output (highest resolution) of the generator, but also at the outputs of the intermediate layers (Fig. 2). As a result, the discriminator becomes a function of multiple scale outputs of the generator and importantly, passes gradients to all the scales simultaneously"
---

**A Style-Based Generator Architecture for Generative Adversarial Networks (StyleGAN)
- [Paper](https://arxiv.org/abs/1812.04948) 
- [Article](https://machinelearningmastery.com/introduction-to-style-generative-adversarial-network-stylegan/)
- Disentanglement of features/properties in images
	
 **Can AdaIN Style-Transfer be used to identify anomalous regions?
 --> Assumption: Style transfer only affects learned features but leaves anomalous features untouched. Creating different styles of the same input image an measuring the distance between those could be used as anomaly score
 --> [Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization](https://arxiv.org/abs/1703.06868)

---

**pi-GAN: Periodic Implicit Generative Adversarial Networks for 3D-Aware
Image Synthesis
- also see Pix2NeRF: Unsupervised Conditional p-GAN for Single Image to Neural Radiance Fields Translation [LINK](https://openaccess.thecvf.com/content/CVPR2022/html/Cai_Pix2NeRF_Unsupervised_Conditional_p-GAN_for_Single_Image_to_Neural_Radiance_CVPR_2022_paper.html)

---

**Unsupervised Discovery of Interpretable Directions in the GAN Latent Space

---

**Deep Autoencoding Models for Unsupervised Anomaly Segmentation in Brain MR Images

---

**Reconstruction by inpainting for visual anomaly detection
- [Article](https://towardsdatascience.com/paper-review-reconstruction-by-inpainting-for-visual-anomaly-detection-70dcf3063c07)
- U-Net Based ([Convolutional Autoencoder](https://medium.com/dataseries/convolutional-autoencoder-in-pytorch-on-mnist-dataset-d65145c132ac))

---

**VT-ADL: A Vision Transformer Network for Image Anomaly Detection and Localization

---

**AnoDDPM: Anomaly Detection With Denoising Diffusion Probabilistic Models Using Simplex Noise

---

**Aggregated Contextual Transformations for High-Resolution Image Inpainting
- Especially the combination of loss functions







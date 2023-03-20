- [arxiv](https://arxiv.org/abs/1703.05921)

**TL;DR
- We train a DCGAN, so it generates images from a given domain
- We take the then trained generator to create two loss functions
	- Residual Loss
	- Discriminitive Loss
- The loss functions are used together with Backpropagation to find the input-noise of a given (anomalous) query image
- Starting from that input image, we try to find the input noise ($z_{\Gamma}$) of the training-image that is closest to it
- Once we found $z_{\Gamma}$, we calculate an anomaly score
- To get the anomalous region we subtract the generated image at $z_{\Gamma}$ from the query image

-----

- Main idea is the mapping of images to the corresponding latent space noise
- Latent space has continuous properties
	- tiny change in latent space coordinates resonates in tiny changes in resulting mapped image
	- sampling two points in latent space that are close to each other creates similar images

- The paper utilizes an approach where the inverse is used
	- $\mu(x) = x \longrightarrow z$
	- given an image $x$ (query image), find the point $z$ in latent space where an image $G(x)$ is most similar to query image $x$ and that is located in the manifold $X$
	- "The degree of similarity of $x$ and $G(z)$ depends on to which extend the query image follows the data distribution $p_{g}$ that was used for training of the generator"

- Starts off with regular GAN
	- Generator learns to map latent space to a realistic image
		- $G(z) = z \longmapsto x$
		- images are part of manifold $X$
	- Randomly sample $z_{1}$ from latent space $Z$
	- Feed $z_{1}$ into Generator to get image $G(z_{1})$ 
	- Based on $G(z_{1})$ define a loss function which provides a gradient to update the coefficients of $z_{1}$ in order to get $z_{2}$
		- $z_{2}$ leads to a generated image that is closer to query image $x$
	- Repeat above step until best fitting image $G(z_{\Gamma})$ is achieved
		- Back-propagation steps: $\gamma = 1,2,3,...,\Gamma$

- In order to achieve the above two different loss functions are defined
	- Residual loss
		- enforces similarity of generated images $G(z_{\gamma})$ and query image $x$
		- $\Large L_{R} = \sum_{}^{} \left| x - G(z_{\gamma}) \right|$
		
	- Discrimination loss
		- Enforces generated image $G(z_{\Gamma})$ to lie in learned mainfold $X$ 
		- Other than in [[Semantic Image Inpainting with Perceptual and Contextual Losses]] $z_{\Gamma}$ is not updated to better fool the discriminator
			- **approach is to update $z_{\Gamma}$ so that it matches $G(z_{\Gamma})$ 
		- Authors use feature matching as proposed in [[Improved Techniques for Training GANs]] (section 3.1) to achieve this
			- Intermediate feature representations are used
			- $\Large L_{D}(z_{\Gamma}) = \sum_{}^{} \left| f(x) - f(G(z_{\Gamma})) \right|$
				- with $f(\bullet)$ being the output of an intermediate layer of the discriminator
					- this takes **all features learned by the discriminator** into account an does not rely only on the final decision of the discriminator

The overall loss function is therefore:

$\Large L(z_{\Gamma}) = (1 - \lambda) \cdot L_{R}(z_{\Gamma}) + \lambda \cdot L_{D}(z_{\Gamma})$

In order to detect anomalies, the loss function is turned into an anomaly-score $A(x)$ with

$\Large A(x) = (1 - \lambda) \cdot R(x) + \lambda \cdot D(x)$, where

- $R(x)$ is the Residual Score
	- Residual loss at the last ($\Gamma^{th}$) iteration of mapping procedure --> $L_{R}(z_{\Gamma})$
- $D(x)$ is the discriminatory score
	- Discriminatory Loss at the last ($\Gamma^{th}$) iteration of mapping procedure --> $L_{D}(z_{\Gamma})$

In order to find the anomalous region, the generated image at then found coordinates $z_{\Gamma}$ is subtracted from the query image $x$. This creates the residual image $x_{R}$

$\Large x_{R} = \left| x - G(z_{\Gamma})\right|$


In the following step, we attempt to identify the locations of those regions. 
\begin{enumerate}
	\item Weakly supervised approach
		\begin{enumerate}
			\item BLA
		\end{enumerate}
	\item Utilizes DCGAN, AnoGAN, and Direction Discovery
	\item A dataset is generated that contains both normal image data and anomalies (polluted dataset)
	\item Image-data is used to train a DCGAN so that it can generate images that represent the domain of the training data (including the anomalies)
	\item After training, DCGAN can map a random noise vector to an image that looks like the ones from the training distribution
	\item Direction Discovery is used to identify meaningful directions inside the DCGAN latent space
	\item Human feedback is utilized to identify directions that lead to anomalous regions in the GAN's latent space
	\item Distance to anomalous directions is used as a metric to yield an anomaly score for a given sample
\end{enumerate}
- general description
- normalization and batch normalization
	- [article](https://towardsdatascience.com/understanding-batch-normalization-for-neural-networks-1cd269786fa6)
- kernel
	- [Kernel](https://medium.com/codex/kernels-filters-in-convolutional-neural-network-cnn-lets-talk-about-them-ee4e94f3319)
	- [Channel and Kernel](https://medium.com/analytics-vidhya/difference-between-channels-and-kernels-in-deep-learning-6db818038a11)
	- padding
	- stride

### Convolution

A convolution is a mathematical operation on two functions defined as

$\Large (f \ast g) (t) = \int_{\infty}^{\infty} f(\tau) \cdot g(t-\tau)d\tau$, where $f$ and $g$ are functions

- The first argument ($f$) is called input (in machine learning it is often an image)
- The second argument ($g$) is called kernel or filter
- The output array is regularly called feature-map

The outcome of the operation is a third function that determines how the second function changes the shape of the first one.

In machine learning applications the input is usually a multidimensional arrays of data.
The kernel is a multidimensional array of parameters that are adopted by the algorithm. Both can also be considered tensors.

Convolutions are often used in image processing and therefore also in machine learning where with image data as the learning subject. In practical terms the kernel can be seen as a filter of a x a pixels that flows over an image

![[Pasted image 20230319145037.png]]
(Source: IBM)

By using different kernel (filters) different effects can be applied to an image. 

![[1 m4IsBwYv7QEND-y6xWw3Yw.gif]]

In machine learning convolutional layer are usually used for feature detection.
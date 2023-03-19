- general description
- normalization and batch normalization
	- [article](https://towardsdatascience.com/understanding-batch-normalization-for-neural-networks-1cd269786fa6)

### Convolution

A convolution is a mathematical operation on two functions defined as

$\Large (f \ast g) (t) = \int_{\infty}^{\infty} f(\tau) \cdot g(t-\tau)d\tau$, where $f$ and $g$ are functions

- The first argument ($f$) is called input (in machine learning it is often an image)
- The second argument ($g$) is called kernel or filter
- The output array is regularly called feature-map

The outcome of the operation is a third function that determines how the second function changes the shape of the first one.

In machine learning applications, the input is usually a multidimensional array of data.
The kernel is a multidimensional array of parameters that are adopted by the algorithm. Both can also be considered tensors.

Convolutions are often used in image processing and therefore also in machine learning, where with image data as the learning subject. In practical terms, the kernel can be seen as a filter of (a x a) pixels that flows over an image

![[Pasted image 20230319145037.png]]
(Source: IBM)

By using different kernel (filters) different effects can be applied to an image. 

In machine learning, convolutional layer are usually used for feature detection.
- The stepsize used to move the kernel is called stride
- In order to maintain tensor-sizes, the input ist often surrounded withadditional fields. The number of rings around the input-tensor is called padding

![[Pasted image 20230319163758.png]]

![[Pasted image 20230319163833.png]]


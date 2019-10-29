# MNIST digits sequence generator

The goal of this project is to write a program that can generate images
representing sequences of numbers, for data augmentation purposes.

These images would be used to train classifiers and generative deep learning
models.  A script that saves examples of generated images is helpful in
inspecting the characteristics of the generated images and in inspecting the
trained models behaviors.

## Specifications

python --version: 3.7.4

Generally, the parameters are passed by command line tool to the APIs.
The APIs of the script are listed below :

* sequence: the sequence of digits to be generated
* min spacing: minimum spacing between consecutive digits
* max spacing: maximum spacing between consecutive digits
* image width: width of the generated image

## Technical talks

The MNIST database contains 60,000 training images and 10,000 testing images,
which can be directly imported by keras library. Meanwhile,
some database like EMNIST contains much more images.

For the generation of training data, let's say digits sequence here,
there're mainly two methods, raw image combinations, and GAN.
It is known that the performance of GAN is highly dependent on the hardware,
that the deep neural network usually needs several GPUs.

On the other side, raw images combinations seem fast at first,
however, when the scale of the training data gets bigger, more time will be spent.
One possible way is to fix the images into a relatively small size,
or generate the images multiple times.

## Test

* output:  
  <pre><code>python driver.py 12345
  python driver.py 67890 0 0
  python driver.py 13579 0 50 14</code></pre>
* output2:  
  use input.xlsx as input

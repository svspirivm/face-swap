face-swap
======

My approach
------

In this project, the goal was to swap faces in two images using mathematical approaches. The project uses [mediapipe](https://github.com/google/mediapipe) to detect facial landmarks in the input images and then uses the detected landmarks to perform face swapping.

At first the facial landmarks are extracted from the input images using mediapipe. Then the face alignment is performed in order to uniform face position for all the images. After that the algorithms iterates through each of the triangles produced by mediapipe's landmarks. A transformation matrix is used to map the triangles of one image onto the other. The last step merges all the transformed triangles into the target face with [seamless cloning approach](https://docs.opencv.org/3.4/df/da0/group__photo__clone.html).

Obviously, this algorithm doesn't produce such state-of-art results as neural network-based approaches, but it doesn't require a large dataset of paired images for training, which can be time-consuming and expensive to collect. The main limitation of this approach is that it does not work well for faces with drastically different facial features or expressions. In such cases, the seamless cloning algorithm may not be able to blend the two faces together seamlessly, resulting in visible artifacts or discontinuities in the final image.

Existing solutions
------

In order to evaluate the effectiveness of the developed approach for face swapping, a comparison was made with existing approaches that utilize neural networks for the same task. As expected, neural network-based approaches produce more realistic results, thanks to their ability to learn complex patterns and generate highly accurate facial reconstructions. Neural networks excel at capturing intricate details, fine textures, and nuances of facial expressions, resulting in visually compelling and seamless face swaps. Furthermore, neural network-based approaches benefit from large-scale training data, allowing them to generalize well across various face types, poses, and lighting conditions. 

Future research and advancements in both mathematical and neural network-based approaches could contribute to bridging the gap between efficiency and realism in face swapping techniques. Combining the strengths of both approaches could potentially yield improved results, allowing for more realistic and efficient face swaps in various contexts.

[GHOST](https://github.com/ai-forever/ghost) was picked for the code analysis. 

[Inference file](https://github.com/ai-forever/ghost/blob/main/inference.py) shows the processing pipeline. After the crop and alignment is performed, the AEI-Net (Adaptive Embedding Integration Network) is applied to the image in the following way:

1. The pretrained ArcFace model encodes the identity of the source person to a vector. 
2. Another UNET-like model extracts the features from the target image. This multi-level attributes encoder concerned with embedding the target image into a space that describes the attributes that we want to preserve when we swap faces. The authors used ResNet at this step and extracted feature maps on the different layers. 
3. AAD Generator (Adaptive Attentional Denormalization Generator) integrates the outputs of the previous two sub-networks in increasing spatial resolution order to produce the final output of AEI-Net. It sequentially mixes the attribute vector evaluated from the target image and the identity vector evaluated from the source image so that a new face will contain both the source identity and the target attribute features. So it mixes attribute and identity vectors with the results of previous generation steps using AAD ResBlocks. Each of AAD ResBlocks consists of several AAD blocks. In each AAD block AdaIN (Adaprive Instance Normalization) is applied to the identity vector. AdaIN first standardizing the output of feature map to a standard Gaussian, then adding the vector as a bias term. SPADE (spatially-adaptive) block is applied to the attribute vector. The use of fewer AAD blocks speeds up the model significantly but results in a small reduction in the model quality. The authors used 2 AAD blocks in the main model.
4. A multiscale discriminator improves the output synthesis quality by comparing real and fake images.

The general loss of the original AEI-Net consists of 4 parts: reconstruction loss, attribute loss, identity loss and adversarial loss. 

1. __Recontruction loss__.
The idea of the original reconstrstion loss is that if we give model two same images then we don't want her to do anything with them - so that for two same inputs we expect the same output. In GHOST implementation we don't want the model to change anything not only when the source and the target images are the same, but also when the source and the target images are different photos of the same person. It makes sense - there is no need to swap faces when it's the same face. 
2. __Attribute loss__. 
This loss ensures that the attribute features for the model output and for the target image are close. 
3. __Identity loss__. 
We want identity encoder outputs for the source image and the model output to be similar in terms of cosine similarity. 
4. __Adversarial loss__.
GAN loss based on multiscale discriminator from Step 4. 

In order to produce more realistic results the authors added special loss for transfering gaze direction. This __Eye loss__ compares the heatmaps of the eye zone for the model output and the target image. 

The __Overall loss__ is the linear combination of all the five losses mentioned above. 



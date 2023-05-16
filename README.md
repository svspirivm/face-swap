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
2. Another UNET-like model extracts the features from the target image. This multi-level attributes encoder concerned with embedding the target image into a space that describes the attributes that we want to preserve when we swap faces.
3. AAD Generator (Adaptive Attentional Denormalization Generator) integrates the outputs of the previous two sub-networks in increasing spatial resolution order to produce the final output of AEI-Net. It sequentially mixes the attribute vector evaluated from the target image and the identity vector evaluated from the source image so that a new face will contain both the source identity and the target attribute features. 


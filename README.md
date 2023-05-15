# face-swap

## My approach
    In this project, the goal was to swap faces in two images using mathematical approaches. The project uses [mediapipe](https://github.com/google/mediapipe) to detect facial landmarks in the input images and then uses the detected landmarks to perform face swapping.

    At first the facial landmarks are extracted from the input images using mediapipe. Then the face alignment is performed in order to uniform face position for all the images. After that the algorithms iterates through each of the triangles produced by mediapipe's landmarks. A transformation matrix is used to map the triangles of one image onto the other. The last step merges all the transformed triangles into the target face with [seamless cloning approach](https://docs.opencv.org/3.4/df/da0/group__photo__clone.html).

    Obviously, this algorithm doesn't produce such state-of-art results as neural network based approaches, but it doesn't require a large dataset of paired images for training, which can be time-consuming and expensive to collect. The main limitation of this approach is that it does not work well for faces with drastically different facial features or expressions. In such cases, the seamless cloning algorithm may not be able to blend the two faces together seamlessly, resulting in visible artifacts or discontinuities in the final image.

## Existing solutions

   


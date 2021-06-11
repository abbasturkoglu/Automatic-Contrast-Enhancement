# Automatic-Contrast-Enhancement
1. Problem Definition:

    Problem is enhancing contrast of an 8-bit grayscale image by stretching pixel value range
    into 0 and 255. Main problem is finding minimum and maximum values in parallel manner.
    Procedure is mainly consists of 3 steps. Firstly, minimum and maximum values computed
    to find dynamic range of image. Then, the global minimum value is subtracted from the each
    pixel in the image to eliminate the effect offset by setting the minimum value to 0. Lastly all
    pixel values are scaled into range 0-255.
    To make enhancement faster, using parallel reduction is key point for design algorithm.
    Parallelization help to traverse the image in shorter time than CPU because of high number of
    cores.

2. Algorithm Description:

    Algorithm takes a bmp image as an input. By using stb library, image converted into a
    1D array. After conversion, by using formula below, contrast of the image enhanced.
    To implement formula on algorithm, I used 4 different CUDA kernels.

    • MinKernel:
    This kernel takes 3 arguments (image, size of image and the minimum pointer),
    Finds the minimum value and writes to the minimum pointer. Kernel starts copying the data
    from global to shared memory. I used warp reducing. One kernel does the job for finding
    minimum. To make process faster, I have also used Parallel Reduction v5.

    • MaxKernel:
    This kernel takes 3 arguments (image, size of image and the minimum pointer),
    finds the maximum value and writes to the maximum pointer. Kernel starts copying the data
    from global to shared memory. I used warp reducing. One kernel does the job for finding
    maximum. To make process faster, I have also used Parallel Reduction v5.

    • SubKernel:
    This kernel takes 2 arguments (image and global minimum). Subract the minimum
    value from the each pixel value and save it again on the same pixel value.

    • ScaleKernel:
    This kernel takes 2 arguments (image and scale constant). Multiply each pixel
    value found in SubKernel with scale constant. (255.0f / (max_host[0] - min_host[0]))

// Do not alter the preprocessor directives
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "stb_image.h"
#include "stb_image_write.h"
#include <stdint.h>

#define NUM_CHANNELS 1


// CPU baseline for finding minimum and maximum pixel values in an image
void min_max_of_img_host(uint8_t* img, uint8_t* min, uint8_t* max, int width, int height) {
    int max_tmp = 0;
    int min_tmp = 255;
    for (int n=0; n < width * height; n++){
        max_tmp = (img[n] > max_tmp) ? img[n] : max_tmp;
        min_tmp = (img[n] < min_tmp) ? img[n] : min_tmp;
    }
    *max = max_tmp;
    *min = min_tmp;

}

// CPU baseline for subtracting a value from all pixels in an image
void sub_host(uint8_t* img, uint8_t sub_value, int width, int height) {
    for (int n=0; n < width * height; n++){
        img[n] -= sub_value;
    }
}


// CPU baseline for scaling pixel values in an image avoiding finite precision
// integer arithmetic given "power" and "constant" values.
void scale_host(uint8_t* img, float constant, int width, int height) {
    for (int n=0; n < width * height; n++){
        img[n] = img[n] * constant; //note the implicit type conversion
    }
}


int main() {
    int width; //image width
    int height; //image height
    int bpp;  //bytes per pixel if the image was RGB (not used)

    uint8_t min_host, max_host;
    
    // Load a grayscale bmp image to an unsigned integer array with its height and weight.
    //  (uint8_t is an alias for "unsigned char")
    uint8_t* image = stbi_load("./samples/640x426.bmp", &width, &height, &bpp, NUM_CHANNELS);

    // Print for sanity check
    printf("Bytes per pixel: %d \n", bpp / 3); //Image is grayscale, so bpp / 3;
    printf("Height: %d \n", height);
    printf("Width: %d \n", width);

    // Calculate the value of minimum and maximum pixels (to be replaced with GPU kernel)
    min_max_of_img_host(image, &min_host, &max_host, width, height);

    float scale_constant = 255.0f / (max_host - min_host);

    // Subtract minimum pixel value from all pixels (to be replaced with GPU kernel)
    sub_host(image, min_host, width, height);

    // Scale pixel values between 0 and 255 (to be replaced with GPU kernel)
    scale_host(image, scale_constant, width, height);

    // Write image array into a bmp file
    stbi_write_bmp("./out_img.bmp", width, height, 1, image);

    // Deallocate memory
    stbi_image_free(image);

    return 0;
}

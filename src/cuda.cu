#include "cuda.cuh"

#include <cstring>
#include<stdio.h>

#include "helper.h"


///
/// Algorithm storage
///
// Host copy of input image
Image cuda_input_image;
// Host copy of image tiles in each dimension
unsigned int cuda_TILES_X, cuda_TILES_Y;
// Pointer to device buffer for calculating the sum of each tile mosaic, this must be passed to a kernel to be used on device
unsigned long long* d_mosaic_sum;
// Pointer to device buffer for storing the output pixels of each tile, this must be passed to a kernel to be used on device
unsigned char* d_mosaic_value;
// Pointer to device image data buffer, for storing the input image, this must be passed to a kernel to be used on device
unsigned char* d_input_image_data;
// Pointer to device image data buffer, for storing the output image data, this must be passed to a kernel to be used on device
unsigned char* d_output_image_data;
// Pointer to device buffer for the global pixel average sum, this must be passed to a kernel to be used on device
unsigned long long* d_global_pixel_sum;
int width = 0;
unsigned long long* mosaic_sum;

void cuda_begin(const Image* input_image) {
    // These are suggested CUDA memory allocations that match the CPU implementation
    // If you would prefer, you can rewrite this function (and cuda_end()) to suit your preference

    cuda_TILES_X = input_image->width / TILE_SIZE;
    cuda_TILES_Y = input_image->height / TILE_SIZE;

    // Allocate buffer for calculating the sum of each tile mosaic
    CUDA_CALL(cudaMalloc(&d_mosaic_sum, cuda_TILES_X * cuda_TILES_Y * input_image->channels * sizeof(unsigned long long)));

    // Allocate buffer for storing the output pixel value of each tile
    CUDA_CALL(cudaMalloc(&d_mosaic_value, cuda_TILES_X * cuda_TILES_Y * input_image->channels * sizeof(unsigned char)));

    const size_t image_data_size = input_image->width * input_image->height * input_image->channels * sizeof(unsigned char);
    // Allocate copy of input image
    cuda_input_image = *input_image;
    cuda_input_image.data = (unsigned char*)malloc(image_data_size);
    memcpy(cuda_input_image.data, input_image->data, image_data_size);

    // Allocate and fill device buffer for storing input image data
    CUDA_CALL(cudaMalloc(&d_input_image_data, image_data_size));
    CUDA_CALL(cudaMemcpy(d_input_image_data, input_image->data, image_data_size, cudaMemcpyHostToDevice));

    // Allocate device buffer for storing output image data
    CUDA_CALL(cudaMalloc(&d_output_image_data, image_data_size));

    // Allocate and zero buffer for calculation global pixel average
    CUDA_CALL(cudaMalloc(&d_global_pixel_sum, input_image->channels * sizeof(unsigned long long)));

    width = cuda_input_image.width;
    unsigned long long* mosaic_sum;
}


__global__ void Mosaic_sum(int width, int cuda_TILES_X, unsigned char* d_input_image_data, unsigned long long* d_mosaic_sum) {
    // set the (x,y) for each pixel position
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
   
    int t_x = x / blockDim.x;
    int t_y = y / blockDim.y;
    int tile_indx = (t_y * gridDim.x + t_x) * 3;
    int pixel_offset = (y * cuda_TILES_X* TILE_SIZE + x) * 3;

    atomicAdd(&d_mosaic_sum[tile_indx], d_input_image_data[pixel_offset]);
    atomicAdd(&d_mosaic_sum[tile_indx + 1], d_input_image_data[pixel_offset + 1]);
    atomicAdd(&d_mosaic_sum[tile_indx + 2], d_input_image_data[pixel_offset + 2]);
}


__global__ void Mosaic_sum_shared(int width, int cuda_TILES_X, unsigned char* d_input_image_data, unsigned long long* d_mosaic_sum) {
    // set the (x,y) for each pixel position
    __shared__ unsigned int r, g, b;
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    int t_x = x / blockDim.x;
    int t_y = y / blockDim.y;
    int tile_indx = (t_y * gridDim.x + t_x) * 3;
    int pixel_offset = (y * cuda_TILES_X * TILE_SIZE + x) * 3;

    atomicAdd(&d_mosaic_sum[tile_indx], d_input_image_data[pixel_offset]);
    atomicAdd(&d_mosaic_sum[tile_indx + 1], d_input_image_data[pixel_offset + 1]);
    atomicAdd(&d_mosaic_sum[tile_indx + 2], d_input_image_data[pixel_offset + 2]);
}

__global__ void Mosaic_sum_1d(int width, int cuda_TILES_X, unsigned char* d_input_image_data, unsigned long long* d_mosaic_sum) {
    int tile_index = (blockIdx.y * gridDim.x + blockIdx.x)*3;
    int tile_offset = (blockIdx.y * gridDim.x * TILE_SIZE * TILE_SIZE + blockIdx.x * TILE_SIZE) * 3;
    int pixel_offset = (threadIdx.y * width + threadIdx.x) * 3;
    
    for (int ch = 0; ch < 3; ++ch) {
        const unsigned char pixel = d_input_image_data[tile_offset + pixel_offset + ch];
        atomicAdd(&d_mosaic_sum[tile_index + ch], pixel);
    }
}


void cuda_stage1() {
    // Optionally during development call the skip function with the correct inputs to skip this stage
    // skip_tile_sum(input_image, mosaic_sum);
    CUDA_CALL(cudaMemset(d_mosaic_sum, 0, cuda_TILES_X * cuda_TILES_Y * 3 * sizeof(unsigned long long)));
    dim3    blocksPerGrid(cuda_TILES_X, cuda_TILES_Y, 1);
    dim3    threadsPerBlock(TILE_SIZE, TILE_SIZE, 1);

    Mosaic_sum_1d << <blocksPerGrid, threadsPerBlock >> > (width, cuda_TILES_X, d_input_image_data, d_mosaic_sum);
    
    mosaic_sum = (unsigned long long*)malloc(cuda_TILES_X * cuda_TILES_Y * 3 * sizeof(unsigned long long));
    CUDA_CALL(cudaMemcpy(mosaic_sum, d_mosaic_sum, cuda_TILES_X * cuda_TILES_Y * 3 * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
#ifdef VALIDATION
    // TODO: Uncomment and call the validation function with the correct inputs
    // You will need to copy the data back to host before passing to these functions
    // (Ensure that data copy is carried out within the ifdef VALIDATION so that it doesn't affect your benchmark results!)
    
    cudaDeviceSynchronize();
    validate_tile_sum(&cuda_input_image, mosaic_sum);
#endif
}

__global__ void average(unsigned long long* d_whole_image_sum,unsigned char* d_mosaic_value, unsigned long long* d_mosaic_sum) {
      // Only 3 is required for the assignment, but this version hypothetically supports upto 4 channels
    int x = threadIdx.x + blockIdx.x*blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int t = y * blockDim.x * gridDim.x + x;
    for (int ch = 0; ch < 3; ++ch) {
        d_mosaic_value[t*3 + ch] = (unsigned char)(d_mosaic_sum[t*3 + ch] / TILE_PIXELS);  // Integer division is fine 
        atomicAdd(&d_whole_image_sum[ch],d_mosaic_value[t*3 + ch]);
    }

}
void cuda_stage2(unsigned char* output_global_average) {
    // Optionally during development call the skip function with the correct inputs to skip this stage
    // skip_compact_mosaic(cuda_TILES_X, cuda_TILES_Y, mosaic_sum, compact_mosaic, global_pixel_average);
    unsigned long long* whole_image_sum = (unsigned long long*)malloc(4 * sizeof(unsigned long long));
    memset(whole_image_sum, 0, 4 * sizeof(unsigned long long));
    unsigned long long* d_whole_image_sum;
    CUDA_CALL(cudaMalloc(&d_whole_image_sum, 4 * sizeof(unsigned long long)))
    CUDA_CALL(cudaMemset(d_whole_image_sum, 0, 4 * sizeof(unsigned long long)));
    CUDA_CALL(cudaMemset(d_mosaic_value, 0, cuda_TILES_X * cuda_TILES_Y * 3 * sizeof(unsigned char)));
    dim3    blocksPerGrid(cuda_TILES_X / 4, cuda_TILES_Y / 4, 1);
    dim3    threadsPerBlock(4, 4, 1);

    average << <blocksPerGrid, threadsPerBlock >> > (d_whole_image_sum, d_mosaic_value, d_mosaic_sum);

    CUDA_CALL(cudaMemcpy(whole_image_sum, d_whole_image_sum, 4 * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    for (int ch = 0; ch < 3; ++ch) {
        output_global_average[ch] = (unsigned char)(whole_image_sum[ch] / (cuda_TILES_X * cuda_TILES_Y));
    }
   
#ifdef VALIDATION
    // TODO: Uncomment and call the validation functions with the correct inputs
    // You will need to copy the data back to host before passing to these functions
    // (Ensure that data copy is carried out within the ifdef VALIDATION so that it doesn't affect your benchmark results!)
    unsigned char* mosaic_value;
    mosaic_value = (unsigned char*)malloc(cuda_TILES_X * cuda_TILES_Y * 3 * sizeof(unsigned char));
    CUDA_CALL(cudaMemcpy(mosaic_value, d_mosaic_value, cuda_TILES_X * cuda_TILES_Y * 3 * sizeof(unsigned char), cudaMemcpyDeviceToHost));
    validate_compact_mosaic(cuda_TILES_X, cuda_TILES_Y, mosaic_sum, mosaic_value, output_global_average);
#endif    
}

__global__ void broadcast(int width, int cuda_TILES_X, unsigned char* d_output_image_data, unsigned char* d_mosaic_value) {
    int tile_index = (blockIdx.y * gridDim.x + blockIdx.x) * 3;
    int tile_offset = (blockIdx.y * gridDim.x * TILE_SIZE * TILE_SIZE + blockIdx.x * TILE_SIZE) * 3;
    int pixel_offset = (threadIdx.y * width + threadIdx.x) * 3;

    for (int ch = 0; ch < 3; ++ch) {
        const unsigned char pixel = d_output_image_data[tile_offset + pixel_offset + ch];
        memcpy(d_output_image_data + tile_offset + pixel_offset, d_mosaic_value + tile_index, 3);
    }
}

void cuda_stage3() {
    // Optionally during development call the skip function with the correct inputs to skip this stage
    // skip_broadcast(input_image, compact_mosaic, output_image);
    dim3    blocksPerGrid(cuda_TILES_X, cuda_TILES_Y, 1);
    dim3    threadsPerBlock(TILE_SIZE, TILE_SIZE, 1);
    CUDA_CALL(cudaMemset(d_input_image_data, 0, cuda_input_image.width * cuda_input_image.height * cuda_input_image.channels * sizeof(unsigned char)));
    broadcast << <blocksPerGrid, threadsPerBlock >> > (width, cuda_TILES_X, d_output_image_data, d_mosaic_value);

#ifdef VALIDATION
    // TODO: Uncomment and call the validation function with the correct inputs
    // You will need to copy the data back to host before passing to these functions
    // (Ensure that data copy is carried out within the ifdef VALIDATION so that it doesn't affect your benchmark results!)
    // validate_broadcast(&input_image, mosaic_value, &output_image);
#endif    
}
void cuda_end(Image* output_image) {
    // This function matches the provided cuda_begin(), you may change it if desired

    // Store return value
    output_image->width = cuda_input_image.width;
    output_image->height = cuda_input_image.height;
    output_image->channels = cuda_input_image.channels;
    CUDA_CALL(cudaMemcpy(output_image->data, d_output_image_data, output_image->width * output_image->height * output_image->channels * sizeof(unsigned char), cudaMemcpyDeviceToHost));
    // Release allocations
    free(cuda_input_image.data);
    CUDA_CALL(cudaFree(d_mosaic_value));
    CUDA_CALL(cudaFree(d_mosaic_sum));
    CUDA_CALL(cudaFree(d_input_image_data));
    CUDA_CALL(cudaFree(d_output_image_data));
}

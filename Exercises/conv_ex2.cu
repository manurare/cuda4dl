/*
 * Master in Computer Vision - Module 5 - Exercise 2
 *
 * INSTRUCTIONS: if you compile this program you will notice that
 * both png results coincide and are actually the same image.
 * This is because this program is not yet finished.
 *
 * First you need to understand what the code is doing to produce the first image. It might help you understand
 * how cuDNN works.
 *
 * The final goal is that you complete the code to produce succesive convolutions based on the previous result.
 * This can be done by using a technique called "ping-pong buffers", so not new memory needs to be allocated.
 * Here we reused d_input and d_output buffers in GPU, to do so we exchange their pointers so they can be reused
 * to read and write the data respectively through succesive iterations.
 *
 * All you should need is to have a glance at the code and check cuDNN SDK documentation.
 *  Let's go for it!! Only 5 parameters are missing there.
 *
 * GOAL: Find the line saying
 *
 * "UNCOMMENT AND FILL THIS MISSING PARAMETERS FOR THIS CUDNN API CALL"
 *
 *  uncomment and fill the missing parameters in order to produce 10 iterations
 *  where the output data of the convolution is convolved again with the proposed
 *  new filter for several times. After 10 iterations of doing this, the final result
 *  should look very similar (if not equal) to the one shown in the instructions slides.
 *
 *
 *
 * Original Author: Peter Goldsborough
 * FreeImage porting / Iterative Conv. Exercise: Jose A. Iglesias-Guitian <jalley@cvc.uab.es>
 * Convolutions with cuDNN
 * Porting to FreeImage and new cuDNN
 *
 * http://www.goldsborough.me/cuda/ml/cudnn/c++/2017/10/01/14-37-23-convolutions_with_cudnn/
 *
 */

#include <cudnn.h>
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <FreeImage.h>


#define checkCUDNN(expression)                               \
  {                                                          \
    cudnnStatus_t status = (expression);                     \
    if (status != CUDNN_STATUS_SUCCESS) {                    \
      std::cerr << "Error on line " << __LINE__ << ": "      \
                << cudnnGetErrorString(status) << std::endl; \
      std::exit(EXIT_FAILURE);                               \
    }                                                        \
  }

void convert_to_freeimage(FIBITMAP* pBits/*destination*/, const unsigned int& dst_pitch, FIBITMAP* pSource, const int& width, const int& height)
{
    for(int y = 0; y < FreeImage_GetHeight(pBits); y++) {
        BYTE *pPixel = (BYTE *)FreeImage_GetScanLine(pBits, y);
        FIRGBAF *rgba = (FIRGBAF *)FreeImage_GetScanLine(pSource, y);
        for(int x = 0; x < FreeImage_GetWidth(pBits); x++)
        {
            pPixel[0] = 255 * rgba[x].blue;
	    pPixel[1] = 255 * rgba[x].green;
	    pPixel[2] = 255 * rgba[x].red;
            pPixel[3] = 255;
            pPixel += 4;
        }
    }
}

void save_image(const char* output_filename,
                FIBITMAP* buffer,
                int height,
                int width) {
    FIBITMAP* pBitmap = FreeImage_AllocateT(FIT_BITMAP, width, height, 8 * 4/*pixelsize*/);
    unsigned int free_image_pitch = FreeImage_GetPitch(pBitmap);
    std::cout << "Pitch: " << free_image_pitch << std::endl;
    //BYTE *pBits = reinterpret_cast<BYTE*>(FreeImage_GetBits(pBitmap));
    convert_to_freeimage(pBitmap/*destination*/, free_image_pitch, buffer/*source*/, FreeImage_GetWidth(pBitmap), FreeImage_GetHeight(pBitmap));
    FreeImage_Save(FIF_PNG, pBitmap, output_filename, 0);
    std::cerr << "Wrote output to " << output_filename << std::endl;
}

void save_RGB(FIBITMAP* pBits/*destination*/, float* pSource, const int& width, const int& height)
{
    for(uint y = 0; y < FreeImage_GetHeight(pBits); y++) {
        BYTE *pPixel = (BYTE *)FreeImage_GetScanLine(pBits, y);
        for(uint x = 0; x < FreeImage_GetWidth(pBits); x++)
        {
            // input is 3 channels, output is 4 channels for PNG writting
            pPixel[x*4+2] = max(0.f, min(255 * pSource[x*3+2], 255.f));
	    pPixel[x*4+1] = max(0.f, min(255 * pSource[x*3+1], 255.f));
	    pPixel[x*4+0] = max(0.f, min(255 * pSource[x*3+0], 255.f));
            pPixel[x*4+3] = 255; //255 * pSource[x*4+3];
        }
        pSource += (width * 3);
    }
}


void save_tensor_image(const char* output_filename,
                float* buffer,
                int height,
                int width) {
    FIBITMAP* pBitmap = FreeImage_AllocateT(FIT_BITMAP, width, height, 8 * 4/*pixelsize*/);
    //unsigned int free_image_pitch = FreeImage_GetPitch(pBitmap);
    //std::cout << "Pitch: " << free_image_pitch << std::endl;
    //BYTE *pBits = reinterpret_cast<BYTE*>(FreeImage_GetBits(pBitmap));
    save_RGB(pBitmap/*destination*/, buffer/*source*/, width, height);
    FreeImage_Save(FIF_PNG, pBitmap, output_filename, 0);
    std::cerr << "Wrote output to " << output_filename << std::endl;
}


FIBITMAP* load_image(const char* image_path) {
    FREE_IMAGE_FORMAT format = FreeImage_GetFileType(image_path, 0);
    FIBITMAP* image = FreeImage_Load(format, image_path);
    return image;
}

int main(int argc, const char* argv[]) {
  if (argc < 2) {
    std::cerr << "usage: conv <image> [gpu=0] [sigmoid=0]" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  int gpu_id = (argc > 2) ? std::atoi(argv[2]) : 0;
  std::cerr << "GPU: " << gpu_id << std::endl;

  bool with_sigmoid = (argc > 3) ? std::atoi(argv[3]) : 0;
  std::cerr << "With sigmoid: " << std::boolalpha << with_sigmoid << std::endl;

  FIBITMAP* image = load_image(argv[1]);

  int width = FreeImage_GetWidth(image);
  int height = FreeImage_GetHeight(image);

  float* tensor_image = new float[width*height*3];

  for (uint i=0; i < FreeImage_GetHeight(image); i++)
  {
      BYTE *pPixel = (BYTE *)FreeImage_GetScanLine(image, i);
      for (uint j=0; j < FreeImage_GetWidth(image); j++)
      {
          tensor_image[(i*width + j)*3 + 0] = (float)pPixel[j*3+0] / 255.0;
          tensor_image[(i*width + j)*3 + 1] = (float)pPixel[j*3+1] / 255.0;
          tensor_image[(i*width + j)*3 + 2] = (float)pPixel[j*3+2] / 255.0;
      }
  }

  cudaSetDevice(gpu_id);

  cudnnHandle_t cudnn;
  cudnnCreate(&cudnn);

  cudnnTensorDescriptor_t input_descriptor;
  checkCUDNN(cudnnCreateTensorDescriptor(&input_descriptor));
  checkCUDNN(cudnnSetTensor4dDescriptor(input_descriptor,
                                        /*format=*/CUDNN_TENSOR_NHWC,
                                        /*dataType=*/CUDNN_DATA_FLOAT,
                                        /*batch_size=*/1,
                                        /*channels=*/3,
                                        /*image_height=*/FreeImage_GetHeight(image),
                                        /*image_width=*/FreeImage_GetWidth(image)));

  cudnnFilterDescriptor_t kernel_descriptor;
  checkCUDNN(cudnnCreateFilterDescriptor(&kernel_descriptor));
  checkCUDNN(cudnnSetFilter4dDescriptor(kernel_descriptor,
                                        /*dataType=*/CUDNN_DATA_FLOAT,
                                        /*format=*/CUDNN_TENSOR_NCHW,
                                        /*out_channels=*/3,
                                        /*in_channels=*/3,
                                        /*kernel_height=*/3,
                                        /*kernel_width=*/3));

  cudnnConvolutionDescriptor_t convolution_descriptor;
  checkCUDNN(cudnnCreateConvolutionDescriptor(&convolution_descriptor));
#if CUDNN_MAJOR < 6
  std::cout << "CUDNN < 6" << std::endl;
  checkCUDNN(cudnnSetConvolution2dDescriptor(convolution_descriptor,
                                             /*pad_height=*/1,
                                             /*pad_width=*/1,
                                             /*vertical_stride=*/1,
                                             /*horizontal_stride=*/1,
                                             /*dilation_height=*/1,
                                             /*dilation_width=*/1,
                                             /*mode=*//*CUDNN_CROSS_CORRELATION*/CUDNN_CONVOLUTION));
#else
  checkCUDNN(cudnnSetConvolution2dDescriptor(convolution_descriptor,
                                             /*pad_height=*/1,
                                             /*pad_width=*/1,
                                             /*vertical_stride=*/1,
                                             /*horizontal_stride=*/1,
                                             /*dilation_height=*/1,
                                             /*dilation_width=*/1,
                                             /*mode=*/CUDNN_CROSS_CORRELATION,
                                             /*computeType=*/CUDNN_DATA_FLOAT));
#endif
  int batch_size{0}, channels{0}; //, height{0}, width{0};
  checkCUDNN(cudnnGetConvolution2dForwardOutputDim(convolution_descriptor,
                                                   input_descriptor,
                                                   kernel_descriptor,
                                                   &batch_size,
                                                   &channels,
                                                   &height,
                                                   &width));

  std::cerr << "Output Image: " << height << " x " << width << " x " << channels << " batch: " << batch_size
            << std::endl;

  cudnnTensorDescriptor_t output_descriptor;
  checkCUDNN(cudnnCreateTensorDescriptor(&output_descriptor));
  checkCUDNN(cudnnSetTensor4dDescriptor(output_descriptor,
                                        /*format=*/CUDNN_TENSOR_NHWC,
                                        /*dataType=*/CUDNN_DATA_FLOAT,
                                        /*batch_size=*/1,
                                        /*channels=*/3,
                                        /*image_height=*/FreeImage_GetHeight(image),
                                        /*image_width=*/FreeImage_GetWidth(image)));

  cudnnConvolutionFwdAlgo_t convolution_algorithm;
  checkCUDNN(
      cudnnGetConvolutionForwardAlgorithm(cudnn,
                                          input_descriptor,
                                          kernel_descriptor,
                                          convolution_descriptor,
                                          output_descriptor,
                                          CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
                                          /*memoryLimitInBytes=*/0,
                                          &convolution_algorithm));

  size_t workspace_bytes{0};
  checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnn,
                                                     input_descriptor,
                                                     kernel_descriptor,
                                                     convolution_descriptor,
                                                     output_descriptor,
                                                     convolution_algorithm,
                                                     &workspace_bytes));
  std::cerr << "Workspace size: " << (workspace_bytes / 1048576.0) << "MB"
            << std::endl;
  assert(workspace_bytes > 0);

  std::cerr << "Convolution Algorithm: " << convolution_algorithm << " type" << std::endl;

  void* d_workspace{/*nullptr*/0};
  cudaMalloc(&d_workspace, workspace_bytes);

  int image_bytes = batch_size * channels * height * width * sizeof(float);

  float* d_input{/*nullptr*/0};
  cudaMalloc(&d_input, image_bytes);
  //cudaMemcpy(d_input, image.ptr<float>(0), image_bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_input, &(tensor_image[0]), image_bytes, cudaMemcpyHostToDevice);

  float* d_output{/*nullptr*/0};
  cudaMalloc(&d_output, image_bytes);
  cudaMemset(d_output, 0, image_bytes);

  // clang-format off
  const float kernel_template[3][3] = {
    {1, 1, 1},
    {1, -8, 1},
    {1, 1, 1}
  };
  // clang-format on

  float h_kernel[3][3][3][3];
  for (int kernel = 0; kernel < 3; ++kernel) {
    for (int channel = 0; channel < 3; ++channel) {
      for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 3; ++column) {
            h_kernel[kernel][channel][row][column] = kernel_template[row][column];
        }
      }
    }
  }

  float* d_kernel{/*nullptr*/0};
  cudaMalloc(&d_kernel, sizeof(h_kernel));
  cudaMemcpy(d_kernel, h_kernel, sizeof(h_kernel), cudaMemcpyHostToDevice);

  float alpha = 1.0f, beta = 0.0f;

  checkCUDNN(cudnnConvolutionForward(cudnn,
                                     &alpha,
                                     input_descriptor,
                                     d_input,
                                     kernel_descriptor,
                                     d_kernel,
                                     convolution_descriptor,
                                     convolution_algorithm,
                                     d_workspace,
                                     workspace_bytes,
                                     &beta,
                                     output_descriptor,
                                     d_output));

  if (with_sigmoid) {
    cudnnActivationDescriptor_t activation_descriptor;
    checkCUDNN(cudnnCreateActivationDescriptor(&activation_descriptor));
    checkCUDNN(cudnnSetActivationDescriptor(activation_descriptor,
                                            CUDNN_ACTIVATION_SIGMOID,
                                            CUDNN_PROPAGATE_NAN,
                                            /*relu_coef=*/0));
    checkCUDNN(cudnnActivationForward(cudnn,
                                      activation_descriptor,
                                      &alpha,
                                      output_descriptor,
                                      d_output,
                                      &beta,
                                      output_descriptor,
                                      d_output));
    cudnnDestroyActivationDescriptor(activation_descriptor);
  }

  cudaMemcpy(&(tensor_image[0]), d_output, image_bytes, cudaMemcpyDeviceToHost);

  save_tensor_image("cudnn-output.png", tensor_image, height, width);

  // MAKE A SECOND CONVOLUTION
  float* d_tmp{/*nullptr*/0};
  cudaMalloc(&d_tmp, image_bytes);

  // NEW FILTER TEMPLATE
  const float new_kernel_template[3][3] = {
      {1, 2, 1},
      {2, 4, 2},
      {1, 2, 1}
  };
  // BUILD NEW KERNEL FILTER
  float new_h_kernel[3][3][3][3];
  for (int kernel = 0; kernel < 3; ++kernel) {
      for (int channel = 0; channel < 3; ++channel) {
          for (int row = 0; row < 3; ++row) {
              for (int column = 0; column < 3; ++column) {
                  new_h_kernel[kernel][channel][row][column] = new_kernel_template[row][column];
              }
          }
      }
  }

  float* d_new_kernel{/*nullptr*/0};
  cudaMalloc(&d_new_kernel, sizeof(new_h_kernel));
  cudaMemcpy(d_new_kernel, new_h_kernel, sizeof(new_h_kernel), cudaMemcpyHostToDevice);

  for (int i=0; i< 10; i++)
  {
      // SWAP VARIABLES
      d_tmp = d_input;
      d_input = d_output;
      d_output = d_tmp;

      // CODING EXERCISE STARTS HERE:

      //////////////////////////////////////////////////////////////////////////////////
      // UNCOMMENT AND FILL THIS MISSING PARAMETERS FOR THIS CUDNN API CALL:
      //////////////////////////////////////////////////////////////////////////////////

      
      checkCUDNN(cudnnConvolutionForward(cudnn,
                                     &alpha,
                                     input_descriptor,
                                     d_input,
                                     kernel_descriptor,
                                     d_new_kernel,
                                     convolution_descriptor,
                                     convolution_algorithm,
                                     d_workspace,
                                     workspace_bytes,
                                     &beta,
                                     output_descriptor,
                                     d_output));
      

      // CODING EXERCISE SHOULD FINISH HERE.

      cudaMemcpy(&(tensor_image[0]), d_output, image_bytes, cudaMemcpyDeviceToHost);
  }

  save_tensor_image("cudnn-your-output.png", tensor_image, height, width);

  // Free resources
  delete[] tensor_image;
  FreeImage_Unload(image);


  cudaFree(d_kernel);
  cudaFree(d_new_kernel);
  cudaFree(d_input);
  cudaFree(d_output);
  cudaFree(d_tmp);
  cudaFree(d_workspace);

  cudnnDestroyTensorDescriptor(input_descriptor);
  cudnnDestroyTensorDescriptor(output_descriptor);
  cudnnDestroyFilterDescriptor(kernel_descriptor);
  cudnnDestroyConvolutionDescriptor(convolution_descriptor);

  cudnnDestroy(cudnn);
}

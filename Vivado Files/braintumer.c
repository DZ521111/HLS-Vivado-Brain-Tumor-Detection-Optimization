#include <math.h> 
#include <string.h>
#include "k2c_include.h"
#include "k2c_tensor_include.h"


//#include "k2c_activations.h"
//#include "k2c_convolution_layers.h"
//#include "k2c_core_layers.h"
//#include "k2c_embedding_layers.h"
//#include "k2c_helper_functions.h"
//#include "k2c_include.h"
//#include "k2c_merge_layers.h"
//#include "k2c_normalization_layers.h"
//#include "k2c_pooling_layers.h"
//#include "k2c_recurrent_layers.h"
//#include "k2c_tensor_include.h"


// K2C Variables
k2c_tensor conv2d_1_output;
k2c_tensor conv2d_31_output;
k2c_tensor conv2d_1_padded_input;
k2c_tensor conv2d_31_padded_input;
k2c_tensor conv2d_1_kernel;
k2c_tensor conv2d_31_kernel;
k2c_tensor conv2d_1_bias;
k2c_tensor max_pooling2d_1_output;
k2c_tensor conv2d_2_output;
k2c_tensor conv2d_2_padded_input;
k2c_tensor conv2d_2_kernel;
k2c_tensor conv2d_2_bias;
k2c_tensor max_pooling2d_2_output;
k2c_tensor conv2d_3_output;
k2c_tensor conv2d_3_padded_input;
k2c_tensor conv2d_3_kernel;
k2c_tensor conv2d_3_bias;
k2c_tensor max_pooling2d_3_output;
k2c_tensor flatten_1_output;
k2c_tensor dense_1_output;
k2c_tensor dense_1_kernel;
k2c_tensor dense_1_bias;
k2c_tensor dense_2_output;
k2c_tensor dense_2_kernel;
k2c_tensor dense_2_bias;
k2c_tensor dense_3_kernel;
k2c_tensor dense_3_bias;

k2c_tensor dense_32_bias;
k2c_tensor dense_32_kernel;
k2c_tensor dense_31_bias;
k2c_tensor dense_31_kernel;
k2c_tensor dense_31_output;
k2c_tensor dense_30_bias;
k2c_tensor dense_30_kernel;
k2c_tensor dense_30_output;
k2c_tensor dense_29_bias;
k2c_tensor dense_29_kernel;
k2c_tensor dense_29_output;
k2c_tensor flatten_8_output;
k2c_tensor conv2d_34_bias;
k2c_tensor conv2d_34_kernel;
k2c_tensor conv2d_34_padded_input;
k2c_tensor conv2d_34_output;
k2c_tensor conv2d_33_bias;
k2c_tensor conv2d_33_kernel;
k2c_tensor conv2d_33_padded_input;
k2c_tensor conv2d_33_output;
k2c_tensor max_pooling2d_19_output;
k2c_tensor conv2d_32_bias;
k2c_tensor conv2d_32_kernel;
k2c_tensor conv2d_32_padded_input;
k2c_tensor conv2d_32_output;
k2c_tensor max_pooling2d_18_output;
k2c_tensor conv2d_31_bias;


void k2c_relu_func(float * x, const size_t size) {

    for (size_t i=0; i < size; ++i) {
		#pragma HLS unroll factor=16
        if (x[i] <= 0.0f) {
            x[i] = 0.0f;
        }
    }
}
// k2c_activationType * k2c_relu = k2c_relu_func;

 void k2c_softmax_func(float* x, const size_t size) {

    float xmax = x[0];
    float sum = 0;
    size_t i = 0;
    for (i=0; i < size; ++i) {
		#pragma HLS unroll factor = 16
        if (x[i]>xmax) {
            xmax = x[i];
        }
    }

    for (i=0; i < size; ++i) {
	#pragma HLS unroll factor = 16
        x[i] = expf(x[i]-xmax);
    }

    for (i=0; i < size; ++i) {
	#pragma HLS unroll factor = 16
        sum += x[i];
    }

    sum = 1.0f/sum;
    for (i=0; i < size; ++i) {
	#pragma HLS unroll factor = 16
        x[i] = x[i]*sum;
    }
}


 void k2c_matmul(float * C, const float * A, const float * B, const size_t outrows,
                 const size_t outcols, const size_t innerdim) {

     // make sure output is empty
 //    memset(C, 0, outrows*outcols*sizeof(C[0]));

 	for (size_t row = 0; row < outrows; ++row) {
 	        // Iterate over each column of C
 	        for (size_t col = 0; col < outcols; ++col) {
 	            // Set each element of C to zero
				//#pragma HLS pipeline
 	            C[row * outcols + col] = 0.0f;
 	        }
 	    }


     for (size_t i = 0 ; i < outrows; ++i) {
         const size_t outrowidx = i*outcols;
         const size_t inneridx = i*innerdim;
         for (size_t k = 0; k < innerdim; ++k) {
             for (size_t j = 0;  j < outcols; ++j) {
				#pragma HLS pipeline
                 C[outrowidx+j] += A[inneridx+k] * B[k*outcols+j];
             }
         }
     }
 }



 void k2c_idx2sub(const size_t idx, size_t * sub, const size_t * shape, const size_t ndim) {

     size_t idx2 = idx;
     for (int i=ndim-1; i>=0; --i) {
		#pragma HLS unroll factor = 16
         sub[i] = idx2%shape[i];
         idx2 /= shape[i];
     }
 }
 size_t k2c_sub2idx(const size_t * sub, const size_t * shape, const size_t ndim) {

     size_t idx = 0;
     size_t temp = 0;
     for (size_t i=0; i<ndim; ++i) {
         temp = sub[i];
         for (size_t j=ndim-1; j>i; --j) {
			#pragma HLS pipeline
             temp *= shape[j];
         }
         idx += temp;
     }
     return idx;
 }


 void k2c_pad2d(k2c_tensor* output, const k2c_tensor* input, const float fill,
                const size_t * pad) {

     const size_t in_height = input->shape[0];
     const size_t in_width = input->shape[1];
     const size_t in_channels = input->shape[2];
     const size_t pad_top = pad[0];
     const size_t pad_left = pad[2];
     const size_t pad_right = pad[3];
     size_t i = 0;

     // set output array to fill value
     if (fabs(fill) < 1e-6) {
         // fill is ~zero, use memset
         // memset(output->array,0,output->numel*sizeof(output->array[0]));
         for ( i = 0; i < output->numel; i++) {
			#pragma HLS unroll factor = 16
             output->array[i] = 0;
         }
     }
     else {
         for(i=0; i<output->numel; ++i) {
			#pragma HLS unroll factor=16
             output->array[i] = fill;
         }
     }k2c_tensor conv2d_31_padded_input;
     // memcpy the old array in the middle
     size_t offset = in_channels*(pad_left+pad_right+in_width)*pad_top +
                     in_channels*pad_left;
     const size_t num = in_channels*in_width;
     const size_t step = num+in_channels*(pad_left+pad_right);
     for (i=0; i<in_height; ++i) {
         // memcpy(&output->array[offset],
         //        &input->array[i*num],
         //        num*sizeof(input->array[0]));
         for (size_t ji = 0; ji < num; ji++) {
			#pragma HLS unroll factor = 16
             output->array[offset + ji] = input->array[i * num + ji];
         }
         offset += step;
     }
 }


 void k2c_dot(k2c_tensor* C, const k2c_tensor* A, const k2c_tensor* B, const size_t * axesA,
              const size_t * axesB, const size_t naxes, const int normalize, float * fwork) {
	#pragma HLS dataflow
     size_t permA[K2C_MAX_NDIM];
     size_t permB[K2C_MAX_NDIM];
     size_t prod_axesA = 1;
     size_t prod_axesB = 1;
     size_t free_axesA, free_axesB;
     size_t freeA[K2C_MAX_NDIM];
     size_t freeB[K2C_MAX_NDIM];
     size_t count;
     int isin;
     size_t newshpA[K2C_MAX_NDIM];
     size_t newshpB[K2C_MAX_NDIM];
     const size_t ndimA = A->ndim;
     const size_t ndimB = B->ndim;
     float *reshapeA = &fwork[0];   // temp working storage
     float *reshapeB = &fwork[A->numel];
     size_t Asub[K2C_MAX_NDIM];
     size_t Bsub[K2C_MAX_NDIM];
     size_t i,j;
     // find which axes are free (ie, not being summed over)
     count=0;
     for (i=0; i<ndimA; ++i) {
         isin = 0;
         for (j=0; j<naxes; ++j) {
#pragma HLS unroll factor = 16
             if (i==axesA[j]) {
                 isin=1;
             }
         }
         if (!isin) {
             freeA[count] = i;
             ++count;
         }
     }
     count=0;
     for (i=0; i<ndimB; ++i) {
         isin = 0;
         for (j=0; j<naxes; ++j) {
#pragma HLS unroll factor = 16
             if (i==axesB[j]) {
                 isin=1;
             }
         }
         if (!isin) {
             freeB[count] = i;
             ++count;
         }
     }

     // number of elements in inner dimension
     for (i=0; i < naxes; ++i) {
#pragma HLS unroll factor = 16
         prod_axesA *= A->shape[axesA[i]];
     }
     for (i=0; i < naxes; ++i) {
#pragma HLS unroll factor = 16
         prod_axesB *= B->shape[axesB[i]];
     }
     // number of elements in free dimension
     free_axesA = A->numel/prod_axesA;
     free_axesB = B->numel/prod_axesB;
     // find permutation of axes to get into matmul shape
     for (i=0; i<ndimA-naxes; ++i) {
#pragma HLS unroll factor = 16
         permA[i] = freeA[i];
     }
     for (i=ndimA-naxes, j=0; i<ndimA; ++i, ++j) {
#pragma HLS unroll factor = 16
         permA[i] = axesA[j];
     }
     for (i=0; i<naxes; ++i) {
#pragma HLS unroll factor=16
         permB[i] = axesB[i];
     }
     for (i=naxes, j=0; i<ndimB; ++i, ++j) {
#pragma HLS unroll factor = 16
         permB[i] = freeB[j];
     }



     for (i=0; i<ndimA; ++i) {
#pragma HLS unroll factor = 16
         newshpA[i] = A->shape[permA[i]];
     }
     for (i=0; i<ndimB; ++i) {
#pragma HLS unroll factor = 16
         newshpB[i] = B->shape[permB[i]];
     }

     // reshape arrays
     for (i=0; i<A->numel; ++i) {
         k2c_idx2sub(i,Asub,A->shape,ndimA);
         for (j=0; j<ndimA; ++j) {
#pragma HLS pipeline
             Bsub[j] = Asub[permA[j]];
         }
         size_t bidx = k2c_sub2idx(Bsub,newshpA,ndimA);
         reshapeA[bidx] = A->array[i];
     }

     for (i=0; i<B->numel; ++i) {
         k2c_idx2sub(i,Bsub,B->shape,ndimB);
         for (j=0; j<ndimB; ++j) {
#pragma HLS pipeline
             Asub[j] = Bsub[permB[j]];
         }
         size_t bidx = k2c_sub2idx(Asub,newshpB,ndimB);
         reshapeB[bidx] = B->array[i];
     }


     if (normalize) {

         float sum;
         float inorm;
         for (i=0; i<free_axesA; ++i) {
             sum = 0;
             for (j=0; j<prod_axesA; ++j) {
#pragma HLS pipeline
                 sum += reshapeA[i*prod_axesA + j]*reshapeA[i*prod_axesA + j];
             }
             inorm = 1.0f/sqrtf(sum);
             for (j=0; j<prod_axesA; ++j) {
#pragma HLS pipeline
                 reshapeA[i*prod_axesA + j] *= inorm;
             }
         }
         for (i=0; i<free_axesB; ++i) {
             sum = 0;
             for (j=0; j<prod_axesB; ++j) {
#pragma HLS pipeline
                 sum += reshapeB[i + free_axesB*j]*reshapeB[i + free_axesB*j];
             }
             inorm = 1.0f/sqrtf(sum);
             for (j=0; j<prod_axesB; ++j) {
#pragma HLS pipeline
                 reshapeB[i + free_axesB*j] *= inorm;
             }
         }
     }

     k2c_matmul(C->array, reshapeA, reshapeB, free_axesA,
                free_axesB, prod_axesA);
 }



 void k2c_bias_add(k2c_tensor* A, const k2c_tensor* b) {

     for (size_t i=0; i<A->numel; i+=b->numel) {
         for (size_t j=0; j<b->numel; ++j) {
#pragma HLS pipeline II = 1
             A->array[i+j] += b->array[j];
         }
     }
 }
 void k2c_affine_matmul(float * C, const float * A, const float * B, const float * d,
                        const size_t outrows,const size_t outcols, const size_t innerdim) {

     // make sure output is empty

     // Iterate over each row of C
     for (size_t row = 0; row < outrows; ++row) {
         // Iterate over each column of C
         for (size_t col = 0; col < outcols; ++col) {
#pragma HLS pipeline
             // Set each element of C to zero
             C[row * outcols + col] = 0.0f;
         }
     }

     // memset(C, 0, outrows*outcols*sizeof(C[0]));

     for (size_t i = 0 ; i < outrows; ++i) {
         const size_t outrowidx = i*outcols;
         const size_t inneridx = i*innerdim;
         for (size_t j = 0;  j < outcols; ++j) {
             for (size_t k = 0; k < innerdim; ++k) {
#pragma HLS pipeline
                 C[outrowidx+j] += A[inneridx+k] * B[k*outcols+j];
             }
             C[outrowidx+j] += d[j];
         }
     }
 }
 void k2c_conv2d(k2c_tensor* output,  k2c_tensor* input, const k2c_tensor* kernel,
                 const k2c_tensor* bias, const size_t*  stride, const size_t*  dilation
                 ) {

     // memset(output->array,0,output->numel*sizeof(output->array[0]));
 	 size_t i;
 	  for ( i = 0; i < output->numel; ++i) {
#pragma HLS unroll factor = 16
         output->array[i] = 0.0f; // Set each element to zero
     }

    // Define loop variables
     size_t x0, x1, z0, z1, k, q;

     // Extract tensor dimensions
     const size_t out_rows = output->shape[0];
     const size_t out_cols = output->shape[1];
     const size_t out_channels = output->shape[2];
     const size_t in_channels = input->shape[2];
     const size_t kernel_height = kernel->shape[0];
     const size_t kernel_width = kernel->shape[1];

     // Perform convolution operation
     for (x0 = 0; x0 < out_rows; ++x0) {
         for (x1 = 0; x1 < out_cols; ++x1) {
             for (z0 = 0; z0 < kernel_height; ++z0) {

                 for (z1 = 0; z1 < kernel_width; ++z1) {
                     for (q = 0; q < in_channels; ++q) {
                         for (k = 0; k < out_channels; ++k) {
#pragma HLS pipeline
                             // Update output using convolution operation
                             size_t output_index = x0 * (out_cols * out_channels) + x1 * out_channels + k;
                             size_t kernel_index = z0 * (kernel_width * out_channels * in_channels)
                                                 + z1 * (out_channels * in_channels)
                                                 + q * out_channels + k;
                             size_t input_index = (x0 * stride[0] + dilation[0] * z0) * (in_channels * out_cols)
                                                + (x1 * stride[1] + dilation[1] * z1) * in_channels + q;

                             output->array[output_index] +=
                                 kernel->array[kernel_index] * input->array[input_index];
                         }
                     }
                 }
             }
         }
     }

     k2c_bias_add(output,bias);
     k2c_relu_func(output->array,output->numel);
 }
 /*void k2c_maxpool2d(k2c_tensor* output, const k2c_tensor* input, const size_t * pool_size,
                    const size_t * stride) {


     const size_t channels = input->shape[2];
     // i,j,l output indices
     /// i, k, m input indices
     for (size_t i=0; i< channels; ++i) {
         for (size_t j=0, k=0; j<output->shape[1]*channels;
                 j+=channels, k+=channels*stride[1]) {
             for (size_t l=0, m=0; l<output->numel; l+=channels*output->shape[1],
                     m+=channels*input->shape[1]*stride[0]) {
                 output->array[l+j+i] = input->array[m+k+i];
                 for (size_t n=0; n<pool_size[1]*channels; n+=channels) {

                     for (size_t p=0; p<pool_size[0]*channels*input->shape[1];
                             p+=channels*input->shape[1]) {
#pragma HLS pipeline
                         if (output->array[l+j+i] < input->array[m+k+i+n+p]) {
                             output->array[l+j+i] = input->array[m+k+i+n+p];
                         }
                     }
                 }
             }
         }
     }
 }*/

 void k2c_maxpool2d(k2c_tensor* output, const k2c_tensor* input, const size_t * pool_size,
                     const size_t * stride) {

      const size_t channels = input->shape[2];
      //#pragma HLS array_partition variable=input->array type=complete
      //#pragma HLS array_partition variable=output->array type=complete

      for (size_t i=0; i< channels; ++i) {
          #pragma HLS unroll
          for (size_t j=0, k=0; j<output->shape[1]*channels;
                  j+=channels, k+=channels*stride[1]) {
              #pragma HLS pipeline II=1
              for (size_t l=0, m=0; l<output->numel; l+=channels*output->shape[1],
                      m+=channels*input->shape[1]*stride[0]) {
                  output->array[l+j+i] = input->array[m+k+i];
                  for (size_t n=0; n<pool_size[1]*channels; n+=channels) {
                      #pragma HLS unroll factor=4
                      for (size_t p=0; p<pool_size[0]*channels*input->shape[1];
                              p+=channels*input->shape[1]) {
                          #pragma HLS pipeline II=1
                          if (output->array[l+j+i] < input->array[m+k+i+n+p]) {
                              output->array[l+j+i] = input->array[m+k+i+n+p];
                          }
                      }
                  }
              }
          }
      }
  }

 void k2c_dense(k2c_tensor* output, const k2c_tensor* input, const k2c_tensor* kernel,
                const k2c_tensor* bias,int flag, float * fwork) {

     if (input->ndim <=2) {
         size_t outrows;

         if (input->ndim>1) {
             outrows = input->shape[0];
         }
         else {
             outrows = 1;
         }
         const size_t outcols = kernel->shape[1];
         const size_t innerdim = kernel->shape[0];
         const size_t outsize = outrows*outcols;
         //k2c_affine_matmul(output->array,input->array,kernel->array,bias->array,
                           //outrows,outcols,innerdim);
          if(flag==0){
             k2c_softmax_func(output->array, output->numel);
         }
         if(flag==1){
          k2c_relu_func(output->array, output->numel);

         }
     }
     else {
         const size_t axesA[1] = {input->ndim-1};
         const size_t axesB[1] = {0};
         const size_t naxes = 1;
         const int normalize = 0;

         k2c_dot(output, input, kernel, axesA, axesB, naxes, normalize, fwork);
         k2c_bias_add(output, bias);
         if(flag==0){
             k2c_softmax_func(output->array, output->numel);
         }
         if(flag==1){
          k2c_relu_func(output->array, output->numel);

         }
     }
 }

 /*void k2c_flatten(k2c_tensor *output, const k2c_tensor* input) {

     // memcpy(output->array, input->array, input->numel*sizeof(input->array[0]));

       for (size_t j = 0; j < input->numel && j < output->numel; ++j) {
#pragma HLS unroll factor = 16
         output->array[j] = input->array[j]; // Copy each element
     }

     for (size_t i=0; i<input->ndim; ++i) {
#pragma HLS unroll factor = 16
         output->shape[i] = 1;
     }
     output->shape[0] = input->numel;
     output->numel = input->numel;
     output->ndim = 1;
 }*/

 void k2c_flatten(k2c_tensor *output, const k2c_tensor* input) {
      // Attempt to use memcpy if possible for better optimization by the HLS tool.
      // However, direct loop-based copying allows more fine-grained control and optimization with HLS pragmas.
      #pragma HLS INLINE off
      //#pragma HLS ARRAY_PARTITION variable=output->array complete
      //#pragma HLS ARRAY_PARTITION variable=input->array complete

      for (size_t j = 0; j < input->numel && j < output->numel; ++j) {
          #pragma HLS UNROLL factor=16
          output->array[j] = input->array[j]; // Copy each element
      }

      #pragma HLS ARRAY_PARTITION variable=output->shape complete
      for (size_t i=0; i<input->ndim; ++i) {
          #pragma HLS UNROLL factor=16
          output->shape[i] = 1;
      }

      output->shape[0] = input->numel;
      output->numel = input->numel;
      output->ndim = 1;

      // Ensure that loops that can be completely unrolled do not consume additional loop control logic.
      #pragma HLS latency min=0 max=1
 }




 // -------------------------------------------------------------------
 // end_of_func


void braintumer(k2c_tensor* input_9_input, k2c_tensor* dense_32_output) { 

size_t conv2d_31_stride[2] = {2,2}; 
size_t conv2d_31_dilation[2] = {1,1}; 
float conv2d_31_output_array[15000] = {0};
size_t i = 0;
//k2c_tensor conv2d_31_output = {&conv2d_31_output_array[0],3,15000,{50,50, 6, 1, 1}};
conv2d_31_output.ndim=3;
conv2d_31_output.numel=15000;
conv2d_31_output.shape[0]=50;
conv2d_31_output.shape[1]=50;
conv2d_31_output.shape[2]=6;
conv2d_31_output.shape[3]=1;
conv2d_31_output.shape[4]=1;
for(i=0;i<15000;i++){
#pragma HLS unroll factor = 128
	conv2d_31_output.array[i] = conv2d_31_output_array[i];
}



float conv2d_31_padded_input_array[10404] = {0}; 
//k2c_tensor conv2d_31_padded_input = {&conv2d_31_padded_input_array[0],3,10404,{102,102,  1,  1,  1}};
conv2d_31_padded_input.ndim=3;
conv2d_31_padded_input.numel=10404;
conv2d_31_padded_input.shape[0]=102;
conv2d_31_padded_input.shape[1]=102;
conv2d_31_padded_input.shape[2]=1;
conv2d_31_padded_input.shape[3]=1;
conv2d_31_padded_input.shape[4]=1;
for(i=0;i<10404;i++){
#pragma HLS unroll factor = 128
//#pragma HLS unroll
	conv2d_31_padded_input.array[i] = conv2d_31_padded_input_array[i];
}


size_t conv2d_31_pad[4] = {1,1,1,1}; 
float conv2d_31_fill = 0.0f; 
float conv2d_31_kernel_array[54] = {
+2.41654307e-01f,-1.37430474e-01f,+1.68644160e-01f,+2.00398818e-01f,+3.36053580e-01f,
-2.97189832e-01f,+1.17284410e-01f,+2.96412766e-01f,-1.13714866e-01f,+2.61324525e-01f,
-3.54669750e-01f,+6.74520954e-02f,-1.57676458e-01f,+3.84038061e-01f,+3.55438232e-01f,
+3.18387091e-01f,-3.16810429e-01f,+1.79685190e-01f,-1.71955407e-01f,+2.17519075e-01f,
+3.82954657e-01f,-4.22581941e-01f,-1.02148727e-01f,-2.70742238e-01f,+2.26014361e-01f,
+2.62729496e-01f,+3.12660784e-01f,-3.29030901e-02f,-1.47171065e-01f,-1.81779087e-01f,
+2.31544971e-01f,-1.52112141e-01f,+7.18344450e-02f,+2.34869093e-01f,+2.49691933e-01f,
+1.75722703e-01f,+1.92320392e-01f,-1.97049305e-01f,-1.41355693e-01f,-1.48304060e-01f,
+1.93268463e-01f,-2.17513621e-01f,-4.20560576e-02f,+3.26804519e-01f,+3.71662468e-01f,
-5.67589421e-04f,+3.99020225e-01f,-4.02405053e-01f,+2.19776303e-01f,+3.27187240e-01f,
+4.06550288e-01f,+1.60747662e-01f,+2.19272479e-01f,+7.72708058e-02f,}; 

// k2c_tensor conv2d_31_kernel = {&conv2d_31_kernel_array[0],4,54,{3,3,1,6,1}};
conv2d_31_kernel.ndim=4;
conv2d_31_kernel.numel=54;
conv2d_31_kernel.shape[0]=3;
conv2d_31_kernel.shape[1]=3;
conv2d_31_kernel.shape[2]=1;
conv2d_31_kernel.shape[3]=6;
conv2d_31_kernel.shape[4]=1;
for(i=0;i<54;i++){
#pragma HLS unroll factor = 16
	conv2d_31_kernel.array[i] = conv2d_31_kernel_array[i];
}


float conv2d_31_bias_array[6] = {
-3.63605581e-02f,-1.71340909e-02f,-1.11229844e-01f,+2.44899187e-02f,+1.03014829e-02f,
+7.81366676e-02f,};

//k2c_tensor conv2d_31_bias = {&conv2d_31_bias_array[0],1,6,{6,1,1,1,1}};
conv2d_31_bias.ndim=1;
conv2d_31_bias.numel=6;
conv2d_31_bias.shape[0]=6;
conv2d_31_bias.shape[1]=1;
conv2d_31_bias.shape[2]=1;
conv2d_31_bias.shape[3]=1;
conv2d_31_bias.shape[4]=1;
for(i=0;i<6;i++){
#pragma HLS unroll factor = 16
	conv2d_31_bias.array[i] = conv2d_31_bias_array[i];
}

 
size_t max_pooling2d_18_stride[2] = {2,2}; 
size_t max_pooling2d_18_pool_size[2] = {2,2}; 
float max_pooling2d_18_output_array[3750] = {0}; 
//k2c_tensor max_pooling2d_18_output = {&max_pooling2d_18_output_array[0],3,3750,{25,25, 6, 1, 1}};
max_pooling2d_18_output.ndim=3;
max_pooling2d_18_output.numel=3750;
max_pooling2d_18_output.shape[0]=25;
max_pooling2d_18_output.shape[1]=25;
max_pooling2d_18_output.shape[2]=6;
max_pooling2d_18_output.shape[3]=1;
max_pooling2d_18_output.shape[4]=1;
for(i=0;i<3750;i++){
#pragma HLS unroll
//#pragma HLS unroll factor=16
	max_pooling2d_18_output.array[i] = max_pooling2d_18_output_array[i];
}


size_t conv2d_32_stride[2] = {2,2}; 
size_t conv2d_32_dilation[2] = {1,1}; 
float conv2d_32_output_array[1014] = {0}; 
//k2c_tensor conv2d_32_output = {&conv2d_32_output_array[0],3,1014,{13,13, 6, 1, 1}};
conv2d_32_output.ndim=3;
conv2d_32_output.numel=1014;
conv2d_32_output.shape[0]=13;
conv2d_32_output.shape[1]=13;
conv2d_32_output.shape[2]=6;
conv2d_32_output.shape[3]=1;
conv2d_32_output.shape[4]=1;
for(i=0;i<1014;i++){
//#pragma HLS unroll factor = 16
#pragma HLS unroll
	conv2d_32_output.array[i] = conv2d_32_output_array[i];
}

float conv2d_32_padded_input_array[4374] = {0}; 
//k2c_tensor conv2d_32_padded_input = {&conv2d_32_padded_input_array[0],3,4374,{27,27, 6, 1, 1}};
conv2d_32_padded_input.ndim=3;
conv2d_32_padded_input.numel=4374;
conv2d_32_padded_input.shape[0]=27;
conv2d_32_padded_input.shape[1]=27;
conv2d_32_padded_input.shape[2]=6;
conv2d_32_padded_input.shape[3]=1;
conv2d_32_padded_input.shape[4]=1;
for(i=0;i<4374;i++){
#pragma HLS unroll factor = 16
	conv2d_32_padded_input.array[i] = conv2d_32_padded_input_array[i];
}


size_t conv2d_32_pad[4] = {1,1,1,1}; 
float conv2d_32_fill = 0.0f; 
float conv2d_32_kernel_array[324] = {
-2.42991507e-01f,-9.41696167e-02f,-2.50555664e-01f,+3.11655849e-01f,-1.65396273e-01f,
-2.06600472e-01f,+5.43172099e-02f,+6.00131080e-02f,-1.69903815e-01f,+2.18926147e-01f,
-1.02864169e-02f,-1.09669976e-01f,+1.04335405e-01f,-2.19096646e-01f,-2.35091820e-01f,
+2.95090675e-01f,+6.33305609e-02f,+1.96887493e-01f,-1.21757686e-01f,-1.52926207e-01f,
+2.49198973e-02f,+5.43366894e-02f,+8.92918408e-02f,+8.43058620e-03f,-2.17486531e-01f,
-1.36678040e-01f,+1.63259462e-01f,-3.55571136e-02f,+7.93687478e-02f,-1.32912397e-01f,
+2.06265956e-01f,-6.71141744e-02f,+1.08925171e-01f,-3.37103866e-02f,+2.63966769e-01f,
+3.64004262e-02f,-1.83752477e-01f,-2.08578240e-02f,-1.45349011e-01f,+1.52751490e-01f,
+2.84373611e-01f,-4.62662429e-03f,-2.49823630e-01f,+1.16315544e-01f,+2.80783445e-01f,
+1.29630819e-01f,-4.94626770e-03f,+5.50445393e-02f,+4.70674075e-02f,-1.36824131e-01f,
+4.69086096e-02f,+2.71243770e-02f,+2.52073526e-01f,-5.53016216e-02f,-1.99188851e-02f,
+8.44234005e-02f,+2.30629310e-01f,-4.59394194e-02f,+2.17185572e-01f,+1.36134803e-01f,
-2.16513425e-01f,-1.05455583e-02f,-3.73622663e-02f,+1.63006946e-01f,+9.43793952e-02f,
+1.96413219e-01f,+3.83610457e-01f,+1.07583649e-01f,+5.28189689e-02f,-6.60886243e-02f,
-9.22870561e-02f,+2.88225740e-01f,+2.16770545e-02f,-1.62092254e-01f,-2.51907073e-02f,
-4.99291271e-02f,+1.65497020e-01f,+7.11964443e-02f,+1.34703770e-01f,-1.45072728e-01f,
+1.17515355e-01f,-1.72248986e-02f,+9.48797911e-02f,+6.93951398e-02f,-1.07312046e-01f,
-1.29096240e-01f,+1.81496948e-01f,-2.77379543e-01f,+2.10978493e-01f,-1.13620833e-01f,
-3.88167128e-02f,-1.76452026e-01f,+1.46200433e-01f,-2.75364935e-01f,+1.16663650e-01f,
+2.04862598e-02f,+8.27394724e-02f,-7.66973644e-02f,+1.93453133e-01f,-2.89601564e-01f,
+1.99851751e-01f,-8.33774731e-02f,+1.64923042e-01f,+1.01163909e-02f,-2.06306368e-01f,
-2.93942057e-02f,+4.24271733e-01f,-2.32739463e-01f,+1.70796234e-02f,+7.19628036e-02f,
-2.92110406e-02f,+1.39662221e-01f,+1.34596884e-01f,-6.76248968e-02f,-1.55020162e-01f,
+1.14862092e-01f,+1.57663152e-01f,-5.49875498e-02f,+3.14930268e-02f,+4.84021157e-02f,
+2.15143617e-02f,-5.08207791e-02f,+1.84389427e-01f,-1.56590686e-04f,-1.96578741e-01f,
+2.60022491e-01f,+1.40031725e-01f,+5.36333025e-02f,-1.10609353e-01f,-5.06840050e-02f,
+2.04826310e-01f,-1.28627375e-01f,-9.77436919e-03f,-1.36591733e-01f,+1.74301937e-01f,
+1.57152981e-01f,+5.42529672e-02f,-1.75982848e-01f,-9.60171968e-02f,-1.79055855e-01f,
-7.53299072e-02f,-6.77911937e-02f,+1.97898239e-01f,-4.39429693e-02f,-2.08379090e-01f,
+1.68994829e-01f,-1.64160743e-01f,+1.38507653e-02f,+2.21173987e-01f,+2.19782084e-01f,
-1.58976957e-01f,+3.62555504e-01f,+3.26623097e-02f,-5.46973981e-02f,-7.77719319e-02f,
-7.07631186e-02f,+8.85020867e-02f,+8.09013993e-02f,+8.40102360e-02f,+2.76371747e-01f,
-2.15709686e-01f,-1.79736421e-01f,+2.33478501e-01f,+2.74220407e-01f,+2.46612892e-01f,
+6.72751246e-03f,+1.28493354e-01f,-1.14644155e-01f,+2.10569128e-01f,+4.06231254e-01f,
+2.43501306e-01f,+4.16837096e-01f,-2.14080170e-01f,-6.86105415e-02f,+4.64578927e-01f,
+4.76255924e-01f,+5.37073873e-02f,+1.27926826e-01f,+2.99726963e-01f,+1.58088535e-01f,
+2.12276101e-01f,-1.12981357e-01f,-1.91375967e-02f,-6.41581975e-03f,-1.30428940e-01f,
-8.33897665e-02f,+3.46222110e-02f,+5.33003546e-02f,+3.53016891e-02f,-2.23588571e-01f,
+1.99370816e-01f,+2.01154083e-01f,+9.97343585e-02f,+1.87515363e-01f,-7.94615000e-02f,
-2.35119119e-01f,-1.63246170e-02f,-2.36754417e-02f,+8.63159224e-02f,+1.09877594e-01f,
-8.72904435e-02f,+1.48108751e-01f,+1.89316511e-01f,+1.89942215e-02f,-1.77918270e-01f,
-1.27698123e-01f,+1.60322919e-01f,-2.38936186e-01f,+1.08815096e-01f,-2.13006973e-01f,
-1.75318159e-02f,-1.81343362e-01f,-3.61369759e-01f,+2.93000728e-01f,+2.88903594e-01f,
-1.76796943e-01f,-1.09587371e-01f,-5.27790710e-02f,-2.10071042e-01f,+2.63924181e-01f,
-2.05201864e-01f,+3.37960660e-01f,+1.48733675e-01f,+1.15576396e-02f,-1.31721169e-01f,
+2.09327519e-01f,-1.25735685e-01f,+6.20676689e-02f,+1.28649026e-01f,-6.11752532e-02f,
-2.46955797e-01f,-1.14892550e-01f,-1.15571350e-01f,+2.75588661e-01f,+1.76819727e-01f,
-2.14138314e-01f,+1.09975776e-02f,+2.57268131e-01f,-2.02800576e-02f,+1.11252710e-01f,
-1.20787106e-01f,+3.35336715e-01f,+2.58621685e-02f,+2.13731825e-01f,-2.30958387e-01f,
+8.46889764e-02f,+1.66381210e-01f,+3.63624513e-01f,-1.14021108e-01f,+1.80250570e-01f,
+1.21718310e-01f,-3.95931304e-02f,-1.47679090e-01f,+3.01764190e-01f,+6.47578463e-02f,
-1.61245242e-02f,-2.10164502e-01f,-1.46986037e-01f,-1.58560336e-01f,+1.15413956e-01f,
-1.18647575e-01f,+5.11189140e-02f,-1.96179822e-01f,+1.02448836e-01f,+1.07115291e-01f,
+2.97764778e-01f,+5.92014492e-02f,+2.28758961e-01f,-1.13412753e-01f,+4.46779914e-02f,
-1.44700050e-01f,+3.23328435e-01f,+8.17080513e-02f,-9.83696654e-02f,+1.23042375e-01f,
+1.17021933e-01f,+1.86637789e-01f,+9.79595855e-02f,-5.13141304e-02f,-1.24195442e-01f,
+2.01624796e-01f,+2.57004082e-01f,+3.07875484e-01f,-3.88218254e-01f,+6.52309656e-02f,
-9.83934035e-04f,-2.71119863e-01f,+2.41245300e-01f,-2.80485988e-01f,+2.92869974e-02f,
-1.14926301e-01f,+1.56813517e-01f,+1.57949090e-01f,+1.17912628e-02f,+5.95899634e-02f,
-1.70107707e-01f,+1.98198915e-01f,-5.52805467e-03f,-1.90302178e-01f,+1.49179041e-01f,
+4.39523198e-02f,-1.67095527e-01f,+1.08542191e-02f,-2.13521793e-01f,-2.66406834e-01f,
-2.54886746e-02f,-9.15632993e-02f,-2.29698196e-01f,+2.76425034e-01f,+1.74955785e-01f,
-6.44290075e-02f,+2.07416087e-01f,+2.55627662e-01f,-3.50741185e-02f,+2.12791905e-01f,
-4.09947447e-02f,+1.83145300e-01f,+1.95811614e-01f,+2.27137282e-01f,-1.97713703e-01f,
+1.05351321e-01f,-2.27460966e-01f,+1.40302911e-01f,+7.91695639e-02f,}; 
//k2c_tensor conv2d_32_kernel = {&conv2d_32_kernel_array[0],4,324,{3,3,6,6,1}};
conv2d_32_kernel.ndim=4;
conv2d_32_kernel.numel=324;
conv2d_32_kernel.shape[0]=3;
conv2d_32_kernel.shape[1]=3;
conv2d_32_kernel.shape[2]=6;
conv2d_32_kernel.shape[3]=6;
conv2d_32_kernel.shape[4]=1;
for(i=0;i<324;i++){
//#pragma HLS unroll factor = 16
#pragma HLS unroll
	conv2d_32_kernel.array[i] = conv2d_32_kernel_array[i];
}


float conv2d_32_bias_array[6] = {
-1.05229644e-02f,-1.86059140e-02f,+4.41780351e-02f,-3.38189267e-02f,+2.72496324e-02f,
+1.03631876e-02f,}; 
//k2c_tensor conv2d_32_bias = {&conv2d_32_bias_array[0],1,6,{6,1,1,1,1}};
conv2d_32_bias.ndim=1;
conv2d_32_bias.numel=6;
conv2d_32_bias.shape[0]=6;
conv2d_32_bias.shape[1]=1;
conv2d_32_bias.shape[2]=1;
conv2d_32_bias.shape[3]=1;
conv2d_32_bias.shape[4]=1;
for(i=0;i<6;i++){
#pragma HLS unroll factor = 16
	conv2d_32_bias.array[i] = conv2d_32_bias_array[i];
}

 
size_t max_pooling2d_19_stride[2] = {2,2}; 
size_t max_pooling2d_19_pool_size[2] = {2,2}; 
float max_pooling2d_19_output_array[216] = {0}; 
//k2c_tensor max_pooling2d_19_output = {&max_pooling2d_19_output_array[0],3,216,{6,6,6,1,1}};
max_pooling2d_19_output.ndim=3;
max_pooling2d_19_output.numel=216;
max_pooling2d_19_output.shape[0]=6;
max_pooling2d_19_output.shape[1]=6;
max_pooling2d_19_output.shape[2]=6;
max_pooling2d_19_output.shape[3]=1;
max_pooling2d_19_output.shape[4]=1;
for(i=0;i<216;i++){
//#pragma HLS unroll factor = 16
#pragma HLS unroll
	max_pooling2d_19_output.array[i] = max_pooling2d_19_output_array[i];
}


size_t conv2d_33_stride[2] = {2,2}; 
size_t conv2d_33_dilation[2] = {1,1}; 
float conv2d_33_output_array[54] = {0}; 
//k2c_tensor conv2d_33_output = {&conv2d_33_output_array[0],3,54,{3,3,6,1,1}};
conv2d_33_output.ndim=3;
conv2d_33_output.numel=54;
conv2d_33_output.shape[0]=3;
conv2d_33_output.shape[1]=3;
conv2d_33_output.shape[2]=6;
conv2d_33_output.shape[3]=1;
conv2d_33_output.shape[4]=1;
for(i=0;i<54;i++){
#pragma HLS unroll factor = 16
	conv2d_33_output.array[i] = conv2d_33_output_array[i];
}


float conv2d_33_padded_input_array[384] = {0}; 
//k2c_tensor conv2d_33_padded_input = {&conv2d_33_padded_input_array[0],3,384,{8,8,6,1,1}};
conv2d_33_padded_input.ndim=3;
conv2d_33_padded_input.numel=384;
conv2d_33_padded_input.shape[0]=8;
conv2d_33_padded_input.shape[1]=8;
conv2d_33_padded_input.shape[2]=6;
conv2d_33_padded_input.shape[3]=1;
conv2d_33_padded_input.shape[4]=1;
for(i=0;i<384;i++){
#pragma HLS unroll factor = 16
	conv2d_33_padded_input.array[i] = conv2d_33_padded_input_array[i];
}


size_t conv2d_33_pad[4] = {1,1,1,1}; 
float conv2d_33_fill = 0.0f; 
float conv2d_33_kernel_array[324] = {
-5.51496111e-02f,-2.83863302e-03f,+2.93984503e-01f,-6.86843172e-02f,+8.06809515e-02f,
+8.06850195e-02f,-1.66452691e-01f,+1.17277920e-01f,+1.28775686e-01f,+2.15194672e-02f,
-9.51426476e-02f,+1.21584266e-01f,+9.47755054e-02f,+2.98163861e-01f,+2.96940416e-01f,
+2.35650733e-01f,+2.30333030e-01f,-1.26420110e-01f,-4.89508174e-02f,+1.53292358e-01f,
-1.15212902e-01f,-8.80191028e-02f,+1.94121048e-01f,-1.84037402e-01f,+4.69109789e-02f,
+5.50377928e-02f,+1.78887978e-01f,+1.78681046e-01f,-3.29551190e-01f,-6.15045149e-03f,
-2.32060075e-01f,+1.39766797e-01f,+1.25350896e-02f,+2.35897198e-01f,+1.24583384e-02f,
+5.96910752e-02f,+1.68879122e-01f,+2.39702180e-01f,+4.02820222e-02f,+3.92773785e-02f,
-9.17100988e-04f,+2.11278707e-01f,+2.78082401e-01f,+1.40180424e-01f,-3.43017094e-02f,
-1.27749011e-01f,-5.70888668e-02f,+2.64862061e-01f,+1.52943954e-01f,+1.59658954e-01f,
-1.08870395e-01f,-3.24171782e-02f,-1.23750269e-01f,+1.99363809e-02f,-2.05995917e-01f,
-9.57779586e-03f,-8.52016360e-02f,+2.56199509e-01f,-1.62234321e-01f,-1.03517145e-01f,
+2.09758908e-01f,+2.30868757e-01f,-8.61602947e-02f,-8.28929618e-03f,+1.65738747e-01f,
-1.17904386e-02f,-1.25920847e-02f,-2.47840732e-01f,+2.24531621e-01f,+2.59399384e-01f,
+2.11235598e-01f,+1.42377242e-02f,+1.53416887e-01f,+1.44716695e-01f,-1.54320210e-01f,
+6.70367405e-02f,+1.46862477e-01f,-5.55358082e-02f,-5.44832014e-02f,+1.13745131e-01f,
-6.57121884e-03f,+6.51032627e-02f,+4.13057059e-02f,-1.19432479e-01f,-2.80411039e-02f,
-1.66439325e-01f,-2.59109344e-02f,-1.39436163e-02f,+1.29570350e-01f,-1.09227471e-01f,
-1.85952336e-01f,-1.34604886e-01f,+2.54689306e-01f,+6.07715882e-02f,-1.12742126e-01f,
+1.60125867e-01f,+2.16243550e-01f,+2.58181065e-01f,+3.04981649e-01f,-1.70879930e-01f,
-1.66899323e-01f,-3.29151414e-02f,+1.92651570e-01f,-1.07604377e-01f,+1.07066995e-02f,
+1.81879625e-01f,-1.55081227e-01f,-7.35557154e-02f,-2.34761849e-01f,+1.24908552e-01f,
+2.13239819e-01f,-4.37986664e-02f,-1.06420778e-01f,+2.07176253e-01f,+9.79883894e-02f,
+1.45830080e-01f,-1.45630553e-01f,-4.42648649e-01f,-1.48953527e-01f,+4.10279512e-01f,
-1.98102921e-01f,+1.60507709e-01f,-2.22395405e-01f,-5.48025295e-02f,+1.32941023e-01f,
-6.34911284e-02f,-2.23264471e-01f,+5.36867939e-02f,+6.12628497e-02f,-6.77458495e-02f,
+2.65800595e-01f,+3.76823813e-01f,-1.56801730e-01f,-1.96892604e-01f,-3.28489542e-02f,
+2.93541282e-01f,+8.14754665e-02f,+1.75759539e-01f,-1.37430832e-01f,+1.49555996e-01f,
+2.06516787e-01f,+1.42919376e-01f,-1.55128971e-01f,-3.78174968e-02f,-1.37115613e-01f,
-1.06086403e-01f,-1.73292369e-01f,-2.09430209e-03f,-1.27235383e-01f,+2.33105898e-01f,
+4.49778363e-02f,+1.40911356e-01f,-1.53517500e-01f,-8.22302978e-03f,-5.99886551e-02f,
+1.20845735e-02f,+8.57889466e-03f,-1.17284931e-01f,+1.34355813e-01f,-2.10256159e-01f,
-2.08813548e-01f,+1.53526694e-01f,-2.48437002e-01f,+1.23043723e-01f,-2.05626577e-01f,
-8.58055875e-02f,+1.52333736e-01f,-1.86079860e-01f,+1.81300417e-01f,-2.31449440e-01f,
+2.37068251e-01f,+2.45377913e-01f,+1.96469963e-01f,+2.15336278e-01f,-1.59793481e-01f,
+1.06092222e-01f,-2.30606183e-01f,-8.34874436e-02f,-2.10610971e-01f,-5.53052276e-02f,
+1.27906099e-01f,-1.00836791e-01f,-3.71027619e-01f,-1.19139999e-01f,-1.25437781e-01f,
-1.59062117e-01f,+1.64420724e-01f,+5.90836555e-02f,-1.82409793e-01f,+4.29659802e-03f,
-1.62863359e-01f,-3.44680212e-02f,-1.04148082e-01f,+1.00593328e-01f,-1.15495138e-01f,
+1.71992362e-01f,-2.05961794e-01f,-1.19573288e-01f,-5.90708926e-02f,-9.26475823e-02f,
-1.45820469e-01f,-5.19344807e-02f,+7.05188736e-02f,-1.80655658e-01f,+1.10382751e-01f,
-1.27057195e-01f,-2.02535003e-01f,-1.22534804e-01f,-8.32247511e-02f,-2.31773973e-01f,
+8.15718397e-02f,+4.11895029e-02f,-1.04211114e-01f,-2.02026367e-01f,+1.39483944e-01f,
-4.29869331e-02f,-2.21142232e-01f,+1.28311828e-01f,+2.72666454e-01f,+2.40478873e-01f,
+1.16688691e-01f,-5.76613098e-02f,-1.77973375e-01f,+1.87265009e-01f,+1.40634123e-02f,
+2.08247826e-01f,+2.79887706e-01f,+1.56247793e-02f,+1.68577895e-01f,+5.67968562e-02f,
-2.26394176e-01f,+1.79026857e-01f,+1.89090163e-01f,+1.12214878e-01f,+1.45703331e-01f,
-2.11022913e-01f,+2.48122085e-02f,-4.03534807e-02f,+2.18421221e-01f,-5.18570989e-02f,
+7.97028542e-02f,-1.82024881e-01f,+1.95127666e-01f,-2.99745630e-02f,+5.50262779e-02f,
-1.64944187e-01f,-1.88812241e-01f,+1.89636156e-01f,+2.43201833e-02f,+6.28818274e-02f,
-1.12630509e-01f,+2.79187173e-01f,+2.19013374e-02f,-1.42974153e-01f,-2.64548004e-01f,
-9.92665067e-02f,+1.77333906e-01f,-2.62964904e-01f,+2.72237025e-02f,-1.26930222e-01f,
-7.05568418e-02f,+2.78046019e-02f,-1.25596285e-01f,+2.31829472e-03f,+1.99356556e-01f,
-2.50119954e-01f,-4.40520272e-02f,-2.29622096e-01f,-7.94604868e-02f,-6.00036457e-02f,
-1.54453337e-01f,+1.44547611e-01f,-1.46896943e-01f,+7.46921822e-02f,+1.59529984e-01f,
+1.72589883e-01f,+2.49118969e-01f,-9.26574320e-02f,+1.78712860e-01f,-9.48799774e-02f,
-3.65295284e-03f,-2.47194603e-01f,+2.10288420e-01f,+1.83297426e-01f,-1.98328614e-01f,
+2.56977469e-01f,-9.83446836e-02f,+1.41729787e-01f,-6.35889322e-02f,-1.31404459e-01f,
+2.67214924e-01f,-1.32852439e-02f,+1.85946766e-02f,-6.47233054e-02f,-1.78084776e-01f,
-3.74355167e-02f,-1.00021668e-01f,+1.42636240e-01f,+9.47594568e-02f,-6.10060990e-02f,
+9.70502794e-02f,+3.06062132e-01f,+2.13770300e-01f,+3.88263632e-03f,+2.05515213e-02f,
+1.86783761e-01f,+1.19127102e-01f,-3.81415454e-03f,+2.48969615e-01f,-1.91465002e-02f,
-2.64582485e-01f,-2.22432777e-01f,-8.77540633e-02f,+1.12600587e-01f,-1.20955342e-02f,
+1.60030842e-01f,-1.24241158e-01f,+2.53842045e-02f,-2.46249542e-01f,-2.19503604e-02f,
+9.00538042e-02f,+7.91006088e-02f,+1.10346470e-02f,+2.04289332e-01f,}; 
//k2c_tensor conv2d_33_kernel = {&conv2d_33_kernel_array[0],4,324,{3,3,6,6,1}};
conv2d_33_kernel.ndim=4;
conv2d_33_kernel.numel=324;
conv2d_33_kernel.shape[0]=3;
conv2d_33_kernel.shape[1]=3;
conv2d_33_kernel.shape[2]=6;
conv2d_33_kernel.shape[3]=6;
conv2d_33_kernel.shape[4]=1;
for(i=0;i<324;i++){
//#pragma HLS unroll factor = 16
#pragma HLS unroll
	conv2d_33_kernel.array[i] = conv2d_33_kernel_array[i];
}


float conv2d_33_bias_array[6] = {
+2.40350924e-02f,+4.66177613e-02f,+6.93593174e-02f,+1.62837710e-02f,+4.88358885e-02f,
+8.79841857e-03f,}; 
//k2c_tensor conv2d_33_bias = {&conv2d_33_bias_array[0],1,6,{6,1,1,1,1}};
conv2d_33_bias.ndim=1;
conv2d_33_bias.numel=6;
conv2d_33_bias.shape[0]=6;
conv2d_33_bias.shape[1]=1;
conv2d_33_bias.shape[2]=1;
conv2d_33_bias.shape[3]=1;
conv2d_33_bias.shape[4]=1;
for(i=0;i<6;i++){
#pragma HLS unroll factor = 16
	conv2d_33_bias.array[i] = conv2d_33_bias_array[i];
}

 
size_t conv2d_34_stride[2] = {2,2}; 
size_t conv2d_34_dilation[2] = {1,1}; 
float conv2d_34_output_array[28] = {0}; 
//k2c_tensor conv2d_34_output = {&conv2d_34_output_array[0],3,28,{2,2,7,1,1}};
conv2d_34_output.ndim=3;
conv2d_34_output.numel=28;
conv2d_34_output.shape[0]=2;
conv2d_34_output.shape[1]=2;
conv2d_34_output.shape[2]=7;
conv2d_34_output.shape[3]=1;
conv2d_34_output.shape[4]=1;
for(i=0;i<28;i++){
#pragma HLS unroll factor = 16
	conv2d_34_output.array[i] = conv2d_34_output_array[i];
}


float conv2d_34_padded_input_array[150] = {0}; 
//k2c_tensor conv2d_34_padded_input = {&conv2d_34_padded_input_array[0],3,150,{5,5,6,1,1}};
conv2d_34_padded_input.ndim=3;
conv2d_34_padded_input.numel=150;
conv2d_34_padded_input.shape[0]=5;
conv2d_34_padded_input.shape[1]=5;
conv2d_34_padded_input.shape[2]=6;
conv2d_34_padded_input.shape[3]=1;
conv2d_34_padded_input.shape[4]=1;
for(i=0;i<150;i++){
//#pragma HLS unroll factor = 16
#pragma HLS unroll
	conv2d_34_padded_input.array[i] = conv2d_34_padded_input_array[i];
}


size_t conv2d_34_pad[4] = {1,1,1,1}; 
float conv2d_34_fill = 0.0f; 
float conv2d_34_kernel_array[378] = {
+2.24145606e-01f,-8.86079967e-02f,+1.09926075e-01f,+1.41116053e-01f,+1.53779000e-01f,
+1.95325568e-01f,-2.18288586e-01f,+1.46245986e-01f,-1.78555459e-01f,-1.11616574e-01f,
-7.99848884e-03f,+1.36989415e-01f,+7.35270604e-02f,+2.70698518e-01f,+1.31896481e-01f,
+5.36857307e-01f,+4.09597903e-02f,+1.03720240e-01f,+1.43310726e-01f,+1.19212285e-01f,
-2.64420927e-01f,+5.49483970e-02f,+1.39965951e-01f,+1.47624522e-01f,+2.06167966e-01f,
-4.88531291e-02f,+3.46192345e-02f,-1.39204338e-01f,+3.34356189e-01f,-3.29930753e-01f,
-8.72489586e-02f,-1.08665779e-01f,-4.81809527e-02f,+9.81131122e-02f,-2.72341728e-01f,
+1.23837469e-02f,+1.57972410e-01f,-1.06450766e-01f,+2.89909374e-02f,-3.38807292e-02f,
-1.21789016e-01f,+1.38845325e-01f,+9.84231830e-02f,+2.74290234e-01f,-1.15526654e-03f,
+1.47285685e-01f,+2.21052602e-01f,-1.89449266e-02f,-2.80515075e-01f,-9.15792510e-02f,
+2.38070041e-01f,-1.94731429e-01f,-1.36573106e-01f,-9.81724113e-02f,-4.34242971e-02f,
+1.71729818e-01f,-2.96880096e-01f,+3.39125514e-01f,+3.11485976e-01f,-3.87439854e-03f,
+1.36859074e-01f,-1.55678213e-01f,+9.55598429e-02f,-1.56445712e-01f,+6.55418262e-02f,
+1.06047817e-01f,+4.45820875e-02f,-2.14645624e-01f,+7.12015554e-02f,+2.55005628e-01f,
+2.40655541e-01f,-5.59415184e-02f,+7.25832162e-03f,-1.79743856e-01f,-4.59004845e-03f,
-1.20974936e-01f,+2.25672036e-01f,+6.15282543e-02f,-2.36403104e-02f,+1.28477201e-01f,
-1.16794594e-01f,+1.25679985e-01f,+2.23039120e-01f,-2.67457604e-01f,-4.32606041e-02f,
+1.02685407e-01f,-4.11798865e-01f,-2.14814544e-01f,-1.33173093e-01f,-1.15616992e-01f,
+1.65466264e-01f,-1.15189955e-01f,+1.01787068e-01f,+1.35783523e-01f,+2.41553068e-01f,
-9.17699039e-02f,-3.31595689e-02f,-1.32089913e-01f,-3.75136137e-02f,+8.36537331e-02f,
+1.95191249e-01f,+8.96301568e-02f,+2.73587734e-01f,+4.29409206e-01f,+5.71349077e-02f,
+1.73441380e-01f,-6.05926961e-02f,-1.99396864e-01f,+1.23127706e-01f,-5.12512811e-02f,
-1.87724847e-02f,+2.37868398e-01f,+2.05285355e-01f,+1.26327395e-01f,+4.79238808e-01f,
+5.12222480e-03f,-1.21651478e-01f,+1.04584932e-01f,+3.30003612e-02f,-1.14646800e-01f,
+2.56680306e-02f,+1.32812411e-02f,+2.33065620e-01f,+5.77191040e-02f,+4.95119877e-02f,
+9.05289687e-03f,-2.06755981e-01f,+7.61919096e-02f,-1.71443462e-01f,+1.29764527e-02f,
-1.68262392e-01f,-4.09627892e-02f,+2.27739096e-01f,+1.87747926e-01f,-1.98942423e-01f,
-2.11202905e-01f,-1.27042487e-01f,+1.27915755e-01f,+2.09093001e-03f,-6.66767061e-02f,
-3.45802575e-01f,-1.45346120e-01f,+1.56494081e-01f,-6.63198531e-02f,-4.90428433e-02f,
+2.75981337e-01f,-8.39811116e-02f,-1.12069082e-02f,+1.54079497e-01f,-1.54668510e-01f,
+4.70534600e-02f,+1.36507988e-01f,+1.22894626e-02f,-1.76030591e-01f,+6.93553360e-03f,
+1.75953403e-01f,+5.77667095e-02f,+3.26263942e-02f,+5.76351807e-02f,+1.13143951e-01f,
-2.60057986e-01f,+5.40493429e-02f,-5.28306700e-02f,+9.93893743e-02f,+1.21243425e-01f,
+6.53099641e-02f,-2.20297933e-01f,-1.22333474e-01f,-1.56766310e-01f,+3.51061486e-02f,
-5.69364391e-02f,+1.73518911e-01f,+1.37752503e-01f,+2.19504476e-01f,-6.60876185e-02f,
-2.01594889e-01f,-2.17210755e-01f,+6.40132949e-02f,-9.11101624e-02f,-1.33178309e-01f,
-1.09316275e-01f,+1.89780467e-03f,-3.21785897e-01f,+9.51922610e-02f,+1.72630936e-01f,
+1.19031549e-01f,-2.97212750e-02f,+9.57667679e-02f,+1.45639896e-01f,-2.27868453e-01f,
-2.39449009e-01f,-1.60599589e-01f,-1.39763787e-01f,+1.16861470e-01f,+1.46135673e-01f,
+2.18628213e-01f,+9.64159220e-02f,-1.28540829e-01f,+2.88591951e-01f,+5.18058836e-02f,
-1.97096869e-01f,+1.61550075e-01f,+3.11124086e-01f,+7.52826035e-02f,+1.95058405e-01f,
-1.05845526e-01f,+1.14985578e-01f,+4.64015640e-02f,+8.63521323e-02f,-1.41869128e-01f,
+8.93438682e-02f,+9.64605585e-02f,-1.93681970e-01f,-2.08257228e-01f,+2.39917845e-01f,
-1.69526562e-01f,-7.86873326e-02f,+1.77791826e-02f,-1.79427862e-01f,+2.51544714e-02f,
+1.56056166e-01f,+1.79733321e-01f,+1.37574330e-01f,+2.21561104e-01f,-9.22568887e-02f,
-3.82618785e-01f,-4.49474245e-01f,-2.22676426e-01f,-2.06574529e-01f,+8.34041312e-02f,
+2.84075707e-01f,+8.11152626e-03f,+1.77778870e-01f,-1.83724493e-01f,+1.27813727e-01f,
+9.74155590e-02f,+1.35494739e-01f,+1.06799016e-02f,-1.55059800e-01f,-1.49196684e-01f,
-5.82290366e-02f,-1.42990649e-01f,+1.76705047e-02f,-1.70190722e-01f,-3.71397175e-02f,
+3.81282307e-02f,+1.27425924e-01f,-1.70751169e-01f,+2.77743429e-01f,+4.72032800e-02f,
+2.70727038e-01f,-3.02325875e-01f,+1.40458182e-01f,-9.78460237e-02f,-1.70146197e-01f,
-2.96149626e-02f,-2.86857877e-02f,-1.95604458e-01f,-1.89242586e-02f,-1.93159416e-01f,
+2.49029193e-02f,-4.82914671e-02f,-1.97638497e-01f,-1.07271135e-01f,-1.09481968e-01f,
+4.87438329e-02f,-1.04010396e-01f,-4.65234742e-02f,-1.04690723e-01f,-2.12388799e-01f,
+3.15485597e-01f,-6.86713308e-02f,-7.28090405e-02f,+1.56345516e-01f,+1.08664140e-01f,
+5.65150753e-02f,+1.65570706e-01f,-1.12736993e-01f,+1.76311836e-01f,+1.71434239e-01f,
+1.18175462e-01f,-2.51497291e-02f,-1.51515305e-01f,+1.44591078e-01f,-9.36756730e-02f,
+1.12701960e-01f,-2.02682167e-01f,+2.16855139e-01f,-1.86101511e-01f,+1.19140700e-01f,
-2.25286677e-01f,-3.17053974e-01f,+2.57644862e-01f,+4.90731932e-02f,+1.16005041e-01f,
-4.49114218e-02f,+1.41304061e-01f,+1.40227407e-01f,+1.01164673e-02f,-2.55052090e-01f,
-1.31063491e-01f,-1.95968792e-01f,+1.31749362e-01f,+5.34476563e-02f,-1.20483801e-01f,
+1.31762266e-01f,+1.02261186e-01f,+1.01698227e-01f,-1.33905739e-01f,-1.76254228e-01f,
+9.06915069e-02f,-1.93292394e-01f,+1.32565200e-01f,+2.76822120e-01f,+6.34457842e-02f,
+2.31109485e-01f,+5.36620915e-02f,-1.77276924e-01f,+4.30907495e-02f,-2.34410271e-01f,
+1.92378387e-02f,-8.02831352e-02f,+2.95489341e-01f,+5.54062687e-02f,-1.75470114e-01f,
-9.71943438e-02f,+1.42239317e-01f,-1.49249911e-01f,+1.74478516e-01f,-7.39388838e-02f,
-1.88180819e-01f,-1.13798685e-01f,-2.94676363e-01f,+2.27136150e-01f,-7.41555542e-02f,
-2.24737063e-01f,+2.51474470e-01f,+2.44194448e-01f,-1.01565104e-02f,+2.42107853e-01f,
-9.13316086e-02f,-5.97336777e-02f,-4.72925715e-02f,+1.22446835e-01f,+3.62288356e-02f,
+2.39661440e-01f,+1.45906880e-01f,+1.68109745e-01f,+1.84464946e-01f,-2.03358650e-01f,
+3.28944296e-01f,+3.58338147e-01f,-5.93382642e-02f,+8.15942809e-02f,-1.48516059e-01f,
+6.52197376e-02f,+1.00544058e-01f,+2.44387299e-01f,+1.73095584e-01f,+6.44647181e-02f,
-1.26521945e-01f,-1.01256609e-01f,+2.26479128e-01f,+2.03589108e-02f,-1.25155961e-02f,
+2.02554613e-01f,-1.88369825e-02f,+2.56608397e-01f,-2.65012383e-01f,-9.15455595e-02f,
+2.94647599e-03f,+9.60816592e-02f,-2.66842283e-02f,-2.12714881e-01f,-6.11218959e-02f,
+2.54258543e-01f,-5.43336086e-02f,-1.38285294e-01f,}; 
//k2c_tensor conv2d_34_kernel = {&conv2d_34_kernel_array[0],4,378,{3,3,6,7,1}};
conv2d_34_kernel.ndim=4;
conv2d_34_kernel.numel=378;
conv2d_34_kernel.shape[0]=3;
conv2d_34_kernel.shape[1]=3;
conv2d_34_kernel.shape[2]=6;
conv2d_34_kernel.shape[3]=7;
conv2d_34_kernel.shape[4]=1;
for(i=0;i<378;i++){
//#pragma HLS unroll factor = 16
#pragma HLS unroll
	conv2d_34_kernel.array[i] = conv2d_34_kernel_array[i];
}


float conv2d_34_bias_array[7] = {
-5.06505882e-03f,+3.27936597e-02f,+1.27908438e-02f,+2.13456601e-02f,+1.88654326e-02f,
+6.39882758e-02f,+6.42094463e-02f,}; 
//k2c_tensor conv2d_34_bias = {&conv2d_34_bias_array[0],1,7,{7,1,1,1,1}};
conv2d_34_bias.ndim=1;
conv2d_34_bias.numel=7;
conv2d_34_bias.shape[0]=7;
conv2d_34_bias.shape[1]=1;
conv2d_34_bias.shape[2]=1;
conv2d_34_bias.shape[3]=1;
conv2d_34_bias.shape[4]=1;
for(i=0;i<7;i++){
#pragma HLS unroll factor = 16
	conv2d_34_bias.array[i] = conv2d_34_bias_array[i];
}

 
float flatten_8_output_array[28] = {0}; 
//k2c_tensor flatten_8_output = {&flatten_8_output_array[0],1,28,{28, 1, 1, 1, 1}};
flatten_8_output.ndim=1;
flatten_8_output.numel=28;
flatten_8_output.shape[0]=28;
flatten_8_output.shape[1]=1;
flatten_8_output.shape[2]=1;
flatten_8_output.shape[3]=1;
flatten_8_output.shape[4]=1;
for(i=0;i<28;i++){
#pragma HLS unroll factor = 16
	flatten_8_output.array[i] = flatten_8_output_array[i];
}


float dense_29_output_array[84] = {0}; 
//k2c_tensor dense_29_output = {&dense_29_output_array[0],1,84,{84, 1, 1, 1, 1}};
dense_29_output.ndim=1;
dense_29_output.numel=84;
dense_29_output.shape[0]=84;
dense_29_output.shape[1]=1;
dense_29_output.shape[2]=1;
dense_29_output.shape[3]=1;
dense_29_output.shape[4]=1;
for(i=0;i<84;i++){
#pragma HLS unroll factor = 16
	dense_29_output.array[i] = dense_29_output_array[i];
}


float dense_29_kernel_array[2352] = {
-1.59480438e-01f,-1.82979956e-01f,+2.05214322e-01f,+2.59178072e-01f,-6.94778329e-03f,
-1.10708959e-01f,+5.93362190e-02f,+4.83253933e-02f,+2.16566712e-01f,-4.51315939e-02f,
-4.89606820e-02f,-2.16429636e-01f,-1.94608480e-01f,-4.24507260e-01f,+3.12974632e-01f,
+2.98573971e-01f,+4.03535478e-02f,-6.97174221e-02f,+9.48413089e-03f,+1.93381801e-01f,
+1.56352714e-01f,-1.88951209e-01f,-2.22038850e-01f,+1.23717591e-01f,-1.69908360e-01f,
+3.47169265e-02f,+1.64765909e-01f,-2.17172131e-01f,+2.16418263e-02f,+2.15070974e-02f,
-2.73084432e-01f,-1.84354946e-01f,-1.47074118e-01f,-8.78807344e-03f,-2.23750517e-01f,
-2.31558204e-01f,-6.18091375e-02f,-1.68503180e-01f,-5.44021185e-03f,-2.15167403e-01f,
+8.60619452e-03f,-1.82290614e-01f,+9.24373195e-02f,-2.27395780e-02f,+2.22813725e-01f,
+1.71431433e-02f,+7.26954788e-02f,+2.75038093e-01f,-7.02966191e-03f,-1.44569725e-01f,
+2.38679945e-01f,+1.30419165e-01f,+2.62200028e-01f,-2.27826908e-01f,+1.22717381e-01f,
-3.93760018e-02f,-3.90243120e-02f,-3.92068960e-02f,-5.07141277e-02f,+2.70171408e-02f,
-1.76136434e-01f,+9.82671231e-03f,+2.05570981e-01f,+1.30887449e-01f,-3.70279811e-02f,
-1.13328777e-01f,-1.46024749e-01f,-2.48132691e-01f,+1.44597173e-01f,+2.14089066e-01f,
-2.31548329e-03f,+8.26303661e-02f,+2.16387480e-01f,+1.21083394e-01f,-2.82889325e-02f,
-5.38544171e-02f,-1.82713330e-01f,-1.99271157e-01f,+7.15527311e-02f,+5.29192351e-02f,
+8.48044679e-02f,-2.23526657e-02f,+8.25346559e-02f,-9.31188464e-02f,+9.97465663e-03f,
-1.09691344e-01f,-1.28527403e-01f,+2.26824462e-01f,+4.42223400e-02f,-4.57711145e-03f,
-2.81555891e-01f,-4.07329500e-02f,-1.45266756e-01f,-2.53465772e-01f,+7.23050758e-02f,
-3.52627814e-01f,-6.56127706e-02f,-4.87734199e-01f,+3.82459551e-01f,+5.38264401e-02f,
+5.87179251e-02f,+2.29090169e-01f,-2.63507307e-01f,+2.87806660e-01f,-1.28452301e-01f,
-3.99654321e-02f,+1.02916256e-01f,-2.33075246e-02f,-2.63409823e-01f,+2.74825603e-01f,
+4.12607819e-01f,+1.35754734e-01f,+4.03318375e-01f,+1.51565328e-01f,-2.33820394e-01f,
-1.34019881e-01f,+1.63749546e-01f,+6.62653595e-02f,+2.26713538e-01f,-1.57057103e-02f,
-2.10091352e-01f,-4.19181809e-02f,+1.46098673e-01f,+1.99007794e-01f,+2.15907227e-02f,
-3.04035574e-01f,+2.48023331e-01f,+1.81832582e-01f,-1.66793051e-03f,+2.33331203e-01f,
+1.83336139e-02f,+2.97167569e-01f,+2.13521644e-01f,+2.21678942e-01f,+1.64911360e-01f,
+1.01298012e-01f,+3.87224182e-02f,+1.70762613e-01f,+2.22099140e-01f,-1.17987894e-01f,
-3.00834745e-01f,-3.16812545e-01f,-2.42913455e-01f,+3.85360420e-01f,-1.34886831e-01f,
-7.39307925e-02f,+2.23053992e-01f,+3.55400801e-01f,+2.20722690e-01f,-1.02751441e-01f,
-1.54992744e-01f,-1.27226502e-01f,-1.25586599e-01f,+8.39184076e-02f,+3.51522937e-02f,
-1.06843963e-01f,+5.20480145e-03f,-3.66979837e-02f,-9.54104662e-02f,+4.07540351e-02f,
-1.71479911e-01f,+1.24207191e-01f,+2.78017759e-01f,-8.72609671e-03f,-3.28807622e-01f,
-2.14199796e-01f,-5.21890596e-02f,+1.48878187e-01f,+1.17555134e-01f,+2.09189057e-01f,
-9.37338173e-02f,-1.66560635e-01f,+2.37475902e-01f,+1.68518230e-01f,-8.34310204e-02f,
+1.05234891e-01f,+4.15655017e-01f,+2.80645549e-01f,-3.96645963e-02f,+2.56210685e-01f,
+3.02173883e-01f,+1.99404955e-01f,-9.92780402e-02f,-1.40246838e-01f,+1.20258749e-01f,
+9.84690487e-02f,+3.69192362e-01f,-1.46825448e-01f,+1.31373554e-01f,-2.06747532e-01f,
-6.09155446e-02f,-1.71477109e-01f,+3.92569661e-01f,-9.46764722e-02f,+5.88088557e-02f,
+9.91320014e-02f,-1.60794362e-01f,+1.72818705e-01f,+2.32632712e-01f,+1.83081403e-01f,
-9.86726880e-02f,+1.86446920e-01f,-4.42173481e-02f,+2.24206254e-01f,-5.97769618e-02f,
+1.41333282e-01f,-2.87203759e-01f,-2.04094835e-02f,+2.63437957e-01f,+4.17098612e-01f,
+9.77469385e-02f,+6.67575970e-02f,-1.65220141e-01f,+3.14323157e-02f,+2.68260896e-01f,
-1.79897636e-01f,+1.84190303e-01f,-5.99825531e-02f,-2.26320937e-01f,+2.08771359e-02f,
+1.07944243e-01f,+2.17159688e-02f,-6.58196434e-02f,+7.27692246e-02f,+6.60954788e-02f,
+2.04854667e-01f,-2.39995494e-02f,-2.05674805e-02f,-7.85925314e-02f,+1.78651929e-01f,
-3.21199805e-01f,+1.33664161e-01f,-2.96377659e-01f,-8.95047262e-02f,-1.49207339e-01f,
+2.83462375e-01f,+1.23016857e-01f,-1.19032346e-01f,-2.03263536e-01f,-1.54923722e-01f,
-4.97633889e-02f,+8.61074626e-02f,+1.63099825e-01f,+1.20476373e-01f,+1.96184099e-01f,
+1.15782917e-01f,-9.61975083e-02f,-1.99678726e-02f,+8.01116005e-02f,+1.50046021e-01f,
-2.36742441e-02f,+7.83001930e-02f,+1.78319067e-01f,-1.79771051e-01f,-1.58118904e-01f,
-4.26482875e-03f,-2.54768103e-01f,+7.40256831e-02f,+3.52312520e-04f,+1.09582953e-01f,
-1.45864666e-01f,+7.12859333e-02f,+2.55625695e-02f,+1.41383857e-01f,+2.54119039e-01f,
-1.64197996e-01f,+2.18246445e-01f,+2.02665925e-01f,+2.25154042e-01f,+2.44482264e-01f,
-1.87278122e-01f,+1.68088719e-01f,+1.04817495e-01f,-5.72543517e-02f,-1.46607608e-01f,
-2.04855055e-01f,-1.59490824e-01f,+1.34578109e-01f,+7.05724731e-02f,-1.98537186e-01f,
+1.49009943e-01f,+2.13231742e-01f,-5.49479760e-02f,+5.51273338e-02f,+7.74910226e-02f,
-7.28112906e-02f,-9.15712416e-02f,-1.51796848e-01f,-1.20623641e-01f,-2.01960057e-01f,
+6.57139421e-02f,-5.86182587e-02f,-1.43903151e-01f,-1.18858904e-01f,+3.31027918e-02f,
+1.83819413e-01f,-1.02449264e-02f,-4.07471731e-02f,-1.45935297e-01f,-9.71535817e-02f,
-1.87136099e-01f,+1.79489061e-01f,+2.48735324e-01f,-4.20524925e-02f,+9.51417256e-03f,
-8.97961184e-02f,-5.79036139e-02f,+9.89952162e-02f,-1.80793107e-01f,+1.33956343e-01f,
+4.42655422e-02f,+1.75527215e-01f,-2.43272558e-01f,-1.72793686e-01f,+3.04406583e-02f,
+2.85301805e-02f,+1.84372485e-01f,+8.26796796e-03f,+3.71563211e-02f,+3.81766222e-02f,
+1.49143398e-01f,-1.67346120e-01f,-1.81097075e-01f,-1.86476186e-01f,-1.37679443e-01f,
+1.89669892e-01f,+4.87076156e-02f,-1.56619415e-01f,+1.63425505e-03f,-1.32733032e-01f,
-1.77075058e-01f,+1.45848423e-01f,-1.55947208e-01f,-9.73565355e-02f,+5.83136380e-02f,
+2.07885150e-02f,-9.44258422e-02f,+5.50410189e-02f,+1.09405145e-01f,+5.81644848e-02f,
+4.56471331e-02f,+5.07115684e-02f,-1.34738371e-01f,+2.66312063e-03f,+2.98407912e-01f,
+1.68567151e-02f,-1.15026772e-01f,+1.17768764e-01f,-1.50170662e-02f,+2.02321947e-01f,
+2.18586817e-01f,-1.49081901e-01f,-1.56673372e-01f,+1.42551243e-01f,-1.82705745e-02f,
+1.00853041e-01f,-2.16878101e-01f,+1.49350107e-01f,+3.78077999e-02f,-1.64231837e-01f,
+2.71598428e-01f,-1.88489810e-01f,-2.28437960e-01f,-3.16778496e-02f,-1.15880236e-01f,
+1.20404512e-01f,+3.08836460e-01f,+8.39459226e-02f,+1.56394839e-01f,-1.27258291e-02f,
+1.77193522e-01f,+3.86000164e-02f,+1.62040323e-01f,-2.24307403e-01f,-1.57039076e-01f,
-1.91659495e-01f,+1.72748283e-01f,+2.51085460e-01f,-2.49875858e-02f,+1.78166553e-01f,
-1.37082100e-01f,-1.27609581e-01f,-1.10079922e-01f,-2.16575697e-01f,-7.20267892e-02f,
+5.85272014e-02f,-4.61791120e-02f,-3.93256880e-02f,-1.42063200e-01f,+1.38640761e-01f,
+5.69714978e-03f,-2.18932033e-01f,+4.81731147e-02f,+2.20424652e-01f,+1.65129825e-01f,
+1.47573734e-02f,+8.15433934e-02f,+1.92795098e-01f,-4.78655398e-02f,-1.95482612e-01f,
-5.94605990e-02f,+4.81322855e-02f,-1.84335932e-01f,-3.83946337e-02f,-1.26280673e-02f,
+1.11804627e-01f,-1.26899660e-01f,+9.74254534e-02f,+5.69049418e-02f,+1.18353404e-01f,
+7.08591640e-02f,+8.20225328e-02f,-2.24722460e-01f,+2.32380703e-02f,-2.25818589e-01f,
+1.12046681e-01f,+1.54181883e-01f,-4.41426747e-02f,-1.09455116e-01f,+1.42643392e-01f,
+6.22262023e-02f,+6.99872850e-03f,-7.27302060e-02f,-3.56217436e-02f,-2.89836302e-02f,
+1.50470868e-01f,+1.17968254e-01f,+6.62975535e-02f,-8.67486820e-02f,+1.13422118e-01f,
-9.28184167e-02f,-1.26187027e-01f,-1.85436904e-01f,+1.08019315e-01f,+2.39434123e-01f,
+6.08663149e-02f,+2.53543526e-01f,+2.27511629e-01f,+1.63773507e-01f,+2.13132992e-01f,
-2.61256844e-01f,-9.30686519e-02f,+5.96559644e-02f,+4.45683561e-02f,-6.49338737e-02f,
+4.29162495e-02f,+1.02962069e-02f,-1.03772320e-01f,-1.06732816e-01f,-2.00655341e-01f,
+1.65178671e-01f,-8.72246251e-02f,+3.09493709e-02f,-9.21614915e-02f,-1.93573102e-01f,
+2.17139333e-01f,+3.46589088e-03f,+1.05042510e-01f,+2.04595968e-01f,-2.00922042e-01f,
-1.90817952e-01f,-1.24544673e-01f,-5.55953458e-02f,+5.36364801e-02f,-7.19599575e-02f,
-1.66004561e-02f,-8.19374844e-02f,+2.21993744e-01f,+6.63594902e-03f,+2.29013965e-01f,
+7.30359778e-02f,-6.77613690e-02f,+4.20152321e-02f,+1.86072081e-01f,-2.21801866e-02f,
-1.28482521e-01f,+8.31520706e-02f,+2.01190803e-02f,+2.53745794e-01f,+2.43251547e-01f,
+1.73649997e-01f,+3.75159867e-02f,+9.49450582e-02f,+2.07177684e-01f,-5.98003007e-02f,
-1.35479212e-01f,-1.80819407e-01f,+1.79247901e-01f,-1.46537170e-01f,+1.15013652e-01f,
+6.98054060e-02f,-2.08091468e-01f,-2.00685859e-01f,+2.09008515e-01f,+2.32995339e-02f,
-8.82098451e-02f,-1.17481016e-01f,-7.15639442e-02f,-9.78212804e-02f,-1.05839573e-01f,
-7.80754238e-02f,+2.23351926e-01f,+4.72129732e-02f,+2.00491846e-01f,+2.25102574e-01f,
+1.92918330e-01f,+2.68354595e-01f,-3.56332064e-02f,-1.65150508e-01f,-1.44055247e-01f,
+9.97976288e-02f,-2.47439370e-01f,-1.81241691e-01f,-2.42722005e-01f,-3.56310718e-02f,
+1.32833540e-01f,-1.28622958e-02f,+1.00508472e-02f,+1.42138764e-01f,+3.03993851e-01f,
+2.83155620e-01f,+2.96524316e-01f,-1.64642781e-01f,+4.53246497e-02f,+4.91846465e-02f,
+7.94087648e-02f,-8.08121040e-02f,-8.15301612e-02f,-3.47230136e-01f,+1.75306633e-01f,
-1.10820919e-01f,-2.19198003e-01f,+1.32255167e-01f,-5.51913902e-02f,+1.07663430e-01f,
-2.58337051e-01f,+1.10077806e-01f,-1.34120554e-01f,+2.19331950e-01f,+8.44699964e-02f,
+4.86988723e-02f,+6.65364563e-02f,+1.12509638e-01f,-2.19119832e-01f,-5.58056943e-02f,
-3.81318510e-01f,+1.81818590e-01f,-1.90905824e-01f,-9.02486518e-02f,+2.19190374e-01f,
-1.44974992e-01f,+2.75046825e-01f,-3.80825251e-02f,-4.89908457e-02f,+2.58296818e-01f,
+2.97722787e-01f,-8.26253835e-03f,-3.02680820e-01f,+4.79040854e-02f,-2.65964009e-02f,
-2.18523100e-01f,-2.14358270e-01f,-1.04427412e-01f,-6.60725683e-03f,-2.71327466e-01f,
-3.30487452e-02f,+2.53819525e-01f,-2.13796988e-01f,+1.15810834e-01f,-3.07229301e-03f,
+2.24305481e-01f,-1.98589548e-01f,-2.48572886e-01f,-7.32347444e-02f,-2.06264168e-01f,
-2.13888530e-02f,-1.42311111e-01f,-2.32913494e-01f,-2.44020388e-01f,-1.13609083e-01f,
-7.66494870e-02f,-5.30320071e-02f,+9.75397788e-03f,+2.66093761e-01f,-1.11949816e-02f,
-2.51800507e-01f,+3.18154842e-02f,+1.78372532e-01f,+2.22332194e-01f,+8.18792135e-02f,
-5.47236986e-02f,-8.63854662e-02f,-1.19191699e-01f,+9.46972296e-02f,-1.52171373e-01f,
-2.50786334e-01f,+8.74059051e-02f,-2.64155924e-01f,+1.56924009e-01f,-2.50812322e-01f,
+6.52002916e-02f,-1.24717541e-01f,+6.85858130e-02f,+2.77021408e-01f,+6.85901120e-02f,
+1.03812769e-01f,-1.28597751e-01f,+2.23298758e-01f,+1.80284992e-01f,-1.17734380e-01f,
+1.66310623e-01f,+1.17488183e-01f,+2.61012949e-02f,+2.36517221e-01f,+3.00064474e-01f,
-5.29672876e-02f,+1.85388148e-01f,+1.85451269e-01f,+6.03686906e-02f,-8.69278610e-02f,
-4.94117499e-04f,+6.01547249e-02f,-1.60302103e-01f,+1.00897804e-01f,+7.96888173e-02f,
-6.37169033e-02f,-4.49072123e-02f,+5.16429991e-02f,+3.07036210e-02f,-5.63213602e-02f,
+4.22247052e-02f,+1.58425629e-01f,+3.54249358e-01f,+1.07165329e-01f,-3.39241087e-01f,
+6.11866685e-03f,-1.62814334e-01f,+3.42018455e-01f,+3.93133909e-01f,+1.29334018e-01f,
+9.89826024e-02f,-1.81820750e-01f,+1.85288161e-01f,-2.15244666e-01f,-2.63851643e-01f,
-2.59932429e-01f,+2.32169610e-02f,+2.17319086e-01f,-2.78311819e-01f,+7.88911059e-02f,
+7.89041221e-02f,+6.32411316e-02f,-4.73952182e-02f,-5.97209521e-02f,+1.81017384e-01f,
-8.59537348e-02f,-1.23824038e-01f,+1.68666482e-01f,-9.86963362e-02f,-2.23856151e-01f,
+2.22530901e-01f,-1.44301534e-01f,-2.04938650e-01f,+1.24366552e-01f,+8.43116641e-03f,
-1.70843393e-01f,+1.59811616e-01f,+5.41622564e-02f,+3.52435112e-02f,-5.45983911e-02f,
-3.08539063e-01f,+1.43180013e-01f,-2.42181048e-02f,+1.57697156e-01f,-1.40064716e-01f,
-1.93050027e-01f,+8.41359422e-02f,-9.98349953e-03f,-2.10488707e-01f,+1.45034552e-01f,
+1.60709351e-01f,-4.16684560e-02f,-5.97614981e-02f,+1.12785965e-01f,+3.27972621e-02f,
+6.84362277e-02f,-1.60809085e-02f,+3.10072787e-02f,-1.71160802e-01f,-1.10734235e-02f,
-3.07877129e-03f,-1.60385266e-01f,+1.42130286e-01f,-1.52922109e-01f,-1.86704323e-01f,
-1.50830057e-02f,+2.96842512e-02f,+8.05108324e-02f,+1.55469328e-01f,-2.02402666e-01f,
-1.63120568e-01f,+2.18300954e-01f,+1.63106099e-01f,+5.96001558e-02f,-2.06109747e-01f,
+3.33879888e-01f,+1.78619444e-01f,-1.33469254e-01f,+1.34113610e-01f,-1.92238435e-01f,
-1.71639621e-01f,-2.30204746e-01f,+1.78840920e-01f,+3.97727154e-02f,+1.00545704e-01f,
-2.02132255e-01f,+1.16031617e-01f,-2.21213758e-01f,-1.01441063e-01f,+2.15686299e-02f,
+3.06368768e-02f,-1.54323325e-01f,+1.69151146e-02f,+1.61417246e-01f,-8.32709968e-02f,
-2.15405390e-01f,+4.82543185e-02f,-3.99462134e-02f,+2.50156462e-01f,+1.93331644e-01f,
+1.78993359e-01f,-3.51483449e-02f,+6.75718710e-02f,+2.27215245e-01f,+1.44515634e-01f,
+6.26864508e-02f,-1.56252429e-01f,+1.59295678e-01f,+2.81293020e-02f,+2.20958013e-02f,
+1.38386652e-01f,-4.58537880e-03f,+1.20059252e-02f,-6.27796948e-02f,-5.49143851e-02f,
-1.35085553e-01f,-7.85492808e-02f,+2.11191222e-01f,+1.13953859e-01f,-4.24547568e-02f,
+1.45705611e-01f,+1.05222918e-01f,-2.11790577e-01f,-1.68073401e-01f,-1.48437366e-01f,
-2.16353759e-01f,-1.83548138e-01f,-1.54389367e-01f,+1.75946727e-01f,+1.63087294e-01f,
+1.40513806e-02f,+1.89630780e-02f,+1.51097685e-01f,-7.99828246e-02f,+1.65821556e-02f,
-1.31029010e-01f,-1.32158145e-01f,-3.40111256e-02f,-7.89516140e-03f,-9.22475606e-02f,
-2.08053142e-01f,-2.14629173e-01f,+1.06544450e-01f,+9.62592438e-02f,-1.69578254e-01f,
-5.62722385e-02f,-9.31082964e-02f,-2.06346616e-01f,-2.13608354e-01f,+1.43246040e-01f,
-2.10208952e-01f,-8.96173716e-02f,-2.89336052e-02f,-3.51058468e-02f,-2.29456306e-01f,
+1.81782126e-01f,-1.74247697e-01f,+2.58367695e-02f,-1.40729517e-01f,+2.99860518e-02f,
+1.46431118e-01f,+2.29494907e-02f,+6.36453032e-02f,+5.10661900e-02f,+1.59507290e-01f,
-9.29764286e-02f,+1.40203848e-01f,-1.65167272e-01f,-1.46865146e-02f,+1.06008396e-01f,
+9.16316882e-02f,+1.83892354e-01f,+7.78676346e-02f,+2.31556550e-01f,+4.67818156e-02f,
+1.25273988e-01f,-5.91092482e-02f,+2.31086969e-01f,-2.18819231e-01f,+1.82399780e-01f,
-1.03620425e-01f,+1.44868493e-01f,+2.12903619e-01f,-2.29030445e-01f,+7.74749815e-02f,
-8.84438902e-02f,+5.68567216e-02f,-1.21855326e-01f,-6.16264120e-02f,+2.14428306e-01f,
-1.53763622e-01f,+1.42694816e-01f,+5.34804165e-02f,+2.22259223e-01f,-2.13191748e-01f,
-1.54344305e-01f,-6.84091449e-02f,-1.23110987e-01f,+1.88022450e-01f,+1.74271166e-01f,
+3.32889380e-03f,-6.09529503e-02f,-1.68275028e-01f,+3.85165401e-02f,-1.30264431e-01f,
-2.02971116e-01f,+1.90284461e-01f,+5.20223081e-02f,+2.18755230e-01f,+2.07741603e-01f,
+2.12159663e-01f,-8.90794024e-03f,+7.08328187e-02f,+3.96206677e-02f,-1.66698724e-01f,
-6.06340431e-02f,+5.53875323e-03f,+1.01110317e-01f,-3.51841114e-02f,+1.36983886e-01f,
-1.93384692e-01f,+2.24157110e-01f,-1.89159736e-01f,+1.76445633e-01f,+4.87068184e-02f,
+7.49565242e-03f,+1.06640287e-01f,-2.11244628e-01f,-2.11068764e-02f,+1.09424748e-01f,
-1.79727644e-01f,+1.62234157e-01f,-6.13050051e-02f,+2.26155147e-01f,+1.12249203e-01f,
+1.40290067e-01f,+2.05375731e-01f,-5.73283769e-02f,+1.31304622e-01f,+1.06943145e-01f,
+2.05200613e-01f,-2.67599933e-02f,-1.09577030e-01f,-1.25520565e-02f,-1.05490044e-01f,
+2.10619062e-01f,-5.66623062e-02f,-1.92869768e-01f,-1.92852363e-01f,+8.52534994e-02f,
-2.11657852e-01f,-1.55525595e-01f,-1.50852099e-01f,-2.92605758e-02f,-1.33598194e-01f,
+6.06548898e-02f,+1.52009428e-01f,-1.50194224e-02f,+2.14522347e-01f,-8.97507817e-02f,
+8.66058394e-02f,-7.38769546e-02f,+1.64500222e-01f,+3.58373113e-02f,-1.08351789e-01f,
-8.88946429e-02f,-4.76693213e-02f,+1.68192148e-01f,+8.18642378e-02f,+2.22438246e-01f,
-2.25322112e-03f,-1.58416018e-01f,-1.62377670e-01f,-2.31170446e-01f,-1.79118603e-01f,
+2.02417880e-01f,-1.76555544e-01f,-2.98702214e-02f,-8.46975297e-02f,+1.68499798e-01f,
-7.07638785e-02f,-1.46490514e-01f,+1.36908978e-01f,+1.80061068e-02f,+2.25743324e-01f,
+1.16608925e-01f,-4.91728485e-02f,-1.52343199e-01f,-4.93800044e-02f,-7.35459197e-03f,
-1.21655921e-03f,+1.07785866e-01f,-1.74666226e-01f,-7.62320980e-02f,-1.27468958e-01f,
+1.07748859e-01f,-1.67689368e-01f,+1.36205092e-01f,-3.54979634e-01f,+4.27090796e-03f,
-3.73998694e-02f,-2.94120193e-01f,-1.55806527e-01f,-3.70325297e-01f,-1.31636024e-01f,
+2.67237946e-02f,+5.69662564e-02f,+3.49991098e-02f,+3.40535164e-01f,+1.73873305e-01f,
+1.29868463e-01f,+4.13174987e-01f,-2.90264249e-01f,+5.29708117e-02f,-1.37293071e-01f,
-9.31546986e-02f,-1.28320575e-01f,-7.32723251e-02f,-9.99426544e-02f,+3.44273895e-01f,
+4.06575948e-01f,+6.60284385e-02f,+4.07063439e-02f,+5.26945293e-02f,+9.65141691e-03f,
-2.40876228e-01f,+2.74992049e-01f,+1.39297411e-01f,-1.08399577e-01f,-2.23685473e-01f,
+1.36552483e-01f,+1.75136074e-01f,+3.48596483e-01f,+1.85056120e-01f,-4.03338879e-01f,
-3.13063622e-01f,+2.53982872e-01f,+8.59386623e-02f,+4.19870555e-01f,+6.08369522e-02f,
-3.65622967e-01f,+2.11166769e-01f,-3.17678452e-02f,-4.48520966e-02f,+1.67323470e-01f,
+1.09759226e-01f,+3.96812744e-02f,-3.11013386e-02f,+6.68999031e-02f,+1.11196339e-02f,
-3.02237332e-01f,-6.38094991e-02f,-2.19473451e-01f,+1.26711056e-01f,-1.68856367e-01f,
-1.96062014e-01f,+1.13810286e-01f,-9.55493525e-02f,+3.66859525e-01f,-1.69945449e-01f,
-8.36890936e-03f,-4.66156751e-01f,-7.25572556e-02f,-5.10776276e-03f,+1.07147127e-01f,
+1.82352290e-01f,-2.02801600e-01f,-3.01696479e-01f,-1.03329524e-01f,-2.30354071e-01f,
+2.14244217e-01f,+1.68554783e-02f,+9.97447744e-02f,+2.57366747e-01f,-2.00346828e-01f,
-1.48085639e-01f,-4.48092073e-01f,+2.47469202e-01f,+9.50367153e-02f,-1.48304552e-01f,
-3.21608186e-02f,+5.66977169e-03f,+1.83250621e-01f,+6.04457594e-02f,+1.11491621e-01f,
+1.67723015e-01f,-9.60378442e-04f,-1.23444594e-01f,-1.10553950e-01f,+4.08643857e-02f,
-5.30328490e-02f,+4.43258956e-02f,-7.40314052e-02f,+1.49805605e-01f,-8.58990923e-02f,
-1.87257186e-01f,+1.69371501e-01f,-2.43784338e-01f,-2.82954741e-02f,-6.24724738e-02f,
+1.05161324e-01f,+2.26181284e-01f,+2.19022676e-01f,+1.94375932e-01f,-2.15594843e-02f,
+1.73753992e-01f,+5.83332591e-02f,-2.07563758e-01f,+1.91710562e-01f,+9.24744457e-02f,
-8.80415440e-02f,+2.35960916e-01f,+6.21050298e-02f,+2.30684340e-01f,-2.03428105e-01f,
-1.40921488e-01f,+7.31906071e-02f,-2.13934541e-01f,-1.60384789e-01f,+2.71054506e-01f,
+1.60135269e-01f,+1.63089573e-01f,+4.99416515e-02f,+2.57052213e-01f,+1.41382664e-01f,
+1.18013665e-01f,-1.09014034e-01f,-1.89105660e-01f,+2.35654041e-01f,-8.78747646e-03f,
+1.62122667e-01f,-1.86802864e-01f,+2.00973958e-01f,+1.96189344e-01f,+2.63308495e-01f,
-3.59198973e-02f,+2.08268806e-01f,-6.90394416e-02f,-1.74866021e-01f,+2.18253255e-01f,
-1.88044593e-01f,-1.79631338e-01f,-8.39694664e-02f,-7.57557750e-02f,+1.25212222e-01f,
+1.67656556e-01f,-9.83409304e-03f,-9.27483961e-02f,-1.20877340e-01f,-2.27168873e-01f,
-2.08490863e-01f,+2.45490193e-01f,-1.06248604e-02f,+1.01366863e-01f,-2.11628929e-01f,
+1.59684882e-01f,+3.78300920e-02f,+1.99378029e-01f,-7.99375474e-02f,-4.46556509e-02f,
+3.82344685e-02f,-4.31808792e-02f,-2.28405505e-01f,-3.57682630e-02f,+9.38072577e-02f,
-7.63423070e-02f,+2.90031075e-01f,-1.95779130e-01f,+2.37900689e-01f,+2.93328017e-01f,
+2.14692309e-01f,+1.76726431e-01f,+1.49939090e-01f,+6.65269271e-02f,+2.23957792e-01f,
-3.01583409e-01f,+2.98568625e-02f,-2.53357470e-01f,+1.47595340e-02f,-1.94027558e-01f,
+2.41610259e-01f,+4.94481735e-02f,-2.86452286e-02f,-1.01351067e-01f,-1.34600118e-01f,
-8.26244652e-02f,+2.45458648e-01f,-2.16446947e-02f,-1.57944672e-02f,+1.56709969e-01f,
-2.02096760e-01f,+3.52139538e-03f,-1.07966088e-01f,+3.12711626e-01f,-1.12788238e-01f,
+1.90222532e-01f,+1.32235885e-03f,+1.62753433e-01f,+2.22402602e-01f,-3.20719033e-02f,
-2.06905246e-01f,+2.32282922e-01f,+2.29706332e-01f,+6.85567707e-02f,-1.17963463e-01f,
+3.68829332e-02f,-1.40423819e-01f,+1.31862298e-01f,+1.12282552e-01f,-7.84564987e-02f,
-3.97626869e-02f,+4.05035317e-02f,-1.49635702e-01f,-8.93691182e-02f,-1.30525827e-01f,
-2.35585291e-02f,-6.02028854e-02f,+1.28415465e-01f,+2.61219274e-02f,+1.86195940e-01f,
+1.83946062e-02f,+2.79652238e-01f,+2.51467288e-01f,+3.35969515e-02f,-3.33255142e-01f,
-1.21118598e-01f,-4.11921553e-03f,+7.39257336e-02f,+1.01120114e-01f,+3.06764513e-01f,
+2.91219741e-01f,-2.78745651e-01f,+2.28172094e-01f,+9.70358476e-02f,+2.23542437e-01f,
+3.21449004e-02f,+2.96188951e-01f,+2.22362682e-01f,-7.52084404e-02f,-2.31103182e-01f,
+3.43620032e-03f,+7.99390376e-02f,+5.56054264e-02f,+3.45655590e-01f,+3.16654205e-01f,
-2.99685657e-01f,-1.59320742e-01f,-1.05648912e-01f,-4.01731171e-02f,+9.56992581e-02f,
+1.01586230e-01f,-4.89677489e-02f,-1.57634690e-01f,-2.64470223e-02f,-9.17144343e-02f,
-2.09243193e-01f,+2.39750240e-02f,-1.52559310e-01f,-1.43861711e-01f,-7.43671656e-02f,
+7.13516027e-02f,-7.01995268e-02f,-1.89768136e-01f,-1.57142833e-01f,-1.23837136e-01f,
-2.71094264e-04f,-1.96969062e-01f,-3.54728103e-03f,+8.05652365e-02f,-1.43815860e-01f,
+1.96910892e-02f,-5.49266562e-02f,-1.20007463e-01f,+1.79485381e-01f,+3.52957658e-02f,
+5.48482612e-02f,+1.47071481e-01f,-5.77885732e-02f,-2.18389556e-02f,+1.94531545e-01f,
+6.53512180e-02f,+1.04249649e-01f,+1.11414284e-01f,-1.10026501e-01f,+5.09047620e-02f,
-5.77033684e-02f,-1.74771041e-01f,-9.81248096e-02f,+1.66996837e-01f,-1.21180072e-01f,
-2.87849233e-02f,-1.81144252e-01f,+2.24023789e-01f,-7.70090595e-02f,-2.27655783e-01f,
+1.69096217e-01f,-1.54223979e-01f,-1.27207771e-01f,-1.97798714e-01f,-2.16323975e-02f,
-8.33515748e-02f,-9.38773826e-02f,+1.41315237e-01f,+1.22615367e-01f,+1.31667957e-01f,
-6.92219660e-02f,+1.95404246e-01f,+1.40814945e-01f,-5.07578477e-02f,-9.78418514e-02f,
+1.09328711e-02f,-1.80675797e-02f,+5.19605167e-02f,+5.87473288e-02f,-1.31787926e-01f,
+7.16306642e-03f,-3.59539315e-02f,+7.13239908e-02f,+1.18260026e-01f,-2.43741758e-02f,
-1.14025556e-01f,-1.71007812e-01f,-3.63904387e-02f,+1.71308890e-01f,+2.74368227e-02f,
+1.56044200e-01f,+1.13240212e-01f,+1.01705462e-01f,+4.34540585e-02f,-7.97539279e-02f,
+5.57904691e-02f,+5.01743183e-02f,-2.42513552e-01f,-2.38858894e-01f,+3.30952317e-01f,
-1.30405337e-01f,-2.15145070e-02f,+1.67388663e-01f,+3.48716617e-01f,-9.74684879e-02f,
+2.72189885e-01f,+7.24113733e-02f,-5.70165887e-02f,-3.14443499e-01f,-1.25032932e-01f,
-1.58457346e-02f,+1.29082084e-01f,-1.11550048e-01f,+1.47611067e-01f,-9.14382786e-02f,
-1.97537746e-02f,-7.56700411e-02f,-3.12588274e-01f,-2.20648691e-01f,+2.31991678e-01f,
+1.41520379e-02f,-9.94832255e-03f,-1.18329607e-01f,+1.52071128e-02f,-1.31281883e-01f,
+6.77950606e-02f,+1.87946826e-01f,-1.08929433e-01f,+3.33196670e-01f,-5.72679937e-02f,
+1.42445669e-01f,-1.08434118e-01f,-1.27739146e-01f,+9.57017988e-02f,+2.30103329e-01f,
+1.44470409e-01f,+2.41830498e-01f,+1.36856198e-01f,+7.51028880e-02f,+3.64069231e-02f,
+5.62586300e-02f,-2.03723051e-02f,-8.73411149e-02f,-1.90879941e-01f,+6.36343583e-02f,
-2.01717749e-01f,-1.91384062e-01f,+3.74036543e-02f,+5.52437901e-02f,-1.09263197e-01f,
+2.09625632e-01f,+1.89012557e-01f,-1.32697493e-01f,+2.60899633e-01f,-3.32796946e-02f,
+1.32302746e-01f,+1.30363151e-01f,+5.17303236e-02f,+2.44212225e-01f,-5.59199676e-02f,
-2.68183816e-02f,+7.48073822e-03f,+3.31671834e-01f,+2.34671578e-01f,-2.74223715e-01f,
-1.01426117e-01f,-1.15739219e-01f,-7.89694861e-02f,+1.84136510e-01f,+3.17403495e-01f,
+2.36976191e-01f,-1.17221378e-01f,+6.54015914e-02f,-1.82967991e-01f,-3.89224431e-03f,
+2.40995422e-01f,+2.76921421e-01f,+3.50246787e-01f,-2.66401261e-01f,-1.90072671e-01f,
+2.18527421e-01f,+3.80899698e-01f,+4.43672210e-01f,-3.71502601e-02f,+2.07009554e-01f,
-1.48436010e-01f,-3.02739888e-01f,+4.05990593e-02f,-6.09851740e-02f,+1.36985824e-01f,
-2.94786543e-01f,+2.22369120e-01f,-8.86851624e-02f,+3.24583709e-01f,+3.96523744e-01f,
+2.81635076e-01f,+2.78102066e-02f,-3.09657872e-01f,+3.64883155e-01f,-2.57604476e-02f,
-3.98775190e-02f,+1.40427515e-01f,+1.44005403e-01f,-2.16192856e-01f,+4.32163209e-01f,
+2.98201263e-01f,-9.96008515e-04f,+1.88166738e-01f,-1.76825747e-01f,-1.79537728e-01f,
+5.71605004e-02f,+1.54490963e-01f,+3.44585866e-01f,+9.86635387e-02f,-1.27069950e-01f,
-1.07054867e-01f,-1.43178836e-01f,+3.02810460e-01f,-1.78284988e-01f,+7.15103559e-03f,
-2.07711443e-01f,+9.96544585e-02f,-1.45206615e-01f,+1.10790230e-01f,+2.68539935e-01f,
-6.03703335e-02f,+3.14618707e-01f,+8.48391056e-02f,+1.86522037e-01f,+4.63202447e-01f,
+3.45001996e-01f,+3.03932250e-01f,-2.53602207e-01f,+2.06192248e-02f,-2.31166527e-01f,
-1.67083949e-01f,-2.55520809e-02f,+1.57243256e-02f,+5.02598047e-01f,-2.09977791e-01f,
-9.92550999e-02f,+1.61777467e-01f,+1.48791283e-01f,+3.94028723e-01f,-1.64264306e-01f,
+7.63591826e-02f,-3.15958890e-03f,-7.52666220e-02f,+3.86884421e-01f,+2.07079500e-01f,
+2.18686059e-01f,-1.27969831e-02f,-2.58429557e-01f,+3.30584586e-01f,-1.81869328e-01f,
-2.16628581e-01f,-1.02443874e-01f,+2.13592142e-01f,+3.12198192e-01f,-2.79299587e-01f,
-1.88040763e-01f,-2.10417643e-01f,+1.41756356e-01f,-2.01512456e-01f,-2.11280540e-01f,
+4.05653827e-02f,-1.27995327e-01f,+1.02067731e-01f,-1.82150587e-01f,+2.36591160e-01f,
+2.04781778e-02f,-3.04122958e-02f,+1.99721456e-01f,+1.57464132e-01f,+7.80661702e-02f,
+2.27766797e-01f,-3.22048604e-01f,+6.75694197e-02f,-1.27836391e-01f,+1.24964587e-01f,
-3.74272242e-02f,-8.42342004e-02f,-1.04714446e-01f,-1.78335354e-01f,-1.46697521e-01f,
-2.78990660e-02f,-1.24817260e-01f,+2.09951147e-01f,+8.92226174e-02f,-1.61688626e-01f,
+1.21036030e-01f,+1.92086503e-01f,-1.54420957e-01f,-2.22272396e-01f,-3.58803757e-02f,
+1.25716761e-01f,+1.16403885e-01f,-1.17136672e-01f,+2.57435173e-01f,-1.69700027e-01f,
-1.75844058e-01f,+1.15582095e-02f,-1.28439739e-01f,+2.60561705e-01f,+2.89981961e-01f,
-1.49554417e-01f,+2.16890909e-02f,-9.18797329e-02f,+1.13182943e-02f,+4.32766639e-02f,
+2.96114404e-02f,+1.99607518e-02f,-2.20194489e-01f,-4.66335798e-03f,-9.20910537e-02f,
+8.41508657e-02f,-8.91183391e-02f,+9.14922729e-02f,-3.96235734e-02f,+9.62245166e-02f,
+2.58347303e-01f,-9.20071602e-02f,-2.93128123e-03f,+4.87957262e-02f,+1.45612806e-01f,
-1.06383279e-01f,+1.67682141e-01f,-3.27447727e-02f,-1.97034851e-02f,-2.13150308e-02f,
-1.44998610e-01f,+2.39556760e-01f,+1.28926978e-01f,+4.24819030e-02f,-1.01286873e-01f,
+3.17590013e-02f,+1.96451500e-01f,+2.22116932e-01f,-3.02181840e-02f,-1.66000664e-01f,
+1.47148475e-01f,-1.16048016e-01f,+7.32651874e-02f,+1.88767567e-01f,+9.04662162e-02f,
+1.56310275e-01f,+8.50083008e-02f,+7.38978982e-02f,-1.79187670e-01f,-1.50899574e-01f,
-1.85766499e-02f,+1.98587507e-01f,-2.59432435e-01f,+2.31656522e-01f,-6.17682189e-02f,
+1.66224137e-01f,+6.85170889e-02f,-5.04812784e-02f,+1.36662781e-01f,+2.14249939e-01f,
-1.48233548e-01f,-7.12724030e-02f,-1.06932983e-01f,+1.16253823e-01f,+4.12329398e-02f,
+2.86749780e-01f,-1.89118281e-01f,+7.66870454e-02f,+2.07132861e-01f,-1.46581069e-01f,
-8.82443488e-02f,+3.08016181e-01f,+5.78854233e-02f,-2.88348228e-01f,+2.19478309e-01f,
-2.25047871e-01f,+1.48484543e-01f,-5.25060706e-02f,+3.42770785e-01f,+1.91435236e-02f,
+3.20205450e-01f,-1.00297302e-01f,+3.14908594e-01f,-1.52908713e-02f,+2.03573436e-01f,
+1.59107354e-02f,+7.83115346e-03f,+3.11468571e-01f,+9.53066200e-02f,-1.12774156e-01f,
+3.04240078e-01f,+5.12108989e-02f,-3.39278609e-01f,+4.42756936e-02f,-1.27220899e-01f,
-1.42864868e-01f,-9.08024088e-02f,-9.73822773e-02f,+2.32478101e-02f,-1.60032362e-01f,
+2.23093331e-02f,+1.10846413e-02f,-5.78303561e-02f,+1.99149415e-01f,-1.86854508e-02f,
+2.81724751e-01f,+2.22862512e-01f,+1.82718396e-01f,-1.02260724e-01f,-2.16947421e-01f,
+2.10703045e-01f,-9.92659032e-02f,-3.05867083e-02f,-7.62158632e-02f,+2.46553451e-01f,
-5.13186976e-02f,-2.31068842e-02f,+1.39237568e-01f,-1.32122546e-01f,-3.29202786e-02f,
+3.32402349e-01f,+1.99701980e-01f,+2.60910988e-01f,+1.87687159e-01f,-1.74724624e-01f,
+1.78659275e-01f,-5.44341356e-02f,+2.15590730e-01f,+2.76396304e-01f,+2.29234800e-01f,
+2.82437447e-02f,-3.97419445e-02f,-6.66918010e-02f,+9.12778899e-02f,+9.78678092e-02f,
+1.22661151e-01f,+1.95098057e-01f,-7.57695213e-02f,+1.47111610e-01f,+3.42220091e-03f,
-1.70853406e-01f,+1.77871928e-01f,+1.86411038e-01f,-1.69611186e-01f,-2.06626114e-02f,
+2.29458690e-01f,-1.20638631e-01f,+5.55054136e-02f,-2.05828324e-01f,+1.76939771e-01f,
-4.74530123e-02f,-1.43433124e-01f,+1.49943218e-01f,-1.62592471e-01f,+1.33011445e-01f,
+8.16524848e-02f,-1.74661702e-03f,+1.50662974e-01f,+2.37364203e-01f,+2.39803120e-01f,
+4.04039919e-02f,-2.01638713e-01f,+1.56094909e-01f,-8.06347525e-04f,-9.12696198e-02f,
-2.27111995e-01f,-6.78901523e-02f,+4.93467748e-02f,-9.21713188e-02f,-1.25326812e-01f,
-8.12316686e-02f,+5.84092848e-02f,+2.34139591e-01f,+5.14970673e-03f,-6.57877028e-02f,
+2.11648017e-01f,+1.72358900e-01f,+1.25662595e-01f,+2.60545760e-02f,+1.95332140e-01f,
-6.59310892e-02f,+2.80414194e-01f,-8.36311728e-02f,+6.85097873e-02f,-7.56704286e-02f,
+1.09431125e-01f,-2.22765103e-01f,-1.45346835e-01f,+1.54545784e-01f,+3.81381856e-03f,
+2.70639539e-01f,-1.13324843e-01f,-8.90415832e-02f,+8.12772363e-02f,+2.40561411e-01f,
+3.70260999e-02f,-1.54928118e-01f,+2.48093735e-02f,-1.98838085e-01f,+1.46560267e-01f,
+2.12549627e-01f,-4.33695950e-02f,-1.66606382e-01f,-1.35741457e-01f,+1.13213211e-01f,
+1.62994802e-01f,+1.72249511e-01f,-1.53290033e-02f,-1.83491632e-01f,+1.84068590e-01f,
+8.79727378e-02f,-9.62054059e-02f,+2.26734832e-01f,+4.46427092e-02f,+1.79773912e-01f,
-2.23673344e-01f,-5.69733419e-02f,+2.56256908e-02f,+1.77560747e-01f,-1.91241577e-01f,
+2.76339620e-01f,-1.88156843e-01f,-1.85405701e-01f,-2.01090291e-01f,-1.56171709e-01f,
-6.82007968e-02f,+2.80041201e-03f,-9.85283479e-02f,-2.65002530e-02f,+6.25633597e-02f,
+1.94099620e-01f,+2.42790312e-01f,+1.87514618e-01f,+7.71775469e-02f,+1.00039676e-01f,
+1.90550908e-01f,-2.32676819e-01f,-7.28659853e-02f,+1.29992783e-01f,-2.16866523e-01f,
+2.66290545e-01f,+1.98963746e-01f,-1.35641813e-01f,+1.22865945e-01f,-1.32618785e-01f,
-1.51421025e-01f,-4.21288274e-02f,+1.00182489e-01f,-2.14844123e-01f,-1.70839012e-01f,
-1.03951789e-01f,-2.20124483e-01f,+2.32269503e-02f,+8.54655504e-02f,-1.44032985e-01f,
-1.54915720e-01f,-1.07563280e-01f,+1.19644828e-01f,-2.35221237e-01f,+2.40847528e-01f,
+4.28779982e-02f,+6.62206253e-03f,+2.19555020e-01f,-2.10232213e-01f,+2.17321411e-01f,
+2.00378135e-01f,+2.53684342e-01f,+1.71503961e-01f,-2.19015643e-01f,+1.47638083e-01f,
-6.18155152e-02f,-1.10084936e-01f,-7.03565180e-02f,-2.40397573e-01f,+2.52441894e-02f,
+1.68595359e-01f,+5.14166392e-02f,+7.38448789e-03f,+1.45896122e-01f,+6.84364587e-02f,
-1.08917728e-01f,-7.13305771e-02f,-6.98075965e-02f,-1.74959376e-01f,+1.91379741e-01f,
-1.58901677e-01f,-1.82695359e-01f,-2.07731664e-01f,-2.06274509e-01f,+4.74764183e-02f,
+1.09631181e-01f,-1.02251887e-01f,+7.12082461e-02f,+2.47939736e-01f,+2.56405264e-01f,
-4.08915281e-02f,-1.88417226e-01f,-6.34018779e-02f,+1.89306214e-02f,-1.82907447e-01f,
+2.62232691e-01f,+3.67484987e-01f,+3.83942336e-01f,-6.67487681e-02f,+3.07065964e-01f,
-1.52159825e-01f,-3.82371098e-01f,-1.09598882e-01f,-1.15821578e-01f,-1.12047732e-01f,
-4.53294627e-02f,-1.34427994e-01f,-6.24257147e-01f,+4.15741533e-01f,+1.48619682e-01f,
+4.35855836e-01f,+2.63051420e-01f,-1.75233305e-01f,+3.00236523e-01f,-1.99414596e-01f,
-2.19145954e-01f,+2.31831409e-02f,+1.97700083e-01f,-2.23926142e-01f,+3.33896041e-01f,
+2.10670710e-01f,-9.55060869e-02f,+2.56361663e-01f,+1.67978257e-02f,-8.39504376e-02f,
-1.79721564e-01f,+1.33527547e-01f,-9.07606706e-02f,-1.33343041e-01f,-1.04141913e-01f,
-1.35213360e-01f,-8.78536925e-02f,+2.21134216e-01f,-1.07477009e-02f,-3.13287348e-01f,
-3.17009240e-01f,+5.87236583e-01f,-8.56769532e-02f,+1.77935332e-01f,+1.84941962e-01f,
+1.07434141e-02f,+9.26014259e-02f,-1.84684470e-01f,+5.23523271e-01f,+1.29706874e-01f,
+4.88688126e-02f,+4.35859263e-01f,+2.34460235e-02f,+3.22260588e-01f,-4.26895916e-02f,
-1.74080685e-01f,-3.10800552e-01f,-1.96227401e-01f,+2.22705349e-01f,-3.37479472e-01f,
+2.19303705e-02f,+2.21115023e-01f,+1.82506293e-01f,+4.77053404e-01f,+4.13841270e-02f,
-1.25886410e-01f,-3.32509667e-01f,-1.69472292e-01f,+3.11856270e-01f,-1.42451018e-01f,
+8.75313282e-02f,+2.97529064e-02f,-1.75443828e-01f,+9.33223665e-02f,-3.51936162e-01f,
-1.13204040e-01f,+3.40815485e-02f,+4.62019324e-01f,+3.49636704e-01f,-3.16501141e-01f,
-4.05618936e-01f,-2.30466351e-01f,+5.54191228e-03f,-3.22024934e-02f,+1.41585320e-01f,
+2.71732956e-01f,+4.00320701e-02f,+1.66748703e-01f,-3.30220954e-03f,+2.01309547e-01f,
-4.64206822e-02f,-4.80014607e-02f,-5.88664375e-02f,+7.15890005e-02f,+6.05635904e-02f,
+3.51438165e-01f,-5.61804831e-01f,-1.08032569e-01f,+2.22567767e-01f,-7.44063929e-02f,
-2.15965003e-01f,-2.50628013e-02f,-6.37880042e-02f,-8.23026448e-02f,-3.76642831e-02f,
-2.43933663e-01f,+3.79502088e-01f,+6.55570403e-02f,+1.90040752e-01f,+3.28879221e-03f,
+1.40308794e-02f,-1.94038361e-01f,-5.91379292e-02f,-1.53695747e-01f,+1.48990322e-02f,
+5.36440499e-02f,+2.57436097e-01f,-5.96806407e-03f,-2.68084109e-02f,-7.17163831e-02f,
+1.04217783e-01f,+1.44335836e-01f,+1.05920233e-01f,-1.07723176e-02f,+2.51295388e-01f,
+2.07975283e-01f,+2.22002849e-01f,+1.34700209e-01f,+2.13811696e-01f,-6.74226135e-02f,
-1.81777179e-01f,+1.20532833e-01f,+1.62421674e-01f,+1.29677445e-01f,+1.69154312e-02f,
+4.81396988e-02f,-1.60314217e-01f,+2.38202140e-01f,+1.01133831e-01f,+1.39887378e-01f,
-5.11893556e-02f,+3.97692174e-02f,-1.62872709e-02f,-2.68963069e-01f,+2.89628021e-02f,
-6.30985945e-02f,+2.27795336e-02f,+2.63770550e-01f,-3.05802170e-02f,-9.44647342e-02f,
-2.24702045e-01f,-1.60673842e-01f,-1.61760315e-01f,+7.63796940e-02f,-1.47712335e-01f,
-4.06652205e-02f,+2.44109437e-01f,+1.23849630e-01f,+8.76731426e-02f,-2.21998781e-01f,
+7.25491941e-02f,+8.82338136e-02f,+1.19624346e-01f,-4.88860197e-02f,+2.44434848e-01f,
+1.20186366e-01f,-1.34414449e-01f,+8.39323327e-02f,+1.26527756e-01f,-2.99326591e-02f,
+2.26227894e-01f,-1.03747144e-01f,-3.39279175e-02f,+1.17560677e-01f,+1.37094229e-01f,
-1.30529240e-01f,-1.39144555e-01f,+1.46675274e-01f,-1.16979312e-02f,-1.40188098e-01f,
-1.21791981e-01f,-7.80080110e-02f,-1.65099800e-01f,-2.00345427e-01f,+6.00643232e-02f,
-9.17938799e-02f,-2.30100960e-01f,+1.26874179e-01f,-1.29916817e-01f,-1.24045648e-01f,
-6.93291351e-02f,+6.73686936e-02f,+2.62197684e-02f,+8.47741738e-02f,+4.33556624e-02f,
+2.67107654e-02f,+3.72073874e-02f,-1.74168795e-01f,-6.88369870e-02f,-1.90765217e-01f,
-1.00699827e-01f,+9.82479155e-02f,+2.01433137e-01f,+1.93798959e-01f,-1.72639061e-02f,
+4.68460768e-02f,-6.08659312e-02f,-1.79719657e-01f,+1.70968279e-01f,+2.08324865e-01f,
-1.14323534e-01f,-2.16266423e-01f,-1.97175667e-01f,-2.46351399e-02f,+1.79406568e-01f,
-5.92217073e-02f,-1.17719002e-01f,-1.01630129e-01f,+2.62856781e-02f,+5.80907501e-02f,
+1.42504364e-01f,+6.21659085e-02f,-1.51334941e-01f,+2.02124715e-02f,+2.19073445e-01f,
-8.62951726e-02f,-1.53871804e-01f,+1.82741672e-01f,-1.20715406e-02f,-8.24649036e-02f,
-1.94717288e-01f,-2.06643134e-01f,-1.56528130e-01f,-2.06502918e-02f,-7.97375590e-02f,
-1.19726285e-01f,+1.26780001e-02f,+9.82093215e-02f,+1.65124416e-01f,+9.19916779e-02f,
-2.80057192e-02f,+2.18351632e-01f,-1.40540954e-02f,-1.36883587e-01f,-5.29198982e-02f,
-6.88236877e-02f,-1.29229575e-01f,-9.33370888e-02f,-1.81315616e-01f,-2.65856832e-02f,
+1.76954150e-01f,-2.27850169e-01f,+1.52227178e-01f,-4.31660190e-02f,-2.06312627e-01f,
-1.08069718e-01f,-1.43058628e-01f,+3.58422138e-02f,+1.83459997e-01f,+5.24073169e-02f,
+2.03259915e-01f,-8.53660144e-03f,+1.18225172e-01f,+1.84849903e-01f,+5.38413040e-02f,
+1.50897190e-01f,+1.89120471e-01f,+1.20761804e-01f,+1.39710695e-01f,+2.09046140e-01f,
-2.35222697e-01f,-8.50548521e-02f,-2.28229016e-01f,-2.21448764e-01f,-6.58535808e-02f,
+1.95017412e-01f,-1.88669175e-01f,-1.47082493e-01f,+1.78896829e-01f,+9.65202749e-02f,
-1.30321816e-01f,+6.69198409e-02f,-1.51226893e-01f,-1.59769356e-01f,-7.85183981e-02f,
+4.45407033e-02f,-4.45281081e-02f,-1.36209637e-01f,-1.50094390e-01f,+5.06978296e-02f,
+2.34717682e-01f,+2.16620952e-01f,+3.89225222e-02f,-1.42224863e-01f,-1.74910948e-01f,
-2.63436399e-02f,+1.80185243e-01f,+1.92501292e-01f,+6.13920242e-02f,-5.29527627e-02f,
+1.31391406e-01f,-2.50921398e-02f,+1.24086246e-01f,+2.05622643e-01f,-2.04252094e-01f,
-1.85406700e-01f,-9.01404843e-02f,-1.82195246e-01f,+2.09000394e-01f,-1.68582182e-02f,
+5.89596219e-02f,-2.93308049e-01f,-6.43803775e-02f,-2.38177747e-01f,-4.63937409e-02f,
+4.85557541e-02f,+2.22690910e-01f,+1.74664602e-01f,+1.44750118e-01f,+5.04233986e-02f,
+1.86430380e-01f,-1.48072228e-01f,-5.77249229e-02f,-2.21410707e-01f,-2.15542857e-02f,
-5.93285263e-02f,+3.82766388e-02f,-1.70916602e-01f,+2.00292721e-01f,-1.14728339e-01f,
+3.45433652e-02f,-1.05847411e-01f,+2.41725236e-01f,-7.29043931e-02f,-1.07464775e-01f,
+1.42936349e-01f,-5.29865175e-03f,+4.65916656e-02f,+8.21182281e-02f,-8.24329723e-03f,
+6.08612150e-02f,+1.42280767e-02f,-2.39723742e-01f,-7.79095069e-02f,-1.94713861e-01f,
+4.67274114e-02f,-3.44862252e-01f,+9.26556438e-02f,-6.06487691e-01f,+2.41314307e-01f,
-2.70056352e-02f,+8.46001208e-02f,+1.19910084e-01f,-1.52757213e-01f,+3.17871958e-01f,
+8.55924413e-02f,-3.60048115e-02f,-7.13502914e-02f,+1.40773669e-01f,-3.51525724e-01f,
+4.32597190e-01f,+1.55558586e-01f,-8.17811936e-02f,+4.46681857e-01f,-9.06729847e-02f,
-2.52854936e-02f,-2.26250798e-01f,+3.15434337e-01f,+2.33655110e-01f,-1.61107957e-01f,
-2.24164985e-02f,-5.54213524e-02f,-1.27900913e-01f,+3.33099952e-03f,+1.18725218e-01f,
-1.94705755e-01f,-1.34453624e-01f,+3.87414366e-01f,+7.39700336e-04f,-5.68127818e-03f,
+3.52389634e-01f,-1.23519368e-01f,+1.26366124e-01f,+1.63426131e-01f,+2.30582118e-01f,
+2.97410727e-01f,+7.85682425e-02f,-9.47561860e-03f,+1.37268379e-01f,+1.67083040e-01f,
+4.59518693e-02f,-2.32038796e-01f,-3.24106306e-01f,-8.50029476e-03f,+5.52881122e-01f,
-1.50746554e-01f,+1.40976846e-01f,+1.07449673e-01f,+1.00736775e-01f,-2.31669880e-02f,
+1.30287051e-01f,-2.75307354e-02f,+1.43601954e-01f,-2.81017125e-01f,+3.56465816e-01f,
-2.13679969e-01f,-1.92691982e-01f,+2.22043112e-01f,-1.01780156e-02f,+1.68478385e-01f,
+6.05863221e-02f,-8.13398957e-02f,-4.78156544e-02f,+1.29619852e-01f,+1.07367456e-01f,
+7.09217191e-02f,-8.27572197e-02f,-8.32415894e-02f,+1.25921771e-01f,-8.51457119e-02f,
-1.52049139e-01f,-2.53301173e-01f,+4.26710993e-02f,-6.47894070e-02f,+6.13968819e-02f,
-1.97497979e-02f,+1.47976488e-01f,-1.60279885e-01f,+6.02929033e-02f,+1.73377171e-01f,
+1.24864191e-01f,-1.53416231e-01f,+1.76836580e-01f,+2.48180225e-01f,+9.61223617e-02f,
+2.02255607e-01f,+2.14091942e-01f,-1.81216612e-01f,+1.09832570e-01f,+5.43943094e-03f,
+1.68391645e-01f,+1.45419016e-01f,-8.06816481e-03f,+2.16341674e-01f,-3.39272767e-02f,
+1.20698534e-01f,-2.10024968e-01f,+7.65192788e-03f,-2.21742511e-01f,+1.17473319e-01f,
-4.05533649e-02f,+1.06188305e-01f,+8.95780027e-02f,+6.81961477e-02f,+1.44292131e-01f,
-2.13400915e-01f,+1.43656850e-01f,-8.12545940e-02f,+8.27571675e-02f,+2.36320496e-01f,
+1.07970066e-01f,+2.17770010e-01f,-2.03631613e-02f,+1.79298416e-01f,+7.38317445e-02f,
+8.68892595e-02f,+3.73599976e-02f,+6.26203492e-02f,+1.11866994e-02f,-1.67672053e-01f,
+2.23731682e-01f,-1.19340338e-01f,-2.74250959e-03f,-1.03308223e-02f,-2.19119370e-01f,
+3.25039402e-02f,-8.66991580e-02f,+1.13980398e-01f,+2.90086120e-01f,+8.52089226e-02f,
-8.05121213e-02f,+1.07051671e-01f,+2.59826123e-03f,+2.45497867e-01f,-1.44572392e-01f,
-5.94322234e-02f,+3.42578925e-02f,+1.38450131e-01f,+1.81811437e-01f,+5.98355308e-02f,
+1.15968242e-01f,+8.34338516e-02f,+4.33940776e-02f,+1.86738178e-01f,-1.65899679e-01f,
+8.17875564e-03f,-1.46568000e-01f,+1.67571127e-01f,-4.10984866e-02f,-9.53205675e-02f,
-1.64075926e-01f,-2.52330340e-02f,+6.29680529e-02f,-9.12197605e-02f,+1.15446925e-01f,
+1.45108057e-02f,+1.07094482e-01f,+2.45174825e-01f,-1.60153270e-01f,+1.28030181e-01f,
-1.09366126e-01f,+2.69950002e-01f,+2.09806845e-01f,+1.68630332e-01f,+1.82791084e-01f,
+2.48367488e-01f,-8.89545530e-02f,+5.75920232e-02f,-2.43103635e-02f,-1.75850391e-01f,
+1.26458555e-01f,+5.57473041e-02f,+8.39420334e-02f,-2.13703886e-01f,-2.07920223e-02f,
+1.35568932e-01f,-7.84213543e-02f,-5.03252186e-02f,+1.75310940e-01f,-3.92991304e-02f,
-2.01562077e-01f,+4.37337160e-02f,-9.81268510e-02f,-5.29443622e-02f,+8.98237601e-02f,
-6.33270293e-02f,+1.70537710e-01f,+4.52684760e-02f,+8.05896968e-02f,-6.44004196e-02f,
-8.65935460e-02f,-7.53716975e-02f,+1.38704211e-01f,+2.61820942e-01f,-9.56507921e-02f,
+2.22545877e-01f,-1.47405952e-01f,-1.94285184e-01f,-2.04415917e-01f,+2.28362471e-01f,
+1.79380059e-01f,-2.02521145e-01f,-1.53926402e-01f,+1.05864190e-01f,-8.11576471e-02f,
-8.42999667e-02f,-2.43009120e-01f,+7.71038830e-02f,-2.13867560e-01f,+1.43937573e-01f,
-8.76298398e-02f,-1.22788198e-01f,+3.18601370e-01f,-3.40456478e-02f,+2.16506600e-01f,
+4.50747758e-02f,+7.65663907e-02f,-4.73304838e-02f,+1.50284857e-01f,-1.83620751e-01f,
+2.23067373e-01f,+2.61849046e-01f,-5.31833470e-02f,-1.01774976e-01f,+1.81144521e-01f,
-1.21339470e-01f,+2.36901075e-01f,+1.31978333e-01f,+2.44880170e-01f,-8.20741057e-02f,
-6.50401562e-02f,+2.14449495e-01f,-2.14877352e-01f,+2.13904500e-01f,-1.16334736e-01f,
+2.21022330e-02f,-2.18018480e-02f,}; 
//k2c_tensor dense_29_kernel = {&dense_29_kernel_array[0],2,2352,{28,84, 1, 1, 1}};
dense_29_kernel.ndim=2;
dense_29_kernel.numel=2352;
dense_29_kernel.shape[0]=28;
dense_29_kernel.shape[1]=84;
dense_29_kernel.shape[2]=1;
dense_29_kernel.shape[3]=1;
dense_29_kernel.shape[4]=1;
for(i=0;i<2352;i++){
#pragma HLS unroll factor = 128
//#pragma HLS unroll
	dense_29_kernel.array[i] = dense_29_kernel_array[i];
}


float dense_29_bias_array[84] = {
-2.43278919e-03f,-1.65159896e-03f,+1.08317789e-02f,+1.87093504e-02f,+2.57995911e-02f,
+2.95694508e-02f,+5.76897301e-02f,+4.85385247e-02f,+7.94014409e-02f,+6.13810681e-02f,
+3.51901762e-02f,-1.74727812e-02f,+5.65436259e-02f,+6.48103952e-02f,+4.49051335e-02f,
+2.01927181e-02f,+5.45971235e-03f,+6.14728266e-03f,+8.96891858e-03f,-2.45481525e-02f,
-2.36361772e-02f,-8.52874946e-03f,+2.15716884e-02f,-1.82579085e-02f,+6.52557164e-02f,
+4.26459312e-02f,+1.56982094e-02f,+1.38440300e-02f,+2.70222947e-02f,-9.93269496e-03f,
+5.79241812e-02f,+3.84068936e-02f,+1.55116068e-02f,+5.25781251e-02f,+0.00000000e+00f,
+2.24114582e-02f,+0.00000000e+00f,-3.26992832e-02f,+1.22759761e-02f,+7.96034746e-03f,
+3.51505689e-02f,+7.73877129e-02f,+3.38794589e-02f,+2.82095764e-02f,+3.03368773e-02f,
+1.40505172e-02f,+2.48326641e-02f,-1.02498634e-02f,+5.54559659e-03f,+1.72173344e-02f,
+4.34525497e-02f,+3.60801853e-02f,-1.07756499e-02f,-9.52366833e-03f,-1.14756860e-02f,
-5.98963210e-03f,+4.94363084e-02f,+3.53661180e-02f,+5.87502643e-02f,+6.36239871e-02f,
+2.58285385e-02f,+5.93739375e-02f,-2.56359670e-02f,-2.85065938e-02f,+2.03762986e-02f,
+6.30164053e-03f,+8.12741555e-03f,+1.87951270e-02f,+4.83369417e-02f,-1.15095451e-02f,
+1.31340884e-02f,+2.85446388e-03f,+5.10678859e-03f,+3.07207555e-02f,+6.98086098e-02f,
+3.77341434e-02f,+0.00000000e+00f,-9.96750686e-03f,+2.15607006e-02f,+3.03066894e-02f,
+5.77021576e-02f,+5.96219636e-02f,+3.67991291e-02f,-1.70882344e-02f,}; 
//k2c_tensor dense_29_bias = {&dense_29_bias_array[0],1,84,{84, 1, 1, 1, 1}};
dense_29_bias.ndim=1;
dense_29_bias.numel=84;
dense_29_bias.shape[0]=84;
dense_29_bias.shape[1]=1;
dense_29_bias.shape[2]=1;
dense_29_bias.shape[3]=1;
dense_29_bias.shape[4]=1;
for(i=0;i<84;i++){
#pragma HLS unroll factor = 16
	dense_29_bias.array[i] = dense_29_bias_array[i];
}


float dense_29_fwork[2380] = {0}; 

 
float dense_30_output_array[70] = {0}; 
//k2c_tensor dense_30_output = {&dense_30_output_array[0],1,70,{70, 1, 1, 1, 1}};
dense_30_output.ndim=1;
dense_30_output.numel=70;
dense_30_output.shape[0]=70;
dense_30_output.shape[1]=1;
dense_30_output.shape[2]=1;
dense_30_output.shape[3]=1;
dense_30_output.shape[4]=1;
for(i=0;i<70;i++){
#pragma HLS unroll factor = 16
	dense_30_output.array[i] = dense_30_output_array[i];
}


float dense_30_kernel_array[5880] = {
+5.83634973e-02f,-1.96211278e-01f,+1.59569234e-01f,-1.43037260e-01f,+1.26073077e-01f,
+8.58703926e-02f,-1.07562363e-01f,-1.13243185e-01f,+9.21487883e-02f,-6.64476380e-02f,
+6.81003109e-02f,-1.07946776e-01f,-1.16577201e-01f,+7.62323961e-02f,+1.58991277e-01f,
+5.24759591e-02f,-1.22356474e-01f,-8.38190466e-02f,+1.84225053e-01f,-1.90153554e-01f,
+3.28737535e-02f,-7.90960416e-02f,-9.09294039e-02f,-2.44465172e-02f,-1.72353506e-01f,
+1.47209510e-01f,-1.84209794e-02f,-1.52660251e-01f,-8.43872577e-02f,+2.26679294e-05f,
+4.96650534e-03f,+1.39103502e-01f,+9.83197317e-02f,+1.86175078e-01f,+2.30064541e-02f,
-1.31263928e-02f,+1.36555642e-01f,+1.51589453e-01f,-1.85547918e-01f,-1.31867796e-01f,
-1.20590568e-01f,-2.43997406e-02f,+6.81751668e-02f,+7.10197352e-03f,+9.74035859e-02f,
-7.25682750e-02f,+9.84068960e-02f,-1.24303319e-01f,+1.16385549e-01f,+6.42102510e-02f,
+1.81149423e-01f,+9.03299004e-02f,+1.82708502e-01f,-1.38831556e-01f,+8.65910947e-02f,
-1.59710646e-02f,-7.43237208e-04f,+3.29089090e-02f,+6.46603703e-02f,-4.93435562e-02f,
-1.51068717e-01f,-1.69353932e-01f,+1.94272753e-02f,+1.06952146e-01f,-8.74579176e-02f,
+2.96334922e-03f,-8.69976357e-02f,+7.47734904e-02f,-1.45017296e-01f,-8.15520361e-02f,
-1.19119629e-01f,+2.09782384e-02f,+1.91989869e-01f,+1.17097542e-01f,-5.95340058e-02f,
-1.21630616e-02f,+1.65402934e-01f,+3.06075830e-02f,-3.39801535e-02f,-1.26679868e-01f,
-1.69914827e-01f,+1.49219707e-01f,-6.67647785e-03f,-1.02788642e-01f,+4.86049689e-02f,
-1.94807827e-01f,-7.92251676e-02f,-5.60464673e-02f,-5.82440570e-02f,+1.65789589e-01f,
-1.36533435e-02f,+1.70462593e-01f,+7.70711526e-02f,-3.39261144e-02f,+1.68995753e-01f,
+7.23106936e-02f,+1.56135127e-01f,-9.69293155e-03f,-1.95077181e-01f,+2.14911252e-02f,
-1.93551213e-01f,-2.36563254e-02f,-1.72038645e-01f,+7.56665245e-02f,-7.86064193e-02f,
+1.62234798e-01f,-5.27595170e-02f,+1.35031819e-01f,-1.28634185e-01f,+2.27315575e-01f,
-1.30157709e-01f,+1.38691932e-01f,+9.66220871e-02f,+7.22177699e-02f,+1.78922147e-01f,
-8.32418501e-02f,-1.85057372e-02f,-1.44866139e-01f,-1.84362177e-02f,-1.42228305e-01f,
+7.67072476e-03f,+9.64653492e-03f,+8.05501118e-02f,+1.41153157e-01f,+1.59074113e-01f,
+5.89684397e-02f,+1.77085936e-01f,-4.49741594e-02f,-1.57722175e-01f,-1.19576871e-01f,
-9.13322088e-04f,-9.57499072e-02f,+1.65967301e-01f,+1.40395820e-01f,+1.46697477e-01f,
+8.86715204e-02f,+1.61296263e-01f,+1.32564351e-01f,+1.13321736e-01f,+1.02391196e-02f,
+2.44709164e-01f,-6.62121475e-02f,+1.21857338e-02f,+1.65824935e-01f,-2.90401150e-02f,
-1.95447475e-01f,-1.46568269e-01f,-3.04110199e-02f,+5.04481159e-02f,+1.23014301e-02f,
+2.17522040e-01f,+1.66232884e-02f,+1.04685491e-02f,-8.34407806e-02f,+2.60465711e-01f,
-4.04789597e-02f,-3.19144912e-02f,-9.09222960e-02f,+8.39753598e-02f,-2.08521843e-01f,
+8.64672959e-02f,-1.61810249e-01f,+2.27452382e-01f,+3.05207193e-01f,-2.77187407e-01f,
+1.54968858e-01f,-1.17892258e-01f,+9.18307453e-02f,+2.21768111e-01f,+4.50057574e-02f,
-2.50618786e-01f,+2.70608574e-01f,+1.57711357e-01f,-2.24929247e-02f,+8.36000368e-02f,
-3.92531008e-02f,-1.25469014e-01f,-1.03316322e-01f,+1.71786129e-01f,+8.46538022e-02f,
-1.67470291e-01f,-1.10561125e-01f,+8.49121064e-02f,+7.25381542e-03f,+8.69830102e-02f,
+2.59553790e-01f,-1.42315164e-01f,+8.52743164e-02f,-1.71698317e-01f,+8.89022201e-02f,
+2.98810601e-01f,-3.26212421e-02f,+2.20842585e-01f,+3.00224811e-01f,-7.99110532e-02f,
-9.60555822e-02f,+1.91822216e-01f,-1.06685705e-01f,+1.04002312e-01f,-1.35686487e-01f,
+9.30749997e-02f,+7.99048617e-02f,+2.65173763e-01f,-2.31473088e-01f,+3.37790474e-02f,
-3.73881795e-02f,+1.92682445e-01f,+2.52925485e-01f,+1.80686668e-01f,+2.70559251e-01f,
+6.53988272e-02f,-1.07971802e-01f,+9.28815268e-03f,+2.06469342e-01f,+1.81081202e-02f,
-5.43945432e-02f,-6.71738386e-02f,-6.02064356e-02f,+2.08618388e-01f,+1.53307229e-01f,
+1.79527804e-01f,-1.13594614e-01f,+6.79429341e-03f,-6.41974658e-02f,+1.90905988e-01f,
+1.26184732e-01f,-1.60918638e-01f,+1.93341240e-01f,+1.19340010e-01f,-6.49799705e-02f,
+8.09072554e-02f,-1.04725525e-01f,-1.38496190e-01f,-5.83546311e-02f,+1.22289747e-01f,
+7.60430321e-02f,-1.04685128e-03f,-1.05380611e-02f,-1.25779539e-01f,-2.11385518e-01f,
+2.27858759e-02f,-4.89287078e-02f,+2.46504039e-01f,+8.25769231e-02f,+8.12771693e-02f,
+2.05333740e-01f,-8.86458438e-03f,+5.23100905e-02f,-2.05676094e-01f,+1.19118653e-01f,
-1.40661374e-01f,-1.87019840e-01f,+5.06475791e-02f,+3.65897529e-02f,+1.01512767e-01f,
-1.30973205e-01f,+1.27567738e-01f,-1.34866744e-01f,+1.36275634e-01f,+4.51775491e-02f,
+1.58730775e-01f,-1.29207715e-01f,+1.64228439e-01f,+7.73205906e-02f,-1.75422564e-01f,
-1.25299320e-01f,-5.27982563e-02f,-2.38409135e-02f,-6.09282367e-02f,+1.44477636e-01f,
+1.82509646e-01f,-1.52876139e-01f,+2.28380561e-01f,+1.15304530e-01f,+1.64193541e-01f,
+1.73639670e-01f,+8.56862068e-02f,+1.11262962e-01f,+9.91116762e-02f,+2.16334537e-01f,
-1.94975168e-01f,-1.27370030e-01f,+7.50465915e-02f,+9.30712149e-02f,+1.76184326e-01f,
+1.78800046e-01f,+1.88153833e-02f,+1.46914661e-01f,-1.47721497e-02f,-1.87807471e-01f,
+1.09106787e-01f,+2.65146136e-01f,-2.18262762e-01f,-1.56119736e-02f,-2.05370739e-01f,
-1.54673234e-01f,-1.08390287e-01f,+2.30843555e-02f,-1.69450998e-01f,+2.16749325e-01f,
-1.37369260e-01f,+4.89292294e-02f,-1.78335831e-01f,-1.44149512e-01f,+2.81898290e-01f,
-4.30193096e-02f,-2.98433304e-02f,+8.28513950e-02f,-2.50396758e-01f,+1.02512717e-01f,
-6.26069084e-02f,-2.18673050e-01f,-7.84216747e-02f,+1.59365848e-01f,-2.79370934e-01f,
-1.31359190e-01f,-2.31876165e-01f,+2.23035701e-02f,-1.61520109e-01f,+2.10395798e-01f,
+2.43322551e-01f,+3.17616910e-01f,-2.19660088e-01f,+2.97153443e-01f,-1.94019247e-02f,
-2.55293965e-01f,+7.00719804e-02f,-1.53138880e-02f,+1.07321724e-01f,-1.46835759e-01f,
-1.64729893e-01f,-1.78872451e-01f,-9.75491777e-02f,-1.16922766e-01f,+1.38542086e-01f,
+1.64676368e-01f,+6.90528452e-02f,+7.90673681e-03f,-9.64286774e-02f,-2.01704423e-03f,
+1.64894044e-01f,-5.54710999e-02f,-1.64249212e-01f,+1.93257317e-01f,+2.90536880e-01f,
+1.16648301e-01f,-7.19432756e-02f,-2.33960804e-02f,-1.92458257e-01f,+1.04812637e-01f,
+1.85601622e-01f,-9.40158516e-02f,+1.79357424e-01f,+2.07008556e-01f,-1.80414394e-01f,
-6.84940815e-02f,-1.30210087e-01f,-1.56316146e-01f,-1.02677405e-01f,+1.89277291e-01f,
+1.49908513e-01f,+8.13344270e-02f,+1.11463837e-01f,-9.10693184e-02f,+6.36990070e-02f,
-9.27165151e-02f,-1.84898376e-01f,+2.11254448e-01f,+1.36455491e-01f,-2.12519750e-01f,
+4.72346991e-02f,-1.57084242e-02f,+1.39607280e-01f,+8.34521279e-02f,-2.11801246e-01f,
+1.80057481e-01f,-1.30963370e-01f,+7.99420998e-02f,+1.72123611e-01f,-1.07367128e-01f,
-2.23646045e-01f,+8.50344300e-02f,-7.26779401e-02f,+1.13570482e-01f,+1.83349892e-01f,
+2.07292754e-03f,+5.74667715e-02f,-4.31508906e-02f,+1.59550518e-01f,+5.50038107e-02f,
-1.75252736e-01f,+1.26233786e-01f,+7.11693615e-03f,+3.39211486e-02f,+3.12170163e-02f,
-4.49495018e-03f,+6.44772053e-02f,-5.18830866e-02f,+5.90884462e-02f,+1.18832894e-01f,
+6.59897849e-02f,+1.25315741e-01f,+1.40557438e-01f,+1.14500582e-01f,-2.43845895e-01f,
-8.16243812e-02f,-9.55477282e-02f,+1.43165156e-01f,+2.00435556e-02f,-5.69951907e-03f,
+1.79000035e-01f,+1.45647213e-01f,+1.28165975e-01f,-2.00451925e-01f,+1.81482702e-01f,
+2.98962593e-02f,-1.40218735e-01f,+6.36143982e-02f,+1.41180649e-01f,+1.95672333e-01f,
+4.91374079e-03f,-1.66872635e-01f,+1.14851743e-01f,-1.47314161e-01f,+1.16645724e-01f,
+2.11350217e-01f,-9.11556035e-02f,+2.26721168e-01f,-4.29957174e-02f,+6.14269031e-03f,
+1.49850547e-01f,+2.97545120e-02f,-4.21494432e-02f,+9.92171913e-02f,+6.94344044e-02f,
-4.46136072e-02f,+1.01282001e-01f,+1.89679086e-01f,+6.61750659e-02f,+2.79956013e-01f,
-1.90152004e-01f,+2.34002575e-01f,-4.17959355e-02f,+2.30744611e-02f,+2.85025030e-01f,
-2.22380161e-02f,+1.63652971e-01f,+2.27775052e-01f,-2.36624524e-01f,+2.70945251e-01f,
+1.62887643e-03f,-2.11101636e-01f,-1.19511873e-01f,+2.54126549e-01f,-1.24605507e-01f,
+1.24180481e-01f,-1.89518631e-01f,+1.76016584e-01f,+1.16266280e-01f,+3.16368699e-01f,
+2.23202705e-01f,+9.73378494e-03f,-1.01378284e-01f,+3.11377972e-01f,+9.36002806e-02f,
-1.12086557e-01f,+1.65399108e-02f,-3.89155112e-02f,+1.78398147e-01f,-9.84328762e-02f,
-1.15624748e-01f,+2.66641546e-02f,-2.29691356e-01f,-1.15726583e-01f,+2.94917405e-01f,
+2.27630913e-01f,+1.58336699e-01f,-1.70318335e-01f,+1.11814417e-01f,+2.08452880e-01f,
+7.21092001e-02f,-2.64867008e-01f,-2.13130280e-01f,+3.20946947e-02f,+9.98798311e-02f,
+1.67095035e-01f,+3.78883816e-02f,-4.77735810e-02f,+3.80074680e-02f,-6.73487857e-02f,
-2.02632725e-01f,-1.48140624e-01f,-2.55879387e-02f,-1.04312468e-02f,+7.87268654e-02f,
+4.81207743e-02f,-8.32346752e-02f,+1.26950428e-01f,-2.51889050e-01f,-2.86464468e-02f,
-1.98708028e-01f,+6.78790035e-03f,+1.16230614e-01f,+1.24134548e-01f,-1.51456684e-01f,
-1.21525221e-01f,-2.45652925e-02f,+3.40930969e-01f,-8.93326029e-02f,+5.23947813e-02f,
-1.26006395e-01f,+1.45105496e-01f,+8.93548355e-02f,-1.36997327e-01f,+2.87128568e-01f,
-2.26544499e-01f,+1.33160904e-01f,-6.37577698e-02f,+1.55349914e-02f,+1.50424197e-01f,
+1.80621222e-01f,-2.49441624e-01f,-1.82320520e-01f,+7.22826868e-02f,+1.14192881e-01f,
-2.12911636e-01f,-2.10086867e-01f,+2.75339901e-01f,-4.35928814e-02f,+1.97356433e-01f,
+1.70868531e-01f,+1.05817303e-01f,-1.15219265e-01f,-6.18143491e-02f,-3.79037708e-02f,
+7.58664459e-02f,+1.19426856e-02f,-7.40713030e-02f,+1.04139388e-01f,-1.07835919e-01f,
-1.99883789e-01f,-4.14122036e-03f,+9.76541117e-02f,+2.77448003e-03f,+1.43694013e-01f,
+1.93218052e-01f,-1.51347831e-01f,-2.82740705e-02f,+9.13160592e-02f,-2.43654083e-02f,
-1.97541326e-01f,-1.37805045e-01f,-1.76838323e-01f,+1.33723527e-01f,-8.07802528e-02f,
+1.65398866e-02f,+6.45394996e-02f,-2.52684891e-01f,-9.35386866e-02f,-1.76601022e-01f,
-2.19068855e-01f,-9.93382037e-02f,-2.07255289e-01f,+1.24859775e-03f,+1.51098311e-01f,
+9.56612751e-02f,+2.74265334e-02f,+2.26216555e-01f,-2.57922430e-02f,-1.69409022e-01f,
-6.31238148e-03f,+2.50691026e-01f,-2.22354636e-01f,+1.29660979e-01f,+1.08920541e-02f,
-6.23211786e-02f,-2.33345237e-02f,+3.43622148e-01f,-1.33308366e-01f,+3.66747886e-01f,
-1.79036051e-01f,+3.33395660e-01f,-1.72618747e-01f,-2.81521603e-02f,+2.47663483e-01f,
-1.40046030e-01f,+1.75620690e-01f,-1.90627929e-02f,-2.28118181e-01f,+2.16121614e-01f,
+1.43201813e-01f,-6.58688694e-03f,+7.75609910e-02f,+2.73258805e-01f,-1.88016593e-02f,
-5.37209176e-02f,-1.65761724e-01f,+6.33269325e-02f,+7.29067922e-02f,+3.42067808e-01f,
+3.05391494e-02f,+1.89458102e-01f,-5.61779588e-02f,+3.64237666e-01f,-5.03744222e-02f,
-4.05676067e-02f,+2.33732928e-02f,+2.69473698e-02f,+2.95160711e-01f,+7.47027844e-02f,
-1.78755373e-01f,+1.37641765e-02f,-1.31708011e-01f,-2.74710178e-01f,+1.74889371e-01f,
+1.60001576e-01f,-7.66024971e-03f,-6.92671612e-02f,-9.79434103e-02f,+2.82426983e-01f,
-1.68450266e-01f,-1.94553323e-02f,-1.80896193e-01f,+2.28050262e-01f,+6.56394437e-02f,
-1.09938055e-01f,-1.10549688e-01f,-2.79976159e-01f,+7.41747767e-02f,-8.80702306e-03f,
-1.65302634e-01f,+1.99869368e-02f,-2.36381024e-01f,+7.62241706e-02f,+2.02956930e-01f,
+2.71793157e-01f,-5.03311008e-02f,+2.73716271e-01f,-1.31729364e-01f,-3.47295478e-02f,
+9.91886109e-02f,+3.02600682e-01f,-1.97351024e-01f,+1.03543196e-02f,-1.57350257e-01f,
-6.09682016e-02f,-3.41518670e-02f,+8.59379545e-02f,-2.57017761e-01f,+2.65777320e-01f,
-1.06025830e-01f,+3.77292745e-02f,-1.36545047e-01f,-2.38621160e-01f,+2.52378583e-01f,
-1.79146528e-01f,+9.39744264e-02f,+8.69560540e-02f,-1.50782779e-01f,+2.28123397e-01f,
+2.97906429e-01f,-2.27462813e-01f,-2.49797493e-01f,+1.94451496e-01f,-2.24069253e-01f,
-1.04549326e-01f,+6.96993470e-02f,+1.48150459e-01f,-6.13765465e-03f,+1.45375818e-01f,
+3.07318181e-01f,+1.34478465e-01f,-1.75280124e-01f,+3.20268035e-01f,+2.59957858e-03f,
-1.15934014e-02f,-7.60859326e-02f,+1.44620359e-01f,+1.93468541e-01f,-8.45473409e-02f,
-1.79409683e-01f,+1.42600134e-01f,-2.07797870e-01f,-1.91907257e-01f,+7.96615854e-02f,
+2.25321129e-01f,-7.67935812e-02f,+1.10705726e-01f,+1.45469293e-01f,+2.83294439e-01f,
+1.83194578e-01f,-2.45090097e-01f,-2.31366977e-01f,+2.64504135e-01f,+1.74246982e-01f,
+5.13594672e-02f,-3.75006907e-02f,+3.63637432e-02f,+2.79186666e-02f,-2.70964473e-01f,
+1.23847649e-01f,-1.47587717e-01f,+1.38928413e-01f,+9.40970238e-03f,+1.96053520e-01f,
+1.08272552e-01f,-1.74293190e-01f,+1.19028473e-02f,+1.07901171e-01f,+1.75252229e-01f,
+7.55563891e-03f,+2.36176118e-01f,-7.38461912e-02f,+4.55884524e-02f,-6.63218349e-02f,
-1.35009527e-01f,+8.29874724e-02f,+1.63022563e-01f,-6.15566038e-02f,-1.15679532e-01f,
-1.71681363e-02f,-4.60869409e-02f,+5.48556224e-02f,+1.72362030e-01f,-1.76946446e-02f,
+8.18581805e-02f,+1.02047876e-01f,+1.56263098e-01f,-1.22033097e-01f,+1.31465301e-01f,
+2.10695505e-01f,-4.59094793e-02f,-1.70136124e-01f,+2.27781713e-01f,-1.45159950e-02f,
-1.46098152e-01f,-2.07483485e-01f,+1.17329054e-01f,+4.33132909e-02f,+2.51829565e-01f,
+2.42973547e-02f,+1.32716686e-01f,+1.24964073e-01f,+2.64483765e-02f,+3.21283117e-02f,
-1.09725527e-01f,-2.91494075e-02f,+2.27960683e-02f,-1.11920416e-01f,-8.45109373e-02f,
+6.24642745e-02f,-1.24729358e-01f,+1.48900881e-01f,-1.50828481e-01f,-1.77938715e-02f,
+1.88759975e-02f,+2.50629842e-01f,-1.05226472e-01f,+7.74198100e-02f,+2.24910930e-01f,
+6.25884235e-02f,-1.08454861e-01f,-2.11907536e-01f,+1.15518853e-01f,-2.37047882e-03f,
-1.63768008e-01f,-8.84605944e-02f,-8.26485083e-03f,+1.11216910e-01f,+1.56800091e-01f,
+5.77212647e-02f,+1.16377920e-01f,-2.08036482e-01f,-1.36600927e-01f,-7.90734440e-02f,
+2.52867788e-01f,+1.38754949e-01f,-1.45637952e-02f,-1.13065258e-01f,-1.32381231e-01f,
-3.04001290e-02f,+1.83169812e-01f,-1.83917835e-01f,+1.38437375e-01f,-9.19704419e-03f,
-1.45442709e-01f,+6.40451983e-02f,+1.06581524e-01f,-1.44461289e-01f,+1.46397740e-01f,
-5.68591431e-02f,-1.43828690e-01f,-9.13852081e-02f,-7.72431046e-02f,+9.31223556e-02f,
-4.28113975e-02f,-1.09969638e-01f,-6.38630465e-02f,-5.25674894e-02f,+4.62375097e-02f,
-6.65033385e-02f,-5.92401214e-02f,-1.19860522e-01f,-5.10130413e-02f,-2.02392176e-01f,
-1.50010929e-01f,+1.74810827e-01f,+1.55391961e-01f,+4.70582442e-03f,+1.73657358e-01f,
+1.08312465e-01f,-1.18954457e-01f,+9.92202833e-02f,-6.69714138e-02f,+7.60094002e-02f,
+2.43753828e-02f,+1.40542015e-01f,-9.17015448e-02f,+1.08255297e-01f,+3.39501277e-02f,
-1.12897672e-01f,-3.75532545e-02f,+1.61081150e-01f,+1.49153084e-01f,+5.54229170e-02f,
+1.34227231e-01f,-4.30784300e-02f,+1.45941347e-01f,+1.56077534e-01f,-1.15918800e-01f,
-2.50978265e-02f,-1.15546644e-01f,+8.51036981e-02f,+2.37311609e-02f,-7.09208101e-02f,
+1.48928970e-01f,-1.70949310e-01f,-3.88724990e-02f,-2.33344757e-03f,-1.93843022e-02f,
+7.98989495e-04f,+1.86393969e-02f,-1.42185435e-01f,-1.12647288e-01f,+2.10044831e-02f,
-1.18624363e-02f,+1.24013633e-01f,-4.57235314e-02f,+1.14654312e-02f,-2.15164274e-01f,
-8.95741507e-02f,+9.32030454e-02f,-1.04317144e-01f,+2.23047271e-01f,-2.35943422e-01f,
-3.03279370e-01f,-1.19961627e-01f,+2.79351085e-01f,-6.06514923e-02f,+2.70379454e-01f,
-2.06943020e-01f,-4.69969772e-02f,-2.31323525e-01f,-2.38584206e-01f,+3.07031453e-01f,
+3.81307080e-02f,+2.34089047e-02f,+9.76296440e-02f,-2.50555992e-01f,+2.67397732e-01f,
+2.08053142e-01f,-1.50803238e-01f,-2.81392574e-01f,+1.17558278e-01f,-9.70437750e-02f,
-1.97425947e-01f,+1.43789604e-01f,-2.56439741e-03f,-5.63570336e-02f,+1.21171199e-01f,
-5.54849841e-02f,+2.33445972e-01f,-2.89703369e-01f,+1.70767814e-01f,-1.82724863e-01f,
-2.03274682e-01f,-8.27125087e-02f,-3.86177115e-02f,+1.84592053e-01f,+4.39621434e-02f,
+1.39226194e-03f,-7.30336038e-03f,-1.98652163e-01f,-2.42294982e-01f,+1.55028403e-01f,
+1.99263796e-01f,+3.93863879e-02f,-1.87151413e-02f,+1.24437563e-01f,+1.15941800e-01f,
+4.14805450e-02f,-1.32471487e-01f,-2.67567277e-01f,+9.73899476e-03f,+3.22358287e-03f,
-2.68092006e-02f,-8.01719353e-02f,-3.60247679e-02f,-1.16648071e-01f,+1.04115218e-01f,
-2.88060963e-01f,-1.10825136e-01f,-2.72396719e-03f,-2.19500527e-01f,+3.58286276e-02f,
+9.58329141e-02f,+6.11249804e-02f,+1.64984360e-01f,-7.36220703e-02f,-2.52010375e-01f,
-6.65952042e-02f,-1.45677105e-01f,-1.78344309e-01f,-8.51904415e-03f,-3.15421104e-01f,
-2.30916157e-01f,+1.69834420e-01f,-1.39046669e-01f,-1.23199355e-03f,-2.85307348e-01f,
-5.02228290e-02f,+1.93129912e-01f,-2.79604584e-01f,-1.90823644e-01f,+7.99158290e-02f,
-2.57233884e-02f,-3.25402170e-02f,+5.17808683e-02f,-1.33188501e-01f,-2.91565359e-02f,
-1.29140094e-01f,-1.71759829e-01f,-1.24119580e-01f,-1.06325902e-01f,-3.08926195e-01f,
-2.57069528e-01f,+2.84665227e-01f,-1.36852086e-01f,+5.92884570e-02f,-1.31480426e-01f,
-2.89000958e-01f,+2.97098476e-02f,+5.32579012e-02f,-4.01457548e-02f,+3.62100899e-02f,
+6.87507307e-03f,+1.70012191e-01f,+1.08993128e-01f,+4.67966944e-02f,-1.91749513e-01f,
-8.87158290e-02f,+2.36877501e-02f,+1.29947215e-01f,-2.77268022e-01f,+7.71765190e-04f,
+6.96780011e-02f,-2.85903692e-01f,-1.81936055e-01f,+1.28204450e-01f,+3.33656073e-02f,
+1.38467446e-01f,-4.53811362e-02f,-2.64658332e-01f,+1.42102778e-01f,+5.67052811e-02f,
-1.28850341e-04f,-6.64265379e-02f,-1.31902888e-01f,+1.57015249e-01f,-1.40694112e-01f,
+1.64719716e-01f,+1.95891678e-01f,-8.80869552e-02f,-1.19795069e-01f,+1.13291584e-01f,
+1.79763213e-01f,+1.56827480e-01f,+1.62031934e-01f,-5.98988459e-02f,+9.07858163e-02f,
-6.26760796e-02f,-1.53107503e-02f,+1.89348236e-01f,+8.43621939e-02f,+2.00488083e-02f,
+1.07563592e-01f,+1.19919263e-01f,+1.79733053e-01f,+2.42375299e-01f,-4.45243977e-02f,
-1.85780060e-02f,-7.22739995e-02f,+1.90518081e-01f,-1.09750830e-01f,+1.12994686e-01f,
+2.02900711e-02f,+5.92332929e-02f,-1.47341743e-01f,+7.31483400e-02f,+1.35909095e-01f,
-2.09940858e-02f,+3.98425125e-02f,-1.84492338e-02f,+8.85958131e-03f,+6.21680990e-02f,
-1.41755968e-01f,+1.23137198e-02f,+1.03336655e-01f,+1.04338773e-01f,+3.81423496e-02f,
-1.53278679e-01f,+8.99031311e-02f,+1.32711709e-01f,+6.56853169e-02f,-1.68352306e-01f,
+6.36591241e-02f,+6.93331240e-04f,-3.73525023e-02f,+2.15163678e-02f,-1.51162431e-01f,
+8.88129789e-03f,-1.80901900e-01f,+5.78727946e-02f,+1.00663871e-01f,-1.89913869e-01f,
+1.89657584e-01f,+9.41944122e-02f,-6.43730387e-02f,-1.82660595e-01f,+1.58084780e-01f,
-1.25561938e-01f,+2.67008729e-02f,+1.49256885e-01f,+4.55324538e-02f,+2.68747747e-01f,
-7.30425864e-02f,-1.99915841e-03f,+1.49978340e-01f,+5.03007472e-02f,+1.73166960e-01f,
+7.82765374e-02f,-9.72769037e-02f,+8.01339597e-02f,-4.67352830e-02f,+3.47290337e-02f,
-2.14401513e-01f,+1.52085304e-01f,-1.37408510e-01f,-1.13504879e-01f,+1.56528383e-01f,
+1.01697631e-01f,+3.97159755e-02f,+3.07757035e-02f,+9.69584435e-02f,+7.81031623e-02f,
+2.06262380e-01f,+2.48923413e-02f,+1.93111271e-01f,+2.62367159e-01f,-1.98780179e-01f,
+1.21732429e-01f,-1.09127529e-01f,-6.89018890e-02f,+1.58549309e-01f,-1.73479125e-01f,
-9.98804197e-02f,-1.39561117e-01f,-1.81526795e-01f,-2.41441987e-02f,-1.77850395e-01f,
+3.03994268e-02f,+1.70716092e-01f,+2.66970128e-01f,-2.60851055e-01f,+1.25008479e-01f,
+1.38289705e-01f,-7.77891129e-02f,+4.21870686e-02f,-1.36902615e-01f,-1.13315471e-01f,
-3.33553441e-02f,-1.37848511e-01f,+1.36221021e-01f,+3.10456213e-02f,-1.85435727e-01f,
-6.28444999e-02f,-1.52501747e-01f,+4.41867560e-02f,+6.44273013e-02f,+1.34496503e-02f,
+1.82611406e-01f,-6.25729412e-02f,+1.88904852e-02f,+2.57415026e-01f,-3.39585617e-02f,
+7.55513683e-02f,+2.45944470e-01f,-7.68261701e-02f,-1.79181606e-01f,+3.77857201e-02f,
-1.53459489e-01f,+1.52126700e-01f,+1.26891404e-01f,-1.23746946e-01f,+2.07780357e-02f,
-1.26836658e-01f,+2.14305744e-01f,+1.28181100e-01f,-1.54647650e-02f,-1.12884015e-01f,
+5.16064763e-02f,-1.31609470e-01f,+2.01008409e-01f,+6.30931184e-02f,+7.03135431e-02f,
-1.60557762e-01f,+1.53100282e-01f,-1.48797318e-01f,+1.76036358e-01f,+2.06709251e-01f,
+1.12135939e-01f,+1.92081362e-01f,+1.60129428e-01f,-1.70913830e-01f,+2.12931663e-01f,
-7.26434439e-02f,-1.08617529e-01f,+8.09303299e-02f,+3.31525505e-02f,-1.02405652e-01f,
-1.19884476e-01f,-4.33214866e-02f,+1.46756589e-01f,-1.20204799e-01f,-9.28043798e-02f,
-3.64270210e-02f,+1.71392366e-01f,+1.62260056e-01f,+3.04845367e-02f,-1.44602865e-01f,
-5.83843924e-02f,+2.12325528e-01f,+8.80561769e-02f,+9.33095291e-02f,-3.73976864e-02f,
-1.33352578e-01f,-9.09150168e-02f,+5.16434051e-02f,-1.81059405e-01f,+1.34128883e-01f,
-1.07391693e-01f,-4.42720689e-02f,-1.24558257e-02f,+1.59940377e-01f,-1.32559121e-01f,
+7.67729655e-02f,+1.67147875e-01f,-3.75527628e-02f,+1.08244129e-01f,+7.26518929e-02f,
+9.22907442e-02f,+7.95095712e-02f,+2.11567789e-01f,-1.43692538e-01f,+4.56787385e-02f,
-1.17388301e-01f,+2.70425141e-01f,-1.53778136e-01f,-1.05088048e-01f,-2.47963630e-02f,
-6.35735989e-02f,+1.06777430e-01f,+1.85747147e-01f,+6.97130011e-03f,+9.53190476e-02f,
+4.21979986e-02f,+2.06379771e-01f,-3.48698236e-02f,+1.08134627e-01f,+1.00769483e-01f,
+1.30141467e-01f,+1.59459457e-01f,+4.68208790e-02f,-9.00069904e-03f,-1.84180453e-01f,
-1.72953345e-02f,+1.62863582e-01f,+5.97250536e-02f,+1.49955414e-02f,+1.96065456e-01f,
+2.05552697e-01f,-1.43451810e-01f,+9.98767763e-02f,+7.94309378e-02f,+1.89684719e-01f,
+2.54133660e-02f,-5.75479567e-02f,-4.71577868e-02f,+1.73729479e-01f,+8.13789386e-03f,
+1.40807107e-01f,-9.68013778e-02f,-1.13018051e-01f,+1.85720876e-01f,+7.97787085e-02f,
-8.82147253e-02f,-1.85533538e-01f,+3.69470641e-02f,+8.29522833e-02f,+1.82264694e-03f,
-4.53952439e-02f,+2.21489996e-01f,+7.86660146e-03f,+1.89949628e-02f,-7.54192378e-03f,
+5.69152758e-02f,+1.90078646e-01f,-1.59831971e-01f,-1.04062796e-01f,+1.51679501e-01f,
-3.22833359e-02f,-1.40693143e-01f,+5.07343560e-02f,-1.97092503e-01f,+1.32907890e-02f,
+2.33068466e-02f,-1.57627881e-01f,-2.86199129e-03f,-1.80346280e-01f,-2.30855793e-02f,
-6.40516430e-02f,-8.94224495e-02f,+4.95235156e-03f,+1.67144433e-01f,-1.45531744e-02f,
+9.39005688e-02f,-1.06664196e-01f,+8.40790719e-02f,+1.26189530e-01f,-7.02944994e-02f,
-1.94178313e-01f,+2.19969511e-01f,+1.93250984e-01f,-4.14959900e-02f,+1.53580219e-01f,
-6.27587140e-02f,+2.07214341e-01f,-3.26493643e-02f,-1.29037529e-01f,+8.06084573e-02f,
-7.68947378e-02f,+6.72123507e-02f,-7.70460740e-02f,-1.22865893e-01f,-3.96404564e-02f,
+7.47136697e-02f,+1.04901791e-01f,+2.79727668e-01f,-3.95827517e-02f,+4.32398058e-02f,
-7.27664754e-02f,-3.91888432e-02f,-4.48142439e-02f,+3.04602440e-02f,-1.40451998e-01f,
-1.43196195e-01f,-1.96807101e-01f,+1.65764615e-02f,+8.80251545e-03f,+2.15114817e-01f,
-7.90968165e-02f,+7.91044906e-02f,-1.16207503e-01f,-7.10988715e-02f,-5.91045618e-03f,
-6.89986423e-02f,+5.31959981e-02f,-1.63905382e-01f,-6.18813261e-02f,+1.93523109e-01f,
+1.34187371e-01f,-1.47919804e-01f,-4.50981036e-02f,-1.34990782e-01f,-1.95609257e-01f,
+1.29747972e-01f,-3.22370306e-02f,+2.97647100e-02f,+8.67881253e-03f,+8.46342221e-02f,
+1.84659719e-01f,+1.58211365e-01f,-1.20892502e-01f,+7.25882798e-02f,+6.36404753e-02f,
+1.84566155e-02f,+5.70027120e-02f,+8.60017389e-02f,+2.09471196e-01f,+1.04476072e-01f,
+1.31134421e-01f,-5.93952574e-02f,-2.95549333e-02f,-5.81347570e-02f,+1.74902573e-01f,
+1.24793619e-01f,-9.60140750e-02f,-1.30532742e-01f,+1.61590472e-01f,+3.37994099e-02f,
-1.54329821e-01f,-1.74838543e-01f,+7.50158429e-02f,-8.35443810e-02f,-5.08417143e-03f,
-1.73915416e-01f,-8.06361213e-02f,+1.30845428e-01f,+1.06961407e-01f,-3.91404741e-02f,
-2.49192826e-02f,+1.10024482e-01f,-1.55115008e-01f,-7.73412809e-02f,+6.21707691e-03f,
-3.05460840e-02f,+2.45049391e-02f,+2.90611433e-03f,-1.26827778e-02f,+9.41542387e-02f,
+1.59701020e-01f,-1.07186370e-01f,+2.98303962e-02f,+7.80851999e-03f,+9.25241932e-02f,
+1.68485716e-01f,+1.65580496e-01f,+1.98595956e-01f,+1.15550481e-01f,-8.98555443e-02f,
+1.53507844e-01f,+2.02296525e-01f,-1.31671980e-01f,+1.03050508e-02f,-9.79603007e-02f,
-4.05665413e-02f,-7.56702572e-02f,-7.96040222e-02f,+9.82455164e-02f,+9.33755040e-02f,
-1.77137464e-01f,+2.93133743e-02f,+7.32142404e-02f,-1.98791876e-01f,-1.24292567e-01f,
+1.41092137e-01f,-1.66452769e-02f,+2.78236531e-02f,+4.38492335e-02f,+7.58617534e-04f,
-1.41967759e-01f,+1.90183103e-01f,+2.03559354e-01f,-9.10558179e-02f,-4.06317487e-02f,
-1.39312297e-01f,-1.19156353e-01f,-9.35541242e-02f,+5.47268502e-02f,-8.42644423e-02f,
-1.04582846e-01f,+1.01543432e-02f,+1.22191329e-02f,+2.34413072e-01f,-1.87124223e-01f,
+2.66965199e-02f,+9.04697254e-02f,-5.31906784e-02f,+1.25278980e-01f,+6.24376722e-02f,
+1.64760888e-01f,+2.01534241e-01f,+2.96761431e-02f,+9.16236490e-02f,-6.05704784e-02f,
-1.21255629e-01f,-8.71696696e-02f,+3.05166864e-03f,-1.61665991e-01f,+1.74283534e-01f,
+9.67498943e-02f,-8.98126364e-02f,-8.68864805e-02f,+1.86408088e-01f,+3.27632576e-02f,
-9.92271379e-02f,-8.14507753e-02f,-6.48275837e-02f,-5.85205629e-02f,+6.71517244e-03f,
-8.31189007e-02f,-1.65741250e-01f,-5.34361117e-02f,-8.18500966e-02f,-1.57253724e-02f,
+5.89442067e-02f,+5.33336028e-03f,+2.76164035e-04f,-2.37384066e-02f,+2.70468183e-03f,
+1.16447508e-01f,+1.56625956e-01f,+1.50980055e-01f,+1.99285541e-02f,+5.69171272e-02f,
+5.60373813e-02f,-1.89320013e-01f,-4.61052358e-02f,-7.34405965e-02f,+6.53778166e-02f,
+1.60874605e-01f,-5.36560677e-02f,-1.48206070e-01f,-2.00022787e-01f,+6.97747916e-02f,
+1.21780597e-01f,+1.14192627e-01f,-2.08080247e-01f,+1.15381345e-01f,+1.81219414e-01f,
-1.95761286e-02f,-1.65172468e-03f,+1.96319334e-02f,+1.81171209e-01f,+1.74240604e-01f,
-6.56847581e-02f,+3.62968370e-02f,+1.87700018e-01f,-8.42126086e-02f,-2.00426579e-01f,
+1.66826658e-02f,-9.42264423e-02f,+8.65971819e-02f,+9.69891623e-02f,+3.70469913e-02f,
+1.33942470e-01f,-1.60660401e-01f,+1.78403795e-01f,+4.03944626e-02f,+3.28855366e-02f,
+1.10387579e-01f,-1.78172573e-01f,+2.95993313e-03f,-1.04964480e-01f,-1.70184061e-01f,
-1.47874415e-01f,+1.40891135e-01f,-3.30970846e-02f,-1.80417150e-01f,+1.87870860e-02f,
-9.17279124e-02f,+1.06897913e-01f,+1.87447667e-02f,-8.06197226e-02f,-1.81996241e-01f,
-1.67623058e-01f,-1.17129922e-01f,-2.41332036e-03f,+1.47887647e-01f,-1.05123408e-01f,
+1.74427673e-01f,+4.44370583e-02f,+2.58292705e-02f,-4.80385274e-02f,-6.06816262e-02f,
-1.33464992e-01f,+1.12989888e-01f,+1.87251374e-01f,+6.40830547e-02f,-1.62757635e-01f,
+1.42829493e-01f,-6.40396550e-02f,+7.45764226e-02f,-1.88706249e-01f,-1.87223777e-01f,
-6.71799481e-02f,+1.50962576e-01f,+1.82496328e-02f,-4.57416289e-02f,-4.37365472e-02f,
-9.63702276e-02f,+4.45999950e-02f,+1.38665378e-01f,+1.90115079e-01f,-1.35276318e-01f,
-9.58422124e-02f,-1.31509878e-05f,-1.39471978e-01f,+1.11242905e-01f,+1.09808967e-01f,
+1.23194136e-01f,-1.08671106e-01f,-1.75014421e-01f,+1.46316648e-01f,-1.51581854e-01f,
-1.67779475e-01f,-1.68163985e-01f,-1.89200014e-01f,+3.11085396e-02f,-1.08750544e-01f,
+1.11872330e-01f,-3.11854947e-02f,-1.08226709e-01f,-1.39501661e-01f,-4.12650071e-02f,
-7.88815543e-02f,+5.93696311e-02f,-1.95328489e-01f,-1.76744908e-01f,+6.56647533e-02f,
+6.63713366e-02f,+1.23216629e-01f,+1.91524655e-01f,+1.39339879e-01f,-6.14876300e-02f,
+1.63500056e-01f,+1.60423771e-01f,-1.92548811e-01f,+4.18711156e-02f,+1.07509177e-03f,
-1.10987490e-02f,+2.49449164e-02f,+5.84375523e-02f,-1.01838768e-01f,+3.14705640e-01f,
+7.82044679e-02f,-4.10160888e-03f,+3.48952919e-01f,-1.07005402e-01f,-2.60300308e-01f,
-1.40934303e-01f,+3.57483000e-01f,-2.68350661e-01f,+2.44573668e-01f,-2.38360047e-01f,
-1.01872407e-01f,-1.73599601e-01f,+1.61974505e-01f,-2.78507531e-01f,+1.04763761e-01f,
-5.49075045e-02f,-1.26892766e-02f,-2.07967117e-01f,+3.00630536e-02f,-6.85988879e-03f,
+1.68791652e-01f,+1.45338520e-01f,+2.11176589e-01f,-1.57293171e-01f,+2.79445261e-01f,
+2.84431785e-01f,-1.64641693e-01f,-2.56093502e-01f,-1.89510379e-02f,-2.42438406e-01f,
-1.31861746e-01f,+1.99366212e-01f,+2.58877903e-01f,+1.62806317e-01f,+2.80194044e-01f,
-1.29299350e-02f,+6.46203384e-02f,-1.70158446e-01f,+2.34175250e-01f,+1.43573046e-01f,
-7.14583322e-02f,+9.21289474e-02f,-1.31962687e-01f,+3.32554907e-01f,-6.88554347e-02f,
-2.16627717e-01f,-1.89524293e-02f,+1.48743719e-01f,-2.90827274e-01f,+1.46409184e-01f,
+2.21795604e-01f,+3.18817943e-01f,-7.80018931e-03f,-4.53230292e-02f,+4.68987860e-02f,
+7.53577799e-03f,+1.90815721e-02f,-2.61380136e-01f,-1.34462630e-02f,-3.14941220e-02f,
+3.29564661e-02f,-5.38837053e-02f,+2.15055477e-02f,-8.22713971e-03f,+3.51201780e-02f,
-7.57509544e-02f,+1.25756236e-02f,-2.66882852e-02f,+2.44266406e-01f,+1.98817864e-01f,
+9.30449218e-02f,-2.11170256e-01f,-1.33954003e-01f,+2.06134796e-01f,+8.80913064e-02f,
-4.69882004e-02f,-7.40323961e-02f,+1.74699634e-01f,-1.55114025e-01f,-9.94490981e-02f,
+1.38024792e-01f,+1.65422961e-01f,+5.50142340e-02f,+1.69016615e-01f,-1.90148830e-01f,
+7.40171298e-02f,+1.05876617e-01f,+1.94128647e-01f,+1.35419322e-02f,-1.03637382e-01f,
+1.96288049e-01f,-6.58104420e-02f,-5.80106974e-02f,+2.08630070e-01f,-3.14925969e-01f,
-1.79308206e-01f,+2.59433508e-01f,-3.68829928e-02f,-9.69899297e-02f,+9.81921926e-02f,
+8.77821967e-02f,-8.85967165e-02f,-3.24746855e-02f,-1.22421302e-01f,-8.37862715e-02f,
-1.26099586e-01f,-1.24284416e-01f,+1.33469328e-01f,+9.76499841e-02f,-1.48871481e-01f,
+2.90069007e-03f,+1.74670756e-01f,-4.27514762e-02f,-3.66960950e-02f,+1.27368480e-01f,
+5.25988303e-02f,-3.05237602e-02f,-9.19223130e-02f,+1.85320541e-01f,-1.21975169e-01f,
+1.19264305e-01f,+4.54677083e-02f,-2.47376189e-02f,+7.25000948e-02f,-1.28035814e-01f,
+8.85773823e-02f,+1.41491398e-01f,-5.20549081e-02f,-1.68083146e-01f,+7.93438405e-03f,
+1.18068099e-01f,+1.39565974e-01f,-6.19746186e-02f,+1.70304865e-01f,-1.02486387e-01f,
-7.33536407e-02f,-1.20343743e-02f,-5.02410308e-02f,-1.44673541e-01f,+3.83460402e-01f,
+3.18397641e-01f,+6.52285963e-02f,+1.80140540e-01f,-1.13255262e-01f,-1.02948166e-01f,
-2.68692821e-01f,+4.01448935e-01f,-1.23888649e-01f,+2.16458201e-01f,-2.09570691e-01f,
-7.82293156e-02f,-9.65819284e-02f,+1.32012442e-01f,-3.02791625e-01f,+1.18693106e-01f,
+3.45226377e-02f,+1.44378856e-01f,-1.35202155e-01f,+4.07257788e-02f,+3.14881764e-02f,
-2.00560257e-01f,+8.56345445e-02f,-1.53756082e-01f,-2.28542686e-01f,+1.97469592e-01f,
-8.43959767e-03f,+4.73799035e-02f,-1.06898285e-01f,+3.52791518e-01f,-1.38627991e-01f,
+1.81582884e-03f,-8.05125758e-02f,+2.26808742e-01f,+7.46640563e-02f,+3.71496052e-01f,
+3.24174881e-01f,+1.72597542e-01f,-6.11484461e-02f,+2.18966767e-01f,-1.52150303e-01f,
-1.86367556e-01f,+3.67161781e-02f,-1.23588294e-01f,+9.07039046e-02f,-3.56703997e-03f,
-2.16413185e-01f,-1.40024111e-01f,-2.10204646e-01f,+3.92306671e-02f,+3.02264541e-01f,
+9.97340828e-02f,+2.35578656e-01f,+1.63698539e-01f,-1.08613610e-01f,+3.87049586e-01f,
-2.66577397e-02f,+7.48182787e-03f,-3.32408398e-01f,+2.08352327e-01f,+8.20957422e-02f,
+4.75510731e-02f,-1.17771089e-01f,-1.64817676e-01f,+9.51057523e-02f,-1.79543301e-01f,
+1.46509960e-01f,+8.14818293e-02f,+3.51237804e-02f,+3.26305218e-02f,+1.09160572e-01f,
+1.77436218e-01f,-1.29929870e-01f,+1.98488042e-01f,+1.47023544e-01f,-1.23812400e-01f,
+1.72119185e-01f,+1.36999980e-01f,-1.24681138e-01f,-1.53905466e-01f,+1.91647455e-01f,
-6.72469810e-02f,-2.03494385e-01f,-2.91443542e-02f,+3.39728259e-02f,+7.04463124e-02f,
+4.33228202e-02f,+7.67220110e-02f,+1.80093765e-01f,+2.44499162e-01f,+8.38464200e-02f,
+2.63229012e-02f,-1.12824790e-01f,+1.21806003e-01f,+1.33537754e-01f,+1.33657649e-01f,
+5.58666559e-03f,+2.22331155e-02f,+2.42180511e-01f,-3.80808190e-02f,+1.41768634e-01f,
+2.21315444e-01f,-1.91521287e-01f,+1.07451811e-01f,+9.37460959e-02f,+6.15239367e-02f,
+1.12643324e-01f,+7.02942461e-02f,-1.08512454e-01f,-1.46307155e-01f,+4.74584959e-02f,
+1.84470695e-03f,-2.68194731e-02f,+5.44840693e-02f,-1.10021560e-02f,-3.29094417e-02f,
+1.78972930e-01f,-1.04039550e-01f,-7.01896176e-02f,+1.52488932e-01f,+1.59596562e-01f,
+1.26271183e-02f,+2.03564152e-01f,+7.77465776e-02f,+3.23586054e-02f,-1.31174594e-01f,
-2.00040750e-02f,-9.63184088e-02f,+2.00899839e-01f,-2.58211829e-02f,+2.57519037e-01f,
+5.05211987e-02f,+1.78207114e-01f,+5.80775142e-02f,-1.90789439e-02f,+1.09280847e-01f,
+2.02061102e-01f,-1.73848160e-02f,+5.45937978e-02f,-5.37134595e-02f,-5.83883896e-02f,
-2.27034595e-02f,+1.66946799e-01f,-1.82447135e-01f,+2.16975391e-01f,-6.97495835e-03f,
+7.93748870e-02f,+5.67442887e-02f,-9.98228565e-02f,-7.08539858e-02f,+5.02048545e-02f,
+1.86775446e-01f,+1.27835989e-01f,+1.77127123e-01f,+2.30634958e-01f,-1.23967424e-01f,
-2.02908099e-01f,-1.05579190e-01f,+2.62080789e-01f,+1.09643966e-01f,-2.65999973e-01f,
+7.32295811e-02f,+8.51176679e-03f,-5.92097007e-02f,+2.87446290e-01f,-2.70927906e-01f,
-1.20429598e-01f,+7.26836696e-02f,+1.11153424e-01f,+3.48140448e-02f,+2.30663687e-01f,
-3.01602352e-02f,-1.45602524e-01f,+4.39261720e-02f,+1.16479963e-01f,+3.12153399e-02f,
-2.51033515e-01f,-1.64996430e-01f,+2.57923543e-01f,+3.72929052e-02f,+1.43213451e-01f,
-3.49455141e-02f,-1.50935978e-01f,-5.88569492e-02f,-6.92843571e-02f,-2.84461342e-02f,
+1.80740282e-01f,+1.84988618e-01f,-1.28963351e-01f,+1.75356213e-02f,+6.87650368e-02f,
-2.17220470e-01f,+1.57548085e-01f,+2.77667716e-02f,-1.72483295e-01f,-9.58871394e-02f,
-8.67816806e-02f,+2.65544116e-01f,+2.07252085e-01f,-2.41418928e-02f,+1.09769166e-01f,
-1.33689530e-02f,+2.32856765e-01f,+2.34368220e-01f,+2.46109702e-02f,+4.26312946e-02f,
-1.28778338e-01f,+1.24578059e-01f,-1.45541489e-01f,-1.31316230e-01f,+1.67973474e-01f,
+7.61087388e-02f,-3.83838899e-02f,-1.55326918e-01f,-1.90627947e-01f,-8.37175995e-02f,
-3.15814279e-03f,-1.31217360e-01f,-8.22588950e-02f,+3.62943187e-02f,+1.25073642e-01f,
+1.21579114e-02f,+5.69111519e-02f,-1.84116855e-01f,-1.13695674e-01f,+1.96439281e-01f,
-9.61630717e-02f,+8.04412067e-02f,+9.91351232e-02f,-1.25339866e-01f,+1.87311456e-01f,
-3.40040028e-03f,+1.06161550e-01f,+1.07673645e-01f,+1.34885594e-01f,+1.54049248e-02f,
+5.59540354e-02f,+5.57474457e-02f,-5.39746275e-03f,+1.91762492e-01f,+1.35120556e-01f,
-1.65910065e-01f,+8.19454654e-05f,+2.13713758e-02f,+7.97872692e-02f,-1.17707197e-02f,
-3.56373452e-02f,-1.53932974e-01f,-1.98958039e-01f,-2.18082219e-03f,-1.67803466e-01f,
-4.49675210e-02f,-1.37151912e-01f,-2.09815260e-02f,-1.45450518e-01f,-1.46510750e-02f,
-1.79036334e-01f,+1.19834572e-01f,+1.18480087e-03f,-2.16717483e-03f,+1.46627605e-01f,
-1.88374668e-01f,+4.21683937e-02f,+8.55673626e-02f,-4.34269831e-02f,-5.15104160e-02f,
-9.84988511e-02f,+4.76074964e-02f,-1.07376300e-01f,+1.91468209e-01f,+2.34166230e-03f,
-1.08457170e-01f,-1.29113317e-01f,-8.66166898e-04f,-1.61249727e-01f,-3.04814009e-03f,
-8.99384096e-02f,+1.75571159e-01f,+5.99018298e-02f,+2.36981615e-01f,+2.14027271e-01f,
+2.04607502e-01f,-7.72768632e-02f,+1.67207807e-01f,-5.47382534e-02f,+1.49370898e-02f,
+1.68803692e-01f,-3.75659391e-02f,+5.65160066e-02f,-1.29006475e-01f,+1.56757921e-01f,
+2.12038457e-01f,+1.54639497e-01f,+2.10669503e-01f,-7.80974254e-02f,-6.01581074e-02f,
-1.73752129e-01f,-3.55807543e-02f,+1.72046915e-01f,+6.70580789e-02f,-1.00329392e-01f,
+2.82343805e-01f,-2.02680081e-02f,-6.80094585e-02f,+1.46247223e-01f,+1.61190793e-01f,
-3.11801638e-02f,+1.44216955e-01f,+3.47192846e-02f,-1.10604577e-02f,+3.04328986e-02f,
+1.83910251e-01f,-3.08580339e-01f,-1.70997098e-01f,-2.96198614e-02f,+5.30335270e-02f,
-1.40669450e-01f,-1.85720101e-01f,+1.13289982e-01f,-6.12129420e-02f,+1.17422856e-01f,
+1.88176513e-01f,-3.79667692e-02f,-4.30155508e-02f,+8.52363184e-02f,-1.65167302e-01f,
+1.26044884e-01f,+1.87354878e-01f,+4.53745201e-02f,+7.59503990e-02f,+1.08471876e-02f,
+1.30601868e-01f,+6.67620897e-02f,+4.87302989e-02f,-5.08990698e-02f,-1.45295560e-01f,
-7.41747320e-02f,-3.00975703e-03f,+4.16303839e-04f,-1.51035115e-01f,+1.72488987e-01f,
-1.67910293e-01f,+2.54275501e-01f,+1.08603865e-01f,-2.55152266e-02f,-4.15636152e-02f,
-1.43858865e-01f,+5.56794368e-02f,+9.52824354e-02f,+1.85363024e-01f,+1.13502920e-01f,
+1.23757750e-01f,-8.00959170e-02f,+4.35499661e-02f,+1.36639148e-01f,-1.77154571e-01f,
+1.08066961e-01f,+1.34073630e-01f,+1.95869640e-01f,-1.26381785e-01f,+1.41757950e-01f,
+9.90848839e-02f,+1.09893374e-01f,-1.55550376e-01f,+1.08767398e-01f,-5.76074421e-03f,
+6.94706067e-02f,+9.83636379e-02f,+2.09033433e-02f,-1.73061285e-02f,+1.11012250e-01f,
-1.38086468e-01f,+1.62709877e-01f,-6.12383485e-02f,-1.01404272e-01f,-1.67437047e-02f,
+2.61887070e-02f,+4.19628732e-02f,-4.21505719e-02f,-1.15878776e-01f,+4.82244790e-02f,
-6.95679337e-02f,+5.11902571e-02f,+1.63168207e-01f,+1.19929165e-02f,+5.26729822e-02f,
-1.22639298e-01f,-2.95614544e-02f,+7.87291601e-02f,+7.66531006e-02f,+6.87649101e-02f,
+5.19572124e-02f,+1.52534857e-01f,-7.45086074e-02f,-9.96791050e-02f,-5.86303025e-02f,
-8.25459659e-02f,-1.46857396e-01f,+1.23285651e-01f,+1.07261896e-01f,-1.61053061e-01f,
-5.85989840e-02f,-5.74167259e-02f,-2.53954455e-02f,-1.21596292e-01f,+1.32186525e-02f,
+1.72396004e-01f,-9.56321955e-02f,-1.52032459e-02f,+1.79104373e-01f,-3.14222835e-02f,
+7.31864572e-03f,-8.75296816e-02f,-1.22895077e-01f,+2.62384601e-02f,+1.28035039e-01f,
-1.94481388e-01f,-1.36359081e-01f,-2.81002998e-01f,-3.73574719e-02f,+3.66743766e-02f,
+6.34761080e-02f,-1.62993819e-01f,+4.02023286e-01f,-2.42588267e-01f,-6.05766028e-02f,
-8.61079898e-03f,+3.25958356e-02f,-3.53563011e-01f,+1.75736994e-01f,-2.84902155e-01f,
-7.35508651e-02f,-1.46635398e-01f,-1.12160303e-01f,-1.81337427e-02f,-9.12721902e-02f,
-1.74434006e-01f,+1.87411994e-01f,-7.54953250e-02f,-3.09706032e-01f,+3.95121604e-01f,
+1.52314395e-01f,+4.21981514e-03f,-1.49528841e-02f,-2.00199068e-01f,-6.81461543e-02f,
+1.23380616e-01f,-1.70841008e-01f,-2.44607747e-01f,+1.07099228e-01f,-1.59352064e-01f,
-3.23958874e-01f,+3.21768895e-02f,+1.98559552e-01f,-1.12572506e-01f,-1.83321536e-01f,
+6.44952357e-02f,+1.23967268e-01f,-2.60771990e-01f,-1.03593752e-01f,-8.82077962e-02f,
-2.95127690e-01f,+2.85575762e-02f,+1.01613559e-01f,+1.36194095e-01f,+1.01498991e-01f,
-1.83360483e-02f,-9.58223343e-02f,+3.53544764e-02f,-2.63473928e-01f,+5.51474094e-02f,
+2.72939324e-01f,-1.10198386e-01f,+1.63898572e-01f,+1.28555179e-01f,+3.14828157e-01f,
-2.83567850e-02f,-3.54844660e-01f,-1.32747978e-01f,+3.18195671e-01f,-2.33981505e-01f,
+1.61028653e-01f,-2.68667638e-01f,-3.21180671e-01f,+4.56463546e-02f,-7.75834396e-02f,
+2.91306973e-02f,-1.70084387e-01f,+2.81022880e-02f,-4.58327383e-02f,+6.80501983e-02f,
+1.81632131e-01f,+5.02815694e-02f,+5.33958077e-02f,-1.82606474e-01f,-6.92020282e-02f,
-8.32849275e-03f,+3.03199857e-01f,-1.35907337e-01f,+2.68904001e-01f,+3.30030546e-02f,
+5.68554662e-02f,-1.25158280e-01f,+2.35846475e-01f,-4.43927795e-02f,+3.04580778e-01f,
-2.00642958e-01f,+1.33025855e-01f,+8.71170163e-02f,+2.55258121e-02f,+2.17254445e-01f,
-3.94578231e-03f,-1.85604408e-01f,+1.40024006e-01f,-2.41232902e-01f,+4.52236086e-02f,
+2.68609136e-01f,-1.65985733e-01f,-2.01382414e-01f,-6.20711446e-02f,-9.25096869e-02f,
-2.39244699e-01f,+3.83586697e-02f,+6.11986592e-02f,-2.52773240e-02f,+2.37582400e-02f,
+2.13762254e-01f,-1.03309639e-02f,-2.94411927e-01f,+1.23134635e-01f,+3.51581983e-02f,
-2.65777707e-01f,+1.56881496e-01f,-1.59108654e-01f,+1.21635154e-01f,+3.42522971e-02f,
-1.64738029e-01f,+1.45125866e-01f,-1.66322649e-01f,+2.88077183e-02f,-5.18887304e-02f,
-2.62927152e-02f,+1.49758933e-02f,-5.05562648e-02f,-8.65547508e-02f,-9.24300402e-03f,
-1.58810347e-01f,-2.56961495e-01f,-2.43386194e-01f,+3.41296964e-03f,+1.08910307e-01f,
-2.31713410e-02f,+4.86722365e-02f,+2.08069496e-02f,+1.72304496e-01f,-2.55321056e-01f,
-1.40161542e-02f,+1.87378332e-01f,+9.32309180e-02f,-7.00913696e-03f,+6.67882189e-02f,
+1.71005771e-01f,-1.26202837e-01f,+1.38215944e-01f,+1.35234937e-01f,+4.98406664e-02f,
+7.38808466e-03f,-1.25496879e-01f,-5.17963134e-02f,-1.80404976e-01f,-8.56841654e-02f,
+2.01306149e-01f,+2.63192095e-02f,+1.08126707e-01f,-4.90158722e-02f,-1.14838861e-01f,
+8.43916833e-02f,+9.54092294e-02f,+1.62681043e-01f,-8.03939700e-02f,+1.66943058e-01f,
+1.08223565e-01f,+1.69046029e-01f,+1.31911054e-01f,+2.40129009e-02f,+7.42758363e-02f,
+1.15546085e-01f,-1.23049006e-01f,+1.98637515e-01f,+7.42618442e-02f,+5.50895296e-02f,
+8.40534049e-04f,+1.84361279e-01f,-1.87177256e-01f,-1.05726451e-01f,+1.91366121e-01f,
-9.56411064e-02f,-4.77193221e-02f,+3.66251692e-02f,+1.86088718e-02f,-1.38284341e-01f,
-1.31385565e-01f,+7.56157488e-02f,-1.42584249e-01f,-1.49336651e-01f,+3.37050483e-02f,
+1.64580986e-01f,+1.64367497e-01f,-1.45506203e-01f,+2.53313519e-02f,-1.27414808e-01f,
-8.48097354e-02f,+1.13443369e-02f,-1.90901712e-01f,+7.39670694e-02f,-3.34803499e-02f,
-1.45206869e-01f,+1.03524901e-01f,+2.28675112e-01f,-1.71567574e-01f,+1.82735119e-02f,
+3.92606435e-03f,-1.09362945e-01f,+9.01936740e-02f,-1.63868725e-01f,+1.04046203e-01f,
-2.24875122e-01f,-3.38970512e-01f,-7.18178898e-02f,-1.82151705e-01f,+2.17325836e-01f,
+2.68091977e-01f,-1.89371202e-02f,+1.78887591e-01f,+4.09006365e-02f,-1.57807454e-01f,
-1.21207960e-01f,+3.63336861e-01f,-1.07074156e-01f,+1.90165684e-01f,-4.44747653e-04f,
-1.85716689e-01f,+1.41423747e-01f,+1.86796680e-01f,-1.48627296e-01f,+3.54014933e-01f,
-1.14355236e-01f,+3.14186186e-01f,-1.27281263e-01f,-1.94689825e-01f,+3.73628318e-01f,
-3.59855965e-03f,-1.34089589e-03f,+4.30772491e-02f,+1.86457746e-02f,+4.68767703e-01f,
+2.54420012e-01f,-2.07543164e-01f,-2.35563502e-01f,+3.38754654e-01f,-1.93580538e-01f,
-1.20159864e-01f,-2.41567612e-01f,+2.33846337e-01f,+3.12035065e-02f,+1.62515849e-01f,
+3.75018790e-02f,+2.82917827e-01f,-1.68019474e-01f,+1.17284566e-01f,-6.63170666e-02f,
-1.47929505e-01f,-1.40804335e-01f,-1.45707861e-01f,-4.73550782e-02f,-2.34627873e-02f,
-8.00252333e-02f,+1.21938035e-01f,+8.84058625e-02f,-1.90084726e-01f,+1.38038605e-01f,
+2.85231650e-01f,+4.18841273e-01f,+1.30054936e-01f,+1.36349991e-01f,+3.81920069e-01f,
-2.56200898e-02f,+5.04002050e-02f,-2.63331145e-01f,+1.47866622e-01f,+3.18816870e-01f,
-1.21200085e-01f,-7.77416527e-02f,-1.69520780e-01f,-8.10806453e-02f,-1.32711306e-01f,
+9.19770151e-02f,+1.72831044e-01f,-1.40356049e-01f,+1.04555592e-01f,+1.20363399e-01f,
+3.10511887e-02f,+1.26971200e-01f,-7.16145337e-03f,+1.97087660e-01f,-1.35195136e-01f,
-1.71980202e-01f,-1.73427731e-01f,-7.72188678e-02f,+9.19555873e-02f,-1.70001209e-01f,
-7.91817084e-02f,-9.01658386e-02f,-1.37787551e-01f,-2.73385048e-02f,+8.78781527e-02f,
-1.24118015e-01f,-1.84006333e-01f,-6.13866746e-02f,-9.98452604e-02f,-1.08129531e-01f,
-1.01498775e-01f,+1.79409161e-01f,-5.29645383e-03f,+1.76588550e-01f,-1.09213896e-01f,
-1.40301138e-01f,+1.43036142e-01f,+1.57313049e-03f,+6.03644699e-02f,+8.28805119e-02f,
+7.12626725e-02f,+1.60155728e-01f,-1.81314945e-01f,+2.57429183e-02f,-1.78251743e-01f,
+4.33069766e-02f,-8.61617029e-02f,+1.01318821e-01f,+8.06353539e-02f,+1.04883030e-01f,
+1.91758201e-01f,+1.73219725e-01f,+2.97079384e-02f,+1.81044623e-01f,-1.11948669e-01f,
+1.85006872e-01f,+1.61038652e-01f,-3.88422608e-03f,+2.88846195e-02f,+1.85699895e-01f,
+1.67247251e-01f,-1.29422426e-01f,+5.12246490e-02f,+6.13314360e-02f,-9.90060344e-02f,
+6.57315552e-03f,+6.01318032e-02f,+1.77502975e-01f,+1.32994875e-01f,-5.87423146e-02f,
-6.91623688e-02f,-1.45453915e-01f,-1.44560054e-01f,-8.20512176e-02f,+1.52153060e-01f,
-1.33625180e-01f,-1.57837182e-01f,-9.02495254e-03f,-1.42160013e-01f,-3.16264965e-02f,
+2.57649273e-02f,-1.59888700e-01f,+9.96066779e-02f,-6.98255971e-02f,-5.28533123e-02f,
-1.95413642e-02f,+7.09244460e-02f,+1.32654399e-01f,+1.50107443e-01f,-9.80625227e-02f,
-1.75326914e-01f,-1.33908302e-01f,+8.54948536e-02f,-1.28170207e-01f,+1.14914849e-01f,
-1.16818659e-01f,+2.39214361e-01f,-3.77093069e-02f,-6.47943318e-02f,+2.19756901e-01f,
-5.58586940e-02f,-1.20429941e-01f,-1.61628395e-01f,-1.84715107e-01f,+2.11790130e-01f,
+1.61296889e-01f,-3.87153402e-02f,+1.64473038e-02f,+2.57470101e-01f,-1.13111995e-01f,
-2.04305738e-01f,-3.08179427e-02f,-1.82570633e-03f,-9.30301286e-03f,+2.48100609e-01f,
+1.25943646e-01f,-2.66335648e-03f,+3.50838788e-02f,+3.07180267e-02f,-1.01973325e-01f,
-6.89612553e-02f,-8.95282179e-02f,+1.72736079e-01f,+1.66166753e-01f,-1.35857224e-01f,
+1.52716815e-01f,+5.95746152e-02f,-1.76970735e-02f,-1.89517483e-01f,-5.97887300e-02f,
+4.38727662e-02f,+2.92830378e-01f,-8.41661841e-02f,+6.40910491e-03f,+1.32029783e-02f,
+1.61341265e-01f,+5.16368747e-02f,+2.55286470e-02f,+9.71995145e-02f,+3.48612405e-02f,
-2.09848389e-01f,-1.09380186e-02f,+1.56074623e-02f,+7.91170523e-02f,-1.04745589e-01f,
-1.65445015e-01f,+1.50142506e-01f,-1.89990491e-01f,-9.41518024e-02f,-1.42956853e-01f,
-3.76038253e-02f,-1.69897825e-02f,+8.12168866e-02f,-1.88360125e-01f,+1.01473466e-01f,
-1.50733948e-01f,+1.33965716e-01f,-1.51998132e-01f,-5.53438514e-02f,+5.33416122e-02f,
-8.46305043e-02f,-8.18031132e-02f,+4.54827249e-02f,-1.76947325e-01f,-7.79203027e-02f,
+1.37135431e-01f,+1.87991872e-01f,-2.42724270e-02f,+1.45009011e-02f,+1.84748575e-01f,
-1.09038219e-01f,+1.33363411e-01f,+1.49520740e-01f,+1.02160111e-01f,+7.11069852e-02f,
+1.86552629e-01f,+1.58201531e-01f,+4.23083156e-02f,-1.14869237e-01f,-1.78122804e-01f,
+1.52362391e-01f,+3.07501853e-02f,+1.03475824e-01f,-1.37911454e-01f,+1.43560469e-02f,
+1.74378768e-01f,-1.84431478e-01f,+6.24982864e-02f,-1.95364878e-01f,-6.78285360e-02f,
+3.81846875e-02f,+1.01538911e-01f,+1.02612987e-01f,+5.26306182e-02f,-5.12844175e-02f,
+9.22159702e-02f,+1.73533157e-01f,-1.06185138e-01f,-1.60790369e-01f,+1.60024747e-01f,
+1.21695861e-01f,-4.30859774e-02f,-5.83886504e-02f,+2.58522332e-02f,+1.54492691e-01f,
-1.21041156e-01f,+5.62446266e-02f,+2.14853138e-02f,+3.64724398e-02f,-1.28575474e-01f,
+1.27187058e-01f,+7.37211555e-02f,-1.20516151e-01f,+1.60073921e-01f,+1.89105138e-01f,
-8.13653618e-02f,+6.23677336e-02f,-3.69130485e-02f,+2.96754157e-03f,+9.96557996e-02f,
+1.28201053e-01f,+1.19891547e-01f,+2.01550238e-02f,-1.11411028e-01f,+2.45973632e-01f,
-1.32375777e-01f,-3.67012317e-03f,-1.34711310e-01f,-6.60747737e-02f,-1.39550969e-01f,
+9.59310383e-02f,-1.83161005e-01f,+1.39327809e-01f,-2.95208599e-02f,+2.17274763e-02f,
-7.62425810e-02f,-1.63052529e-01f,+1.90300390e-01f,+1.29761219e-01f,-4.09555584e-02f,
+1.28457800e-01f,+1.47554412e-01f,+2.14900866e-01f,-8.33488479e-02f,+1.71211630e-01f,
+1.06340066e-01f,-9.57376063e-02f,-9.73067805e-02f,-1.45175353e-01f,-1.39102802e-01f,
-1.40108973e-01f,-1.69796854e-01f,-1.26180604e-01f,-1.46606639e-01f,+6.75383881e-02f,
+3.98426689e-02f,-1.30305022e-01f,-4.99630608e-02f,-2.14890018e-01f,-5.62593788e-02f,
+2.30264530e-01f,+1.65961757e-01f,-1.79456860e-01f,-1.58363342e-01f,+1.35159060e-01f,
-1.35806769e-01f,+1.78818718e-01f,-1.68900788e-01f,-1.04310490e-01f,-2.15715393e-01f,
-1.84539646e-01f,-6.75726533e-02f,-3.00707985e-02f,-4.76854406e-02f,+2.15305537e-01f,
-6.75488785e-02f,+2.10185587e-01f,-9.36234668e-02f,+1.20986234e-02f,+7.76903108e-02f,
-9.26212594e-02f,+3.16387862e-02f,+1.80283070e-01f,+1.92996129e-01f,+1.88759297e-01f,
+2.05025747e-01f,+1.34134009e-01f,+2.40424454e-01f,-6.62986073e-04f,-1.19352698e-01f,
-1.26927532e-02f,-1.58381253e-01f,-2.02523783e-01f,-1.03914410e-01f,+4.95126471e-02f,
-3.63861024e-02f,-1.20401643e-01f,+1.68932617e-01f,-8.30619335e-02f,+2.68726110e-01f,
+1.33805409e-01f,-2.00911045e-01f,-6.86505809e-02f,+1.96981579e-01f,-1.61929518e-01f,
-1.73014775e-01f,-1.53193668e-01f,-2.19176747e-02f,-3.94968167e-02f,-3.87538970e-02f,
-1.59354955e-01f,+7.65085071e-02f,+1.05143851e-02f,+7.64115751e-02f,-1.74411297e-01f,
-1.59972414e-01f,+2.17145175e-01f,+1.15504868e-01f,-1.71494365e-01f,-7.62029644e-03f,
+2.27292329e-01f,+1.19364910e-01f,-1.15890220e-01f,-1.79105297e-01f,+1.63205728e-01f,
-2.02934742e-01f,-1.31702259e-01f,+2.94745803e-01f,-6.19622655e-02f,-1.55676976e-01f,
+4.41501513e-02f,-1.52904853e-01f,-7.45256245e-02f,+2.46258639e-02f,+1.02324255e-01f,
+1.42054275e-01f,-5.63679263e-02f,+2.26240203e-01f,+4.66596801e-03f,-1.98679622e-02f,
+1.43867508e-02f,+2.13124186e-01f,-7.76063576e-02f,+5.63057400e-02f,-2.27934495e-01f,
-9.24002379e-02f,+2.32474327e-01f,+1.66510433e-01f,+5.96380606e-02f,-5.85956015e-02f,
+1.74723238e-01f,+9.88970995e-02f,-7.29869008e-02f,-1.11748271e-01f,+1.00745313e-01f,
-1.56910449e-01f,-1.94920167e-01f,+8.61824304e-02f,+1.08564354e-01f,+1.97665431e-02f,
+1.91598475e-01f,-1.05697876e-02f,-1.58493537e-02f,-1.22139938e-01f,-7.39060566e-02f,
-7.07313837e-03f,+1.40200287e-01f,-7.86623359e-03f,+6.16574325e-02f,-3.17478478e-02f,
+1.74245954e-01f,-1.14710622e-01f,+2.50157304e-02f,+1.18982852e-01f,+1.59616858e-01f,
-9.14391577e-02f,+3.45431641e-02f,-1.15962423e-01f,+1.46490231e-01f,+1.61454573e-01f,
+2.59461701e-02f,+5.99095970e-02f,-9.99804959e-02f,+1.27291843e-01f,-5.46497926e-02f,
+1.45593390e-01f,+8.17294717e-02f,-4.51055169e-02f,-2.42325775e-02f,-2.28679162e-02f,
+8.36407393e-02f,-5.62080108e-02f,-1.74615681e-01f,+1.47098273e-01f,-1.57407984e-01f,
+1.60774589e-01f,+1.54337138e-01f,-1.65348172e-01f,-1.67603776e-01f,+1.05406102e-02f,
+2.71852501e-02f,+1.40860781e-01f,-1.49439871e-01f,-1.97333246e-01f,+4.95739430e-02f,
-7.14984462e-02f,-1.83289021e-01f,-1.03952274e-01f,-1.28052592e-01f,+9.79544371e-02f,
-2.50410251e-02f,-1.15549594e-01f,+1.44496530e-01f,+2.92405412e-02f,-1.90706909e-01f,
+8.45715255e-02f,+1.28732204e-01f,-1.67533129e-01f,+4.28663800e-03f,+1.74681515e-01f,
-9.56850648e-02f,-2.18946300e-02f,+1.06217518e-01f,+1.56672999e-01f,-1.93579197e-01f,
-2.52422243e-01f,+3.86840031e-02f,+2.00000452e-03f,-1.14777982e-01f,+1.20417178e-01f,
+1.01379678e-01f,-9.67259780e-02f,+2.86490291e-01f,-4.80835475e-02f,-2.52589136e-01f,
-9.34689417e-02f,+1.53696209e-01f,-2.26465940e-01f,+1.36789605e-01f,-4.03614268e-02f,
-9.02559459e-02f,-5.44044795e-03f,+1.68660015e-01f,-6.63521588e-02f,+2.63093442e-01f,
-1.24139056e-01f,+1.97977677e-01f,-2.27265075e-01f,-1.74893185e-01f,+2.80653656e-01f,
-6.01507537e-02f,-1.55998215e-01f,+1.28753379e-01f,+4.54753190e-02f,+1.32855371e-01f,
-3.05657629e-02f,-2.59528786e-01f,+1.49653517e-02f,+1.80294126e-01f,+2.08374709e-02f,
+5.27916886e-02f,+8.43203068e-02f,+1.65629268e-01f,-1.93790093e-01f,+1.12691469e-01f,
-6.52163699e-02f,-4.06048540e-03f,+8.26472118e-02f,+1.90720588e-01f,+1.26507536e-01f,
-1.01421587e-02f,-6.40202910e-02f,-3.10819857e-02f,+1.45349085e-01f,-8.32206383e-02f,
-1.36120185e-01f,+5.02029844e-02f,+5.56286201e-02f,-2.68485606e-01f,+1.61287397e-01f,
+1.27349719e-01f,+2.36776963e-01f,-1.40131205e-01f,-1.36397779e-01f,+2.80360013e-01f,
+1.11599542e-01f,-2.21678093e-02f,+7.61384964e-02f,+2.89303303e-01f,-3.55968811e-02f,
-6.44104630e-02f,+9.78155285e-02f,-2.05260471e-01f,-2.80591398e-02f,-2.73623079e-01f,
-2.14212477e-01f,-2.14429311e-02f,-1.40210703e-01f,-2.62410015e-01f,+2.63434231e-01f,
+1.89637229e-01f,-1.81457981e-01f,+3.60922039e-01f,-9.16500911e-02f,-1.72302127e-01f,
-2.30632611e-02f,+1.01349860e-01f,-1.52739435e-01f,+7.23760426e-02f,-7.07562119e-02f,
-2.64919490e-01f,-3.73490900e-02f,+3.46443146e-01f,-1.23599023e-02f,+1.01777390e-01f,
+1.10254437e-02f,+3.16411644e-01f,-6.11361824e-02f,-2.27363929e-02f,+3.88389602e-02f,
-2.38241814e-02f,-5.83801866e-02f,+1.70213029e-01f,-2.84874707e-01f,+3.44717711e-01f,
+2.81566262e-01f,-1.07827269e-01f,-1.54997781e-01f,+3.84631902e-01f,+3.99462618e-02f,
+6.17709244e-03f,-1.48154035e-01f,+3.49970669e-01f,+1.30131304e-01f,+3.63968760e-01f,
+3.11953366e-01f,+3.69294614e-01f,-6.42030016e-02f,+3.58038634e-01f,+1.09714225e-01f,
-1.69299960e-01f,-1.52298659e-02f,+1.44394442e-01f,-1.52284447e-02f,+7.99491853e-02f,
+6.76631778e-02f,+1.22083277e-01f,-2.44975090e-01f,-1.86254606e-01f,+3.42134476e-01f,
+3.19480538e-01f,+1.34552792e-01f,-1.42338768e-01f,+1.58228382e-01f,+2.08987072e-01f,
+1.19029798e-01f,-5.36559746e-02f,-4.81799282e-02f,+3.21810842e-01f,+2.07400426e-01f,
+8.08282197e-02f,+3.39535587e-02f,-1.20039165e-01f,-9.43883285e-02f,-6.38543144e-02f,
+2.58315522e-02f,-8.45951438e-02f,+9.42849889e-02f,+2.14434013e-01f,+1.27722353e-01f,
-2.81540006e-02f,-9.31884646e-02f,+1.60788953e-01f,-6.68812469e-02f,+1.10649347e-01f,
+4.63201758e-03f,+6.82577193e-02f,+1.68078199e-01f,-3.78775969e-03f,-1.15880951e-01f,
+4.30548191e-02f,-1.25174090e-01f,+1.58136323e-01f,+6.88925833e-02f,+9.43500772e-02f,
+1.31312355e-01f,+2.03489453e-01f,+1.82621732e-01f,+8.13054442e-02f,+5.26155420e-02f,
+1.69418722e-01f,+1.03608266e-01f,-6.53175116e-02f,-1.04326658e-01f,+6.87433332e-02f,
-2.29419917e-02f,+9.78751108e-02f,-1.23232603e-01f,+9.15026367e-02f,+2.28459507e-01f,
+1.74196184e-01f,+7.73234516e-02f,-1.93762273e-01f,+1.47609562e-01f,+1.31237075e-01f,
+5.31643778e-02f,-7.76030198e-02f,+1.87620386e-01f,-7.13031292e-02f,+3.12890150e-02f,
-4.07656282e-02f,-1.13843828e-02f,-1.89919382e-01f,-1.52539954e-01f,+4.33130302e-02f,
+2.46704295e-01f,-1.27374396e-01f,+7.09975064e-02f,+9.87572297e-02f,-1.60932675e-01f,
-1.05404630e-01f,+8.97990167e-02f,+1.51950745e-02f,-7.73081183e-02f,+1.88923582e-01f,
+2.85481419e-02f,+8.47537369e-02f,+8.82223547e-02f,+1.65160611e-01f,+8.15960392e-02f,
-1.90976053e-01f,-1.16912372e-01f,+1.14979940e-02f,+4.95169796e-02f,-1.60464796e-03f,
+5.96849211e-02f,+6.30278885e-02f,+1.43068701e-01f,+2.00526528e-02f,+6.18686676e-02f,
+2.52720773e-01f,+1.93339467e-01f,+1.18389189e-01f,+4.99248430e-02f,-1.15577884e-01f,
+7.11840019e-02f,+1.35049358e-01f,+1.35246646e-02f,-6.08161427e-02f,+1.54419113e-02f,
+1.13462612e-01f,-8.62367377e-02f,+1.00369409e-01f,+1.47107512e-01f,+3.48364897e-02f,
+1.62102178e-01f,+1.69615790e-01f,-1.79064080e-01f,+7.41195008e-02f,-9.96555462e-02f,
-7.78181702e-02f,-1.41850024e-01f,+6.11610524e-02f,-4.75569516e-02f,+6.62071481e-02f,
+2.11060613e-01f,+1.34298310e-01f,+6.27482980e-02f,+4.88993078e-02f,+6.56545814e-03f,
-3.99444103e-02f,-7.28326589e-02f,-1.31674439e-01f,+1.87365174e-01f,+2.14603603e-01f,
-6.62742183e-02f,+1.68006554e-01f,+7.16581717e-02f,-8.48285481e-02f,-7.23187029e-02f,
-1.65950537e-01f,+5.72519451e-02f,-7.19631091e-02f,+9.53342766e-02f,-1.76264763e-01f,
+5.47924824e-03f,-6.70045707e-03f,-1.52854353e-01f,+8.42552911e-03f,+2.11798579e-01f,
+2.63252676e-01f,+6.40483648e-02f,-1.30660739e-03f,-1.36025762e-02f,+1.14799500e-01f,
+1.27892196e-01f,-7.37653822e-02f,-1.89776853e-01f,-8.13746545e-03f,+1.55400559e-01f,
-1.95940286e-01f,-1.76918611e-01f,-1.14774510e-01f,-7.14430958e-02f,-2.43247598e-02f,
+1.82895750e-01f,-4.95447740e-02f,+8.32259748e-03f,-9.34192687e-02f,-6.06942587e-02f,
-1.57451332e-01f,+3.13280919e-03f,-6.48625568e-02f,-8.63068085e-03f,+1.79852903e-01f,
+2.68924385e-01f,-1.57742456e-01f,-6.07844852e-02f,+1.03907257e-01f,-8.64848681e-03f,
+1.57768235e-01f,-1.00445054e-01f,-6.09105965e-03f,+1.75167620e-01f,-2.08473071e-01f,
-2.31095105e-02f,-8.22271630e-02f,+1.08666696e-01f,-9.55118388e-02f,-1.16685644e-01f,
+2.38083433e-02f,-1.15773268e-01f,+1.65633321e-01f,+1.77436471e-01f,+1.08319715e-01f,
-2.30793580e-01f,+3.92994657e-02f,-3.09750214e-02f,-1.21022448e-01f,+2.42107868e-01f,
+1.33457452e-01f,+8.18930101e-03f,-2.37104878e-01f,+8.84604827e-02f,+1.28157914e-01f,
-4.08521928e-02f,-2.14889377e-01f,+1.36463255e-01f,+3.39400545e-02f,-1.76185220e-02f,
+1.59131661e-01f,+1.13636348e-02f,+8.64008963e-02f,-1.91682085e-01f,-4.87939008e-02f,
+2.20362276e-01f,+1.01399170e-02f,-9.02960375e-02f,+1.19101532e-01f,+7.32535645e-02f,
-2.28904083e-01f,-1.13184221e-01f,-1.66880786e-01f,+2.51668785e-02f,-4.84867319e-02f,
-3.30572464e-02f,+1.59509033e-01f,+6.77473098e-02f,-5.11424206e-02f,+2.41526723e-01f,
-1.23249665e-01f,+7.01116920e-02f,+1.41463056e-01f,+1.04036890e-01f,+1.45027146e-01f,
+2.92090833e-01f,+2.60829866e-01f,+2.57789969e-01f,+3.83843958e-01f,-2.69861162e-01f,
-3.76737505e-01f,+1.05628654e-01f,-3.31068784e-01f,+3.22461307e-01f,+2.65641212e-01f,
+5.30340731e-01f,-3.67044121e-01f,+3.64835769e-01f,-2.28527084e-01f,+4.19311255e-01f,
+2.31035665e-01f,-3.61513123e-02f,+7.84710124e-02f,+4.18210864e-01f,-1.06851064e-01f,
-1.84918627e-01f,-4.13381934e-01f,+2.99721181e-01f,+3.58519673e-01f,-1.39351517e-01f,
+2.35421322e-02f,-8.62600133e-02f,+1.30889043e-01f,+4.65480626e-01f,-2.04633415e-01f,
-3.15784574e-01f,+3.49440008e-01f,+3.38396370e-01f,-1.07237026e-01f,+1.90895796e-01f,
+4.86398071e-01f,-2.00369939e-01f,-2.49138936e-01f,+1.43113628e-01f,+1.19681649e-01f,
-2.85813719e-01f,-2.30138659e-01f,+4.23074931e-01f,-6.77480772e-02f,-1.26547351e-01f,
+1.52273238e-01f,-9.82049406e-02f,+1.09634787e-01f,-1.46116450e-01f,-8.89945030e-02f,
+1.77151024e-01f,+4.43469211e-02f,-3.34542827e-03f,+3.38765711e-01f,-4.07311350e-01f,
-3.71589303e-01f,+6.94717407e-01f,+1.89254686e-01f,+1.48997620e-01f,-3.33969176e-01f,
-4.82151508e-02f,+4.45792615e-01f,+1.42791033e-01f,-2.89536029e-01f,+6.15789175e-01f,
+1.02704309e-01f,+4.37462032e-01f,+3.15896273e-01f,-9.59728211e-02f,+1.25841424e-01f,
-1.77720442e-01f,-1.06488839e-01f,-9.68656242e-02f,+1.11587405e-01f,-8.15417692e-02f,
+1.71271905e-01f,-1.86657626e-02f,-4.98220213e-02f,+2.24010330e-02f,-1.97437882e-01f,
-4.25233766e-02f,+9.39840600e-02f,+1.54341459e-01f,-3.24512459e-02f,-1.53383985e-01f,
-1.32790534e-02f,+1.64140746e-01f,+1.04345731e-01f,-3.94666493e-02f,-1.13902755e-01f,
+1.01766646e-01f,-2.92906407e-02f,-1.49659783e-01f,+2.23526936e-02f,+2.66776769e-03f,
-7.31947348e-02f,-2.04196870e-02f,+7.17376545e-03f,-1.82033733e-01f,+1.94265977e-01f,
+2.49863997e-01f,-3.30427140e-02f,+1.50400326e-01f,+1.75045013e-01f,+9.77715626e-02f,
-2.59056222e-02f,-1.01733848e-01f,+4.92431149e-02f,-3.65058295e-02f,+2.53457785e-01f,
-7.93161765e-02f,+1.35130420e-01f,-1.41183212e-01f,+1.12147190e-01f,+6.32726178e-02f,
-5.06331101e-02f,+7.99719542e-02f,-8.25432613e-02f,-4.42983732e-02f,+1.09198511e-01f,
+1.45572945e-02f,+6.89039677e-02f,-3.08788400e-02f,-1.79806754e-01f,+3.17943096e-02f,
+1.75702497e-01f,+2.88528621e-01f,-1.67930588e-01f,-1.90636162e-02f,+1.65615007e-01f,
+1.09256759e-01f,+1.11746751e-01f,-3.64116356e-02f,+2.83393194e-03f,+1.54750824e-01f,
-1.65044621e-01f,-3.78208011e-02f,-1.76689729e-01f,-1.80045351e-01f,+1.38422593e-01f,
+7.78778121e-02f,+3.80528569e-02f,+2.20042780e-01f,+1.52302012e-01f,-8.68549645e-02f,
-1.55109227e-01f,-1.11335866e-01f,-1.64321557e-01f,+1.01292230e-01f,+6.53399900e-02f,
-8.67096707e-02f,+6.62744790e-02f,-3.96070704e-02f,-3.28132659e-02f,+1.77668110e-01f,
+6.60274327e-02f,+1.57540381e-01f,+6.49948642e-02f,+1.63640916e-01f,+1.02748312e-01f,
+5.59602752e-02f,+6.32935986e-02f,+1.60861775e-01f,+4.05580783e-03f,-1.08917966e-01f,
+3.27210873e-02f,-1.83962569e-01f,+1.04768991e-01f,+8.09944421e-02f,-1.82878718e-01f,
+1.85411200e-01f,+1.49362668e-01f,+1.54818580e-01f,-2.00794324e-01f,+1.00459844e-01f,
-8.79501253e-02f,-1.69852138e-01f,+1.06764689e-01f,-7.91648589e-03f,+9.54731181e-02f,
-4.51145731e-02f,+1.52145013e-01f,+8.31371173e-02f,+2.89026815e-02f,-1.86037019e-01f,
-7.10390806e-02f,+1.24518149e-01f,-1.58810377e-01f,-1.49519697e-01f,+1.45436963e-02f,
+1.52946576e-01f,-7.93001875e-02f,+1.14773855e-01f,+1.04071610e-01f,-1.04371339e-01f,
-1.29128993e-02f,+2.25004479e-01f,-2.54287515e-02f,+3.68665233e-02f,-1.61603719e-01f,
+1.47203784e-02f,-1.60080031e-01f,-4.40839231e-02f,+1.55551061e-01f,+2.08382700e-02f,
+9.94249154e-03f,-1.37850046e-01f,+7.09671974e-02f,-1.01100057e-01f,-1.94128919e-02f,
+1.87108889e-02f,+1.42098367e-02f,+1.43831000e-01f,-5.66909835e-02f,-4.44575734e-02f,
+1.33464977e-01f,-9.40740295e-03f,+8.22678357e-02f,+1.13355413e-01f,+9.34922546e-02f,
+4.31405753e-02f,+8.96088332e-02f,-3.52981687e-03f,+1.65133938e-01f,+1.03912577e-02f,
-4.21617478e-02f,-3.16653438e-02f,-1.42305970e-01f,-1.64211392e-01f,+1.78504601e-01f,
-1.57165363e-01f,-9.75566208e-02f,-1.08065113e-01f,-2.33049989e-02f,+3.52115110e-02f,
+9.09184664e-02f,-1.82682768e-01f,-1.94889277e-01f,+1.14631943e-01f,+1.86418936e-01f,
+3.75980400e-02f,+1.59239039e-01f,+7.67575726e-02f,-1.59765095e-01f,+1.14152670e-01f,
-5.74557185e-02f,-1.12994589e-01f,+7.82590136e-02f,+6.30414039e-02f,-1.32032678e-01f,
-6.00195900e-02f,-1.09982163e-01f,-6.31795079e-03f,-6.37405217e-02f,+1.55707285e-01f,
-1.59455180e-01f,-2.95326412e-02f,-1.38788044e-01f,-5.29411808e-02f,-1.88364401e-01f,
+1.93552300e-01f,-1.97022766e-01f,-3.50657268e-03f,-1.19164482e-01f,+4.75603901e-02f,
+1.93108141e-01f,+1.37124896e-01f,+1.60570458e-01f,-2.32943762e-02f,+1.37516752e-01f,
-1.74441546e-01f,+3.33562195e-02f,+1.47668973e-01f,+8.32567066e-02f,-3.28423083e-02f,
+2.43957639e-02f,-2.11641509e-02f,-5.70481345e-02f,-1.26164615e-01f,-1.44932866e-01f,
-3.62366438e-02f,-9.93977264e-02f,+1.12011567e-01f,+1.84605345e-01f,-3.69639210e-02f,
+6.34706542e-02f,-9.54681039e-02f,-1.81328848e-01f,-1.26063347e-01f,-5.34761744e-03f,
+1.86029300e-01f,-2.07136959e-01f,+6.79418296e-02f,+1.18177131e-01f,+1.33439988e-01f,
+2.05412969e-01f,-1.62545130e-01f,+1.22877993e-01f,-1.19320817e-01f,+1.03457130e-01f,
-4.77600433e-02f,+1.51880741e-01f,-4.37528230e-02f,-6.36602892e-03f,-2.29803652e-01f,
-1.35420471e-01f,+1.89581349e-01f,-1.29486412e-01f,-9.71122459e-02f,-1.74003705e-01f,
-1.27237335e-01f,-3.82429641e-03f,-4.91610952e-02f,-1.33035481e-02f,-7.51276836e-02f,
+2.19444796e-01f,+3.97979654e-02f,-1.19702891e-01f,-1.76333010e-01f,+1.90643162e-01f,
-1.09021083e-01f,-1.54192105e-01f,+1.13283962e-01f,-8.75019282e-02f,-1.51111841e-01f,
+9.72424448e-02f,+1.16039231e-01f,-1.60386041e-02f,+1.20970950e-01f,-1.79385960e-01f,
+1.20994650e-01f,+9.74043738e-04f,+7.38760978e-02f,+3.83479893e-02f,-5.64035028e-02f,
-1.39594272e-01f,+2.18149960e-01f,+4.16906513e-02f,-1.15851693e-01f,+1.01698516e-02f,
+8.07955712e-02f,+6.51683286e-03f,+2.87829787e-02f,-9.13979188e-02f,+5.93071654e-02f,
+2.43536662e-02f,+2.79452540e-02f,-4.32897434e-02f,+9.37939510e-02f,+1.16138853e-01f,
-8.19029100e-03f,+1.76350549e-01f,-3.04814074e-02f,+2.09090203e-01f,-1.77264437e-02f,
+1.36562482e-01f,+1.37142539e-01f,-1.53888106e-01f,+1.74349800e-01f,+2.01161593e-01f,
-5.35272136e-02f,-1.80905610e-01f,+1.08810045e-01f,+5.97044751e-02f,-5.84883355e-02f,
-8.94895494e-02f,+1.66260660e-01f,+1.79720402e-01f,+1.08066119e-01f,+7.17167482e-02f,
-8.92434120e-02f,+4.50682938e-02f,+4.68415730e-02f,+1.78622678e-01f,+1.10062204e-01f,
+7.29681328e-02f,+3.22086364e-02f,+1.01364419e-01f,+2.36850142e-01f,-7.71918520e-02f,
-3.11669670e-02f,+2.32892126e-01f,+2.14377001e-01f,+1.40098065e-01f,-1.27431542e-01f,
-5.02533652e-02f,+1.56847864e-01f,-1.08991280e-01f,+6.87989742e-02f,+4.30980958e-02f,
+1.06482655e-01f,-9.18317288e-02f,+6.90139160e-02f,-7.12292418e-02f,-1.83307350e-01f,
+1.41551033e-01f,-1.67230695e-01f,+1.79552883e-01f,-1.65333658e-01f,-8.73847827e-02f,
-7.03265145e-02f,-1.59202188e-01f,+9.71896052e-02f,+4.22405638e-02f,+1.69068769e-01f,
+2.79647135e-03f,-4.88659553e-02f,-4.23237309e-02f,-1.49117514e-01f,+7.86904320e-02f,
+1.19077917e-02f,+1.23837799e-01f,+3.22906822e-02f,+1.31875709e-01f,+2.36601442e-01f,
-1.98189318e-01f,+2.52333045e-01f,+1.35237360e-02f,-1.81398898e-01f,+1.11281708e-01f,
+3.38100642e-02f,+2.37049118e-01f,+1.25369191e-01f,+1.44516155e-01f,+1.45660326e-01f,
-1.14201464e-01f,+2.40202434e-02f,+8.18030983e-02f,+5.25755994e-02f,+2.28657529e-01f,
+1.62779927e-01f,+2.77231131e-02f,+1.77294984e-01f,-1.69235125e-01f,-5.70495799e-02f,
+6.56968057e-02f,-7.39680082e-02f,-9.94107574e-02f,-3.18375044e-02f,-1.73455730e-01f,
-3.84033956e-02f,-1.09261833e-01f,+4.05074796e-03f,+1.63458958e-01f,-2.02985093e-01f,
-9.60948914e-02f,-1.23150349e-02f,-8.17045756e-03f,+9.70645770e-02f,-2.52627850e-01f,
-2.15782166e-01f,+2.14261964e-01f,-9.25456658e-02f,-1.84207141e-01f,+5.35104312e-02f,
+6.18787743e-02f,+3.72409001e-02f,-1.73565090e-01f,+7.90322870e-02f,+1.62154302e-01f,
-8.52782056e-02f,+4.57667410e-02f,+2.58714445e-02f,-1.64217979e-01f,-3.38773206e-02f,
-9.78828222e-02f,-7.92978033e-02f,-1.62070036e-01f,-1.46029696e-01f,-9.29913744e-02f,
-7.01877698e-02f,-4.87814099e-02f,+2.25200728e-01f,-1.15545504e-01f,+4.93246503e-02f,
-1.64153680e-01f,-1.13197289e-01f,+1.70944765e-01f,+1.50795460e-01f,+1.75219312e-01f,
+8.03768784e-02f,+1.79401040e-01f,+1.14532359e-01f,+5.35311922e-02f,+1.39498543e-02f,
-9.98095274e-02f,-4.66914475e-02f,+2.76873931e-02f,+1.59894258e-01f,+1.65917024e-01f,
-1.09711580e-01f,+9.41299573e-02f,+8.36094692e-02f,+2.62074381e-01f,+4.57874723e-02f,
-6.18958026e-02f,+1.76828146e-01f,-1.00313514e-01f,+4.45799939e-02f,+1.19642906e-01f,
+7.24187493e-02f,+7.34167546e-02f,+1.90415666e-01f,-9.11775380e-02f,+7.80438930e-02f,
+2.21729934e-01f,-6.81076795e-02f,+6.77215382e-02f,-3.49091589e-02f,-1.87389672e-01f,
+2.29932312e-02f,-1.77075103e-01f,-7.95939863e-02f,-9.45717841e-02f,-6.27428368e-02f,
+1.54554218e-01f,-1.75202236e-01f,-1.82146639e-01f,-8.25986862e-02f,-1.50166422e-01f,
-3.32765877e-02f,-1.68983894e-03f,-1.22769676e-01f,-1.74229890e-01f,+9.97260865e-03f,
-3.79854031e-02f,-5.70627078e-02f,-6.03340007e-02f,-1.02651693e-01f,+6.96076378e-02f,
+1.08860515e-01f,+5.00687100e-02f,+2.44388416e-01f,-1.04079008e-01f,-5.03850840e-02f,
-8.51785112e-03f,+1.27199173e-01f,-2.27332264e-02f,+1.53049037e-01f,-1.83597967e-01f,
+2.00701237e-01f,+1.72033861e-01f,+1.60361920e-02f,-8.76380056e-02f,-4.82091308e-02f,
-7.70215616e-02f,+2.78081566e-01f,-2.01189201e-02f,-4.80463132e-02f,+1.30690113e-01f,
-2.93211341e-02f,+1.71694383e-01f,+2.49613121e-01f,-8.33701938e-02f,+5.25831468e-02f,
-1.84573546e-01f,+2.35620663e-01f,+1.31366178e-01f,-1.97744027e-01f,-1.16347909e-01f,
-1.51083782e-01f,+2.46004038e-05f,-2.32917160e-01f,+5.97909465e-02f,-3.87286325e-03f,
+2.28961870e-01f,+5.77306338e-02f,-6.92557693e-02f,-8.21945965e-02f,-1.68218657e-01f,
+1.86402828e-01f,+8.60829726e-02f,-4.26250603e-03f,-6.20916001e-02f,+8.89155194e-02f,
-1.45280346e-01f,+3.02006602e-02f,-5.07951751e-02f,-1.48187438e-02f,+2.29551539e-01f,
-1.23440605e-02f,+7.65384808e-02f,-1.38341978e-01f,-7.58962855e-02f,+1.98403420e-03f,
-1.62812740e-01f,+1.63972035e-01f,-1.37768194e-01f,+1.16631992e-01f,+1.16287924e-01f,
+2.07178757e-01f,-8.48421752e-02f,+3.99639308e-02f,-1.15825444e-01f,+1.62962481e-01f,
+1.48969190e-02f,-2.26328243e-02f,-1.14935853e-01f,-4.22864705e-02f,-4.21283813e-03f,
+1.16731130e-01f,+1.90750137e-01f,+3.87546681e-02f,-9.59661230e-02f,+1.62874714e-01f,
-1.37649477e-01f,-5.15822172e-02f,+7.92549700e-02f,-9.41329636e-03f,-7.64581859e-02f,
-8.28551129e-02f,+5.57992905e-02f,+1.42383099e-01f,-1.32940844e-01f,-5.65056428e-02f,
-5.11247898e-03f,+5.58603592e-02f,-1.85165390e-01f,+5.37325814e-02f,-4.83705439e-02f,
-1.96109451e-02f,+4.95129451e-02f,+7.79940635e-02f,-8.05225521e-02f,+2.13255867e-01f,
+1.16604634e-01f,-4.88218516e-02f,-1.89362869e-01f,-1.58636510e-01f,+1.01585589e-01f,
-3.99446972e-02f,-1.37553170e-01f,-1.47922680e-01f,+8.21022019e-02f,-5.26492335e-02f,
-6.37401715e-02f,+7.42792115e-02f,+1.79641396e-01f,+1.74974903e-01f,-2.17520948e-02f,
-6.68614358e-02f,-8.95309970e-02f,-8.95414054e-02f,+1.95248738e-01f,+1.82905570e-02f,
+1.49786502e-01f,-4.09634002e-02f,+2.75772482e-01f,-2.99775526e-02f,-5.20371273e-02f,
-1.69450819e-01f,-4.04334217e-02f,-1.33830637e-01f,+1.20821998e-01f,+1.53965345e-02f,
-3.35494094e-02f,-9.37427506e-02f,-9.23927128e-02f,+1.51802242e-01f,-1.72577277e-01f,
-9.92520079e-02f,-1.07609771e-01f,-1.35497317e-01f,-1.47297859e-01f,-1.29018739e-01f,
+4.24269848e-02f,-1.46327317e-01f,+1.42957196e-01f,-7.11154416e-02f,+6.71631843e-02f,
+6.88009858e-02f,+1.00910023e-01f,+2.09938869e-01f,-1.97474599e-01f,-1.34638632e-02f,
+1.91124499e-01f,-1.98383734e-01f,-1.16621800e-01f,-1.02286942e-01f,-1.25319907e-03f,
-6.31872937e-02f,+7.20559135e-02f,-1.22053936e-01f,+2.58025620e-02f,+4.07492034e-02f,
-7.78688816e-03f,+9.72442552e-02f,+1.83914617e-01f,-4.90575917e-02f,-1.84941143e-02f,
-7.96366408e-02f,+7.29416162e-02f,+6.59648180e-02f,-1.54827565e-01f,+2.43817300e-01f,
+1.08078890e-01f,+1.22767292e-01f,+1.38824210e-01f,-3.63473594e-02f,-8.10055882e-02f,
+5.89944012e-02f,-1.84969425e-01f,-2.11309455e-02f,+4.55696099e-02f,-2.76135448e-02f,
-6.11135215e-02f,+1.99667841e-01f,-5.27341254e-02f,-3.03757321e-02f,-1.82732269e-01f,
+1.43637747e-01f,+1.65562928e-01f,-1.43002123e-01f,+2.22443379e-02f,-1.51844084e-01f,
-1.44358888e-01f,-8.83838460e-02f,-1.06734410e-02f,-6.73530102e-02f,+1.72838822e-01f,
-1.70292750e-01f,-1.55205011e-01f,-6.83427751e-02f,-1.53406560e-01f,-9.68648046e-02f,
+9.37433988e-02f,-1.07081972e-01f,+1.82907894e-01f,-1.49085820e-01f,-1.53190315e-01f,
+1.17714874e-01f,+1.67836249e-01f,+9.46531743e-02f,-1.04023516e-01f,+1.52879998e-01f,
+1.04564473e-01f,-1.00298502e-01f,-1.19350620e-01f,-1.30548343e-01f,-2.55914181e-02f,
+7.38694593e-02f,-1.99008301e-01f,+6.10982515e-02f,+1.80512071e-01f,-1.74033213e-02f,
+6.86771348e-02f,-1.67935509e-02f,-1.30240887e-01f,-3.66905034e-02f,-2.20311284e-02f,
+1.92312822e-01f,-3.34579796e-02f,-1.58089474e-01f,-5.03543615e-02f,-2.93639284e-02f,
+7.78386518e-02f,+5.34399226e-02f,+4.85920310e-02f,-1.84641145e-02f,-1.57895789e-01f,
-4.43111882e-02f,-1.41936049e-01f,-9.93159562e-02f,+7.28596151e-02f,+4.97431904e-02f,
+9.38631445e-02f,+1.51205197e-01f,-7.69922733e-02f,-4.66700457e-02f,-1.03743322e-01f,
-3.65201123e-02f,-2.71154702e-01f,-3.53489727e-01f,-1.80541962e-01f,+1.33693889e-01f,
+3.15598398e-01f,-8.19803178e-02f,+2.52227277e-01f,-2.00705081e-01f,-1.13033250e-01f,
-8.27777684e-02f,+3.29293311e-01f,-3.24180603e-01f,+3.84998880e-02f,-3.18715051e-02f,
-1.53739661e-01f,+1.16001114e-01f,+1.14181139e-01f,-1.38892546e-01f,+1.71922266e-01f,
+1.55337766e-01f,+3.78974885e-01f,-1.87523942e-02f,-2.33957753e-01f,+7.54304901e-02f,
+1.36573017e-02f,-1.34702787e-01f,-2.04398975e-01f,-2.31051758e-01f,+1.62889156e-02f,
+2.23491699e-01f,-3.45211744e-01f,-2.19891518e-01f,+1.73894048e-01f,-2.93303519e-01f,
-2.83808243e-02f,+1.04791829e-02f,+1.95777699e-01f,+2.76736915e-02f,+2.88336188e-01f,
+3.82783473e-01f,+1.24077715e-01f,-1.93709850e-01f,+2.96985894e-01f,-1.82367489e-01f,
-1.39175385e-01f,+1.42335445e-01f,-8.19863826e-02f,+2.07908750e-01f,-1.43238127e-01f,
-3.36003035e-01f,+4.79480810e-02f,-2.84757167e-01f,-2.63189286e-01f,+2.91174144e-01f,
+2.40402371e-01f,+1.53656632e-01f,+8.70294049e-02f,+1.01934381e-01f,+4.11963761e-01f,
-1.91226408e-01f,-5.16687259e-02f,-2.54540145e-01f,+3.93350691e-01f,-1.01774685e-01f,
-1.69665977e-01f,-2.76802063e-01f,-3.25143427e-01f,+1.89444423e-02f,-2.35257506e-01f,
+5.81497401e-02f,-1.52794242e-01f,+1.12571120e-02f,-2.21628010e-01f,+1.29496887e-01f,
+1.64397452e-02f,-2.20314413e-02f,+2.20167249e-01f,-2.32754424e-01f,+5.30294655e-03f,
-3.77002321e-02f,+2.48102069e-01f,-6.12426363e-02f,+5.45167513e-02f,-1.08131185e-01f,
+1.07171379e-01f,+5.92634268e-02f,-8.44130442e-02f,+8.68163183e-02f,+3.58651802e-02f,
-1.94874257e-01f,+1.47999689e-01f,-2.59385835e-02f,+7.80922547e-02f,-1.99181698e-02f,
-5.74153587e-02f,-9.36051011e-02f,-1.29377306e-01f,-6.10673875e-02f,-2.00250708e-02f,
+4.90747672e-03f,+1.01621374e-01f,-1.38501778e-01f,+2.12573335e-01f,+1.05640829e-01f,
+1.43694982e-01f,-1.19815178e-01f,-7.10709468e-02f,-1.46703109e-01f,+2.02488318e-01f,
+9.99799073e-02f,+2.20011875e-01f,-9.00783539e-02f,+2.27241248e-01f,+2.91660689e-02f,
+9.92612690e-02f,-1.36802256e-01f,+1.02139056e-01f,-8.37787092e-02f,-1.66827440e-01f,
-2.21073017e-01f,-5.50982207e-02f,+7.24394992e-02f,+1.86908897e-02f,+7.28916824e-02f,
+2.24854693e-01f,+1.30364284e-01f,+9.81231779e-02f,-1.48619294e-01f,+1.78560421e-01f,
-3.47287022e-02f,-1.34208962e-01f,-4.02518734e-02f,+5.10345288e-02f,-8.90339836e-02f,
+9.83324423e-02f,-7.65735582e-02f,-2.15681031e-01f,+2.55278051e-02f,-2.02926192e-02f,
-1.76330179e-01f,-1.53949663e-01f,-1.30291566e-01f,-5.67636900e-02f,+2.10311398e-01f,
+2.01196760e-01f,-1.39736593e-01f,+2.15015024e-01f,+3.32828537e-02f,+8.11497346e-02f,
+9.67604443e-02f,-8.54643714e-03f,-1.65254712e-01f,+1.14059605e-01f,-1.08724579e-01f,
-3.55102913e-03f,-1.73113141e-02f,+1.46580189e-01f,-2.36933962e-01f,+2.98146874e-01f,
-5.21828197e-02f,+1.63978532e-01f,-1.55450672e-01f,-3.94876227e-02f,-4.95053641e-02f,
-3.19768228e-02f,+1.67668238e-01f,+1.70622021e-01f,-3.23640071e-02f,+3.21122967e-02f,
+2.92789847e-01f,+4.92108352e-02f,-1.44440800e-01f,+1.11961290e-01f,-2.10625693e-01f,
-1.83326051e-01f,+2.40128875e-01f,+1.01020299e-01f,+9.57081094e-03f,+1.61466934e-03f,
+1.50903016e-01f,+2.26309612e-01f,-1.36810586e-01f,+1.90384686e-01f,-1.18924990e-01f,
+5.43791056e-02f,+1.47295356e-01f,+8.38314742e-02f,-5.34722432e-02f,-4.18097526e-02f,
-6.34710044e-02f,+1.30535319e-01f,-2.03214362e-01f,-1.18436992e-01f,+1.83971375e-01f,
+1.86547324e-01f,+1.55327335e-01f,-4.90633920e-02f,-1.58541620e-01f,+3.16235185e-01f,
-1.38919577e-01f,-2.21716657e-01f,-2.29334146e-01f,+2.07019925e-01f,+1.68770507e-01f,
+9.56112891e-02f,-1.93712071e-01f,-2.73063928e-01f,+1.52851045e-02f,-2.24132985e-01f,
+5.25723584e-02f,+1.80486068e-01f,+1.81688115e-01f,+3.25785354e-02f,+9.78972465e-02f,
+1.75559685e-01f,-3.22350860e-03f,+1.93833515e-01f,+1.02052175e-01f,-1.39998510e-01f,
+1.15929522e-01f,-8.16443339e-02f,+4.57746126e-02f,+6.59186468e-02f,+8.09815750e-02f,
+9.64082628e-02f,+3.05990670e-02f,+2.69496083e-01f,-1.29593864e-01f,-1.58783004e-01f,
-1.69532642e-01f,+1.60377726e-01f,+8.32393914e-02f,+1.70209303e-01f,+1.14417925e-01f,
-3.84165607e-02f,+9.30939764e-02f,-8.03774148e-02f,+4.92475461e-03f,-8.21317583e-02f,
-8.10412094e-02f,+5.33804260e-02f,+1.27313539e-01f,+1.79627001e-01f,+2.10939988e-01f,
-5.10946028e-02f,+2.19551567e-02f,-3.96821760e-02f,-1.09623432e-01f,-4.54576612e-02f,
+1.51242882e-01f,-1.25686973e-01f,-2.94591133e-02f,-3.97435613e-02f,+1.63928062e-01f,
+1.65434688e-01f,-1.55514061e-01f,-1.07718319e-01f,+1.14390858e-01f,+1.59136117e-01f,
+1.38812944e-01f,-4.01524343e-02f,+7.15641975e-02f,-6.58831671e-02f,+1.74601287e-01f,
+8.83578211e-02f,+2.93862730e-01f,-1.22478448e-01f,+4.41936851e-02f,+1.76603034e-01f,
+1.32504180e-01f,+5.10493964e-02f,+1.84186593e-01f,+1.71167001e-01f,+1.20482422e-01f,
+1.16890475e-01f,-1.10114172e-01f,+4.47287895e-02f,-7.10566640e-02f,-3.43534984e-02f,
-4.72954474e-02f,-1.47522300e-01f,-2.38622889e-01f,-1.79352805e-01f,+2.57558107e-01f,
+1.61119625e-01f,-1.98318481e-01f,+2.32523784e-01f,-9.07189548e-02f,-1.73949018e-01f,
-2.33752318e-02f,-2.12659854e-02f,-1.55414152e-03f,+1.18973225e-01f,-1.85525954e-01f,
-1.80002913e-01f,+7.87335485e-02f,-5.16498275e-02f,-1.81834757e-01f,+2.28864953e-01f,
+9.16556828e-03f,+4.97912467e-02f,-2.71481276e-01f,-2.30047405e-01f,+7.45592192e-02f,
-9.96132419e-02f,-6.17349744e-02f,+3.96031775e-02f,-1.36193419e-02f,+2.67607093e-01f,
+1.27170756e-01f,-2.43623406e-01f,-2.41834268e-01f,-5.82039431e-02f,-2.90893197e-01f,
+1.10771200e-02f,+7.80622512e-02f,-2.68347394e-02f,-8.90113935e-02f,+1.22960642e-01f,
+2.69830495e-01f,-1.34316506e-02f,-2.25889400e-01f,+1.63782910e-01f,-5.88386953e-02f,
+8.83754566e-02f,-1.03451140e-01f,+1.74243897e-02f,+2.94051498e-01f,-4.24477868e-02f,
-1.51890844e-01f,+1.00648999e-02f,+7.41848126e-02f,+3.89751717e-02f,-8.49278048e-02f,
+2.21812814e-01f,+7.55186006e-02f,+1.07332673e-02f,-1.85047060e-01f,+8.09541494e-02f,
-7.87618160e-02f,-2.15706035e-01f,+2.31755078e-02f,+2.91463822e-01f,-1.06598906e-01f,
+1.76710889e-01f,-1.04105808e-01f,-7.51522183e-02f,+8.50376673e-03f,-2.00958535e-01f,
-1.39230162e-01f,+5.20480983e-02f,-1.80930242e-01f,-9.96783525e-02f,+2.36322284e-01f,
+3.79086100e-02f,-1.69650331e-01f,-9.13307741e-02f,+5.08898273e-02f,+1.45348683e-01f,
+2.17013821e-01f,+1.10041864e-01f,-5.94226783e-03f,+5.49010783e-02f,+1.29763499e-01f,
-1.57586280e-02f,+1.52264714e-01f,+1.47467121e-01f,+1.34824723e-01f,+1.07202388e-01f,
-9.04220119e-02f,+1.81117728e-01f,-4.02921364e-02f,-7.30624944e-02f,+2.12498263e-01f,
+5.81548959e-02f,+1.29136309e-01f,-1.54174075e-01f,-3.61866578e-02f,+1.21829592e-01f,
+1.93536907e-01f,-8.73580500e-02f,-5.83954714e-02f,-4.22076210e-02f,-7.55122602e-02f,
+1.95117921e-01f,-1.15276426e-01f,+1.32154033e-01f,+1.63521901e-01f,+1.29701793e-01f,
-1.29558712e-01f,+5.40881157e-02f,-1.15428455e-02f,-6.87542483e-02f,-5.64090908e-02f,
-1.79259479e-01f,+8.46147314e-02f,-3.10889576e-02f,+1.22569278e-01f,-1.24851614e-01f,
+1.26337498e-01f,+3.81893516e-02f,+2.97974143e-02f,+4.89787897e-04f,+3.37763838e-02f,
+9.36567504e-03f,+9.41332132e-02f,-4.03838307e-02f,+1.11071892e-01f,+2.52491236e-01f,
+1.49183437e-01f,+1.26757801e-01f,-1.19373407e-02f,+7.32851699e-02f,-4.44646701e-02f,
-4.34910059e-02f,+2.90275160e-02f,-2.67669763e-02f,-1.18382275e-01f,-1.22554943e-01f,
+1.14597768e-01f,+2.57117897e-01f,+1.58459306e-01f,+2.17176259e-01f,-1.01383943e-02f,
-9.07451361e-02f,-1.49637565e-01f,+3.26389149e-02f,+1.25463277e-01f,+3.01014981e-03f,
+2.94611640e-02f,+3.72376963e-02f,-3.86652746e-03f,+3.34729590e-02f,+2.64775664e-01f,
+2.27216542e-01f,+1.18615381e-01f,-7.79763684e-02f,+1.39081210e-01f,-7.88224712e-02f,
-9.44189653e-02f,-8.99244621e-02f,+2.41822049e-01f,+2.20337868e-01f,+8.18943456e-02f,
+1.71414152e-01f,-8.80596936e-02f,+1.21121250e-01f,-6.34045601e-02f,-4.78524417e-02f,
+6.57531098e-02f,+1.84018970e-01f,+5.81159405e-02f,-1.33708492e-01f,+1.11956038e-01f,
-1.95620041e-02f,+1.55250147e-01f,-1.38293669e-01f,+5.28430864e-02f,-5.09615839e-02f,
-1.40184090e-01f,-7.74263889e-02f,+6.62138984e-02f,-4.35180403e-03f,+1.66657925e-01f,
+2.35669240e-01f,-2.03148350e-02f,+1.05219387e-01f,+6.54662624e-02f,-3.62357534e-02f,
-3.51467840e-02f,-1.93532020e-01f,+5.38091036e-03f,+1.86409771e-01f,+7.72177726e-02f,
+5.14295436e-02f,+5.93404584e-02f,-7.55921230e-02f,+9.72359180e-02f,-7.48563334e-02f,
+7.44028091e-02f,+2.17186840e-04f,-3.65594476e-02f,-2.16650099e-01f,+3.98290381e-02f,
-3.44753042e-02f,-3.38978283e-02f,+9.77762714e-02f,+3.24961394e-02f,+1.74283832e-01f,
-6.24600016e-02f,+3.27585898e-02f,+1.03635810e-01f,+1.81413531e-01f,+1.14993840e-01f,
+4.84897196e-02f,-1.73926398e-01f,-1.25638358e-02f,+2.19466656e-01f,-1.05516456e-01f,
+6.52674288e-02f,+2.26032361e-02f,+3.91494744e-02f,+1.47295091e-02f,-8.83338749e-02f,
+7.88235813e-02f,-1.40860556e-02f,+1.21198595e-01f,+7.85603654e-03f,+1.21336870e-01f,
+1.26893103e-01f,+1.35583967e-01f,+4.52791788e-02f,+2.94561144e-02f,+1.64613247e-01f,
+7.07956851e-02f,-1.60742134e-01f,-1.31225809e-01f,+1.93612069e-01f,-6.63593858e-02f,
+6.17057495e-02f,-7.52566233e-02f,-1.18307941e-01f,+1.60523191e-01f,+3.28246392e-02f,
+3.42344269e-02f,-1.71285629e-01f,+8.20996612e-02f,+5.63984737e-02f,-9.25489962e-02f,
-5.83453365e-02f,+1.69059172e-01f,-9.32313278e-02f,-2.96851387e-03f,+1.16498329e-01f,
+2.06724316e-01f,+1.22446753e-01f,+4.21356186e-02f,-6.74661472e-02f,-9.36054736e-02f,
+4.08677459e-02f,+1.07916012e-01f,+5.71869966e-03f,+3.62035558e-02f,+1.20785117e-01f,
-8.60822648e-02f,-5.43353483e-02f,-9.65124667e-02f,+1.29420102e-01f,-5.67732677e-02f,
+1.21369690e-01f,+9.76872370e-02f,+1.74754232e-01f,+3.26993242e-02f,-7.76895508e-02f,
-5.83994724e-02f,-2.04440728e-02f,+7.27159381e-02f,-4.74820919e-02f,+1.61520138e-01f,
+2.64383167e-01f,-3.10480427e-02f,+9.06120688e-02f,+1.38456196e-01f,-2.37938672e-01f,
-1.03913829e-01f,-9.72388661e-04f,-7.17541426e-02f,-8.09007213e-02f,+8.19607824e-02f,
+2.26801649e-01f,+1.14149131e-01f,-2.79348101e-02f,-3.88923027e-02f,+2.09482208e-01f,
+2.39774019e-01f,+9.01669636e-02f,-3.28484252e-02f,+4.44813929e-02f,+1.77845415e-02f,
+1.74394205e-01f,+9.53105465e-02f,+1.01951905e-01f,+1.68511018e-01f,-9.45357084e-02f,
-1.32782817e-01f,-1.46640211e-01f,+1.35366380e-01f,+2.51561433e-01f,-2.60331541e-01f,
+6.38284236e-02f,+2.00076550e-01f,+8.48390386e-02f,-2.74145603e-01f,+8.70766193e-02f,
-3.17651369e-02f,-2.51865461e-02f,-1.34237576e-02f,-8.63590464e-02f,+1.13275439e-01f,
-7.95326531e-02f,+1.20691046e-01f,+1.22854607e-02f,-1.04257949e-01f,+5.67853749e-02f,
-9.18927193e-02f,-1.89283732e-02f,-1.90388560e-01f,+7.52393752e-02f,+9.86684710e-02f,
+7.67513961e-02f,+8.33101198e-02f,+1.78010672e-01f,+1.05461597e-01f,+1.87803302e-02f,
-7.67542496e-02f,+1.16410866e-01f,-1.91977814e-01f,+3.32045406e-02f,+1.32201903e-03f,
-1.39584988e-01f,+2.58311424e-02f,+8.74419957e-02f,-1.52292982e-01f,+2.50242472e-01f,
+4.30323035e-02f,-6.76100552e-02f,+2.57979959e-01f,-1.26723453e-01f,+2.52262354e-01f,
-1.39507744e-02f,-1.44677401e-01f,-1.71566769e-01f,+1.41555876e-01f,+9.23263803e-02f,
-1.47231102e-01f,-1.63418949e-01f,+3.67254503e-02f,+1.41284928e-01f,-1.03184082e-01f,
-1.57236643e-02f,+1.84560478e-01f,-8.41684639e-02f,+1.33696377e-01f,+1.62564412e-01f,
-5.26729748e-02f,-1.18420705e-01f,-6.00812025e-02f,+1.13463916e-01f,-7.69257843e-02f,
-7.94924423e-02f,-4.13467512e-02f,+1.58981591e-01f,+1.20536946e-01f,-1.09829418e-01f,
-5.05228788e-02f,-1.52462333e-01f,+1.20493062e-01f,-1.62544101e-01f,+6.95162266e-02f,
+1.35235162e-02f,+1.97441742e-01f,+1.43917903e-01f,-1.11428224e-01f,-6.15823641e-02f,
+1.61035687e-01f,+1.02000393e-01f,-9.55194086e-02f,-4.41119485e-02f,+1.98956370e-01f,
+1.47577912e-01f,+1.54266670e-01f,+4.12938073e-02f,-1.33478180e-01f,-1.37051687e-01f,
-1.68724909e-01f,+6.08566254e-02f,+1.87522128e-01f,-1.36112645e-02f,-2.31036954e-02f,
+1.56978006e-03f,+6.59991056e-02f,-1.01124138e-01f,-4.54666764e-02f,-2.34204270e-02f,
+3.75502219e-04f,-4.97878762e-03f,-1.77936316e-01f,+1.57505289e-01f,-1.74139619e-01f,
-1.05772935e-01f,-1.99205309e-01f,-2.89671011e-02f,-1.17542848e-01f,-2.77640931e-02f,
+4.67494167e-02f,+2.54916642e-02f,+4.57263365e-02f,+3.39390039e-02f,-8.25179070e-02f,
-8.82635340e-02f,-1.52196050e-01f,-1.21619932e-01f,+1.53227806e-01f,-8.76314193e-02f,
+5.38269384e-03f,+1.37262568e-01f,-4.26366329e-02f,-1.17241591e-01f,-1.56898622e-03f,
-6.64084107e-02f,+1.17976725e-01f,-1.12016484e-01f,-4.97265048e-02f,-6.08519018e-02f,
-4.14480157e-02f,+1.22110479e-01f,+3.00004538e-02f,+8.41649324e-02f,+9.94029716e-02f,
-1.90508626e-02f,+7.57312328e-02f,-9.65834633e-02f,-1.30625367e-01f,+1.29384205e-01f,
+1.84138730e-01f,-1.12464502e-01f,+6.40199929e-02f,+1.88780338e-01f,-1.58913180e-01f,
-1.35630682e-01f,+9.53572392e-02f,+1.85739361e-02f,+8.67178589e-02f,+1.54010624e-01f,
+2.97521967e-02f,+3.13982964e-02f,-9.16947797e-02f,-1.48924634e-01f,-1.30688623e-01f,
-2.40456662e-03f,-1.20240517e-01f,-1.48825303e-01f,+1.99545249e-01f,+4.85666022e-02f,
+1.55009866e-01f,+5.99414147e-02f,-2.20626239e-02f,+1.56306513e-02f,+5.41988388e-02f,
-1.80978123e-02f,-1.66986153e-01f,+1.39264509e-01f,+4.57425714e-02f,-1.37330011e-01f,
+1.15279704e-01f,+1.91357836e-01f,+1.91551432e-01f,-6.07905872e-02f,-1.64257571e-01f,
+1.20235056e-01f,-1.75774649e-01f,-6.75441846e-02f,+1.09470151e-01f,+1.98320160e-03f,
+4.70349193e-02f,+3.45878042e-02f,+6.39287308e-02f,+7.72436634e-02f,-9.34892297e-02f,
-1.80268794e-01f,-4.70405519e-02f,-2.86350936e-01f,-2.87598550e-01f,+2.13767201e-01f,
+1.46020418e-02f,+1.79413185e-01f,+2.25815520e-01f,-1.96082503e-01f,-2.75504738e-01f,
-6.89193979e-02f,+1.61945298e-01f,-6.76684454e-02f,+3.14946383e-01f,+5.43259270e-02f,
-2.18628064e-01f,+2.98644993e-02f,+5.55174686e-02f,-3.02181598e-02f,+2.00118586e-01f,
+2.22946070e-02f,+1.51100785e-01f,-1.16659120e-01f,-2.61391789e-01f,+2.67203748e-01f,
-2.18406379e-01f,-6.17712587e-02f,-1.05461977e-01f,+1.30751086e-02f,+2.73924053e-01f,
+2.51864940e-01f,+7.62742162e-02f,-2.08109364e-01f,+2.18532547e-01f,-2.10079163e-01f,
+3.23216133e-02f,+9.67873558e-02f,+2.56275028e-01f,+6.04809485e-02f,+1.97859481e-02f,
+3.02491307e-01f,+7.59769976e-02f,+8.19986165e-02f,+2.12230477e-02f,+5.41051105e-03f,
-3.09460491e-01f,-1.00775070e-01f,-1.06197938e-01f,+1.62434876e-01f,+1.48329496e-01f,
-1.02973245e-01f,+2.34698947e-03f,-2.02193499e-01f,-1.42810613e-01f,+1.18804522e-01f,
-4.30360138e-02f,+6.30558655e-02f,+1.96184814e-01f,-1.30032420e-01f,+3.46843183e-01f,
+8.11066702e-02f,+6.77058846e-02f,+6.21569753e-02f,+1.63330927e-01f,+9.69475135e-02f,
+5.64747490e-02f,-1.73127636e-01f,-2.58362412e-01f,+8.17406699e-02f,+1.04395999e-02f,
+1.22195855e-01f,-1.03753246e-01f,-1.55881345e-02f,-6.07884191e-02f,+1.55088753e-01f,
+1.27383739e-01f,+9.09331068e-02f,+2.07267955e-01f,-1.98050588e-01f,-1.23374790e-01f,
+1.56627044e-01f,+2.36789897e-01f,-1.00822318e-02f,+1.13114178e-01f,-1.90863952e-01f,
-1.08774535e-01f,-2.04087719e-01f,+2.54876286e-01f,-5.64117543e-02f,-9.54974443e-02f,
-1.52005121e-01f,-1.61090288e-02f,-2.20941510e-02f,-9.48313326e-02f,+5.21561764e-02f,
+3.37062515e-02f,-1.45187467e-01f,-1.06807955e-01f,+4.38874355e-03f,+1.86335910e-02f,
-1.71343703e-02f,-1.42990593e-02f,-9.92114376e-03f,-6.97200000e-02f,+1.13787822e-01f,
+2.13416503e-03f,+2.97124237e-02f,+2.37839058e-01f,-1.95826560e-01f,+2.09304497e-01f,
+1.60159111e-01f,+1.81320980e-01f,-1.25678688e-01f,+2.72593170e-01f,-7.39599019e-02f,
-3.71980965e-02f,-1.36023015e-01f,-9.49315913e-03f,+7.30276406e-02f,+1.77965477e-01f,
-2.36852337e-02f,-6.09117933e-02f,+8.41837749e-02f,-1.79514334e-01f,+1.46384090e-01f,
+1.77570865e-01f,+1.39485389e-01f,-6.64951131e-02f,-1.22233659e-01f,-4.10167351e-02f,
-1.64335296e-01f,-1.06749840e-01f,-1.19105140e-02f,-1.77098829e-02f,+9.56110433e-02f,
-8.68787989e-02f,-7.26035014e-02f,+5.28727211e-02f,+4.97968718e-02f,-1.17629610e-01f,
+1.74167052e-01f,-2.38859877e-02f,+2.03058243e-01f,+1.67855144e-01f,-1.60049543e-01f,
+1.32724613e-01f,-7.75947468e-03f,-5.42253368e-02f,-1.43755153e-01f,-5.25732525e-02f,
+6.49753213e-02f,-1.08277880e-01f,+1.03265285e-01f,-1.66419759e-01f,+1.81537539e-01f,
-1.49155825e-01f,+1.75218731e-01f,+4.72443774e-02f,-1.14664190e-01f,+7.99747854e-02f,
-9.75220576e-02f,+9.40733775e-02f,+6.47219270e-02f,+8.51638019e-02f,-7.70478025e-02f,
+1.97386876e-01f,-7.85635263e-02f,-1.39309421e-01f,+1.12382293e-01f,+1.25517353e-01f,
-1.52119875e-01f,-2.50024591e-02f,+3.66330855e-02f,+4.17229868e-02f,+1.40557364e-01f,
+5.65227009e-02f,+6.07065745e-02f,+9.92625728e-02f,+1.22371770e-01f,-8.93479139e-02f,
+2.52075382e-02f,+1.01688698e-01f,+1.76742598e-02f,-1.88957945e-01f,+1.78997833e-02f,
+1.48831196e-02f,-1.14298768e-01f,-2.10463181e-01f,-2.28400841e-01f,+1.38624430e-01f,
+3.28291878e-02f,-4.00642417e-02f,+1.55099362e-01f,-1.27146572e-01f,-1.09007902e-01f,
+1.07849933e-01f,+2.02446714e-01f,-7.91477114e-02f,-3.05056758e-02f,+2.06214413e-01f,
+3.07958778e-02f,+1.58014566e-01f,-4.48405109e-02f,-1.84704125e-01f,+1.96489200e-01f,
-5.67633659e-02f,+1.18082374e-01f,+2.76492331e-02f,+7.80223683e-02f,+1.30529508e-01f,
+1.18581235e-01f,-1.37918994e-01f,-1.73563898e-01f,+1.42242163e-01f,-6.77931905e-02f,
+2.90013365e-02f,-1.74098551e-01f,+1.99735090e-01f,-2.06497371e-01f,-3.92481647e-02f,
-1.40321990e-02f,-3.75412703e-02f,+1.36873499e-01f,-1.34562820e-01f,-1.09180786e-01f,
-7.88617432e-02f,-3.80971581e-02f,+1.47992387e-01f,-1.55359223e-01f,-1.02421813e-01f,
+1.26598224e-01f,+1.02559716e-01f,-1.12404823e-01f,+9.23372433e-02f,+1.15561597e-01f,
-7.51524046e-02f,+1.01832703e-01f,-1.69747084e-01f,+1.08753756e-01f,+1.27356201e-02f,
-1.48989394e-01f,-1.89411074e-01f,-1.26571149e-01f,+1.05141498e-01f,+1.72828034e-01f,
-1.78764015e-02f,-5.09370565e-02f,+1.55023168e-04f,+2.91338563e-02f,-6.97842762e-02f,
-4.22254317e-02f,-1.21927232e-01f,-5.33723980e-02f,+3.84500436e-02f,-1.64269164e-01f,
-1.01582997e-01f,+8.32728818e-02f,+2.67944336e-02f,+1.02143750e-01f,-1.10164329e-01f,
+3.44049372e-02f,+6.47315755e-02f,+1.42619550e-01f,-8.96035135e-02f,-4.70218062e-02f,
-1.24031439e-01f,+9.17419419e-02f,-5.05298786e-02f,+1.83268726e-01f,+1.97171003e-01f,
+1.69840589e-01f,-1.04952447e-01f,+5.09762950e-02f,-8.93084053e-03f,-1.36200815e-01f,
-3.03472579e-02f,-9.58394334e-02f,-1.70544162e-01f,-9.91191193e-02f,-5.54465503e-02f,
+2.34985482e-02f,-1.74123645e-01f,+4.71100211e-02f,-6.58974275e-02f,+1.73205696e-02f,
+8.88239592e-02f,+6.77979961e-02f,-1.34295449e-01f,+1.90150782e-01f,+3.57994996e-02f,
+2.01971069e-01f,-5.13348505e-02f,+1.16820499e-01f,-9.19366777e-02f,-5.51511832e-02f,
+2.40813699e-02f,+1.25062078e-01f,+1.73746571e-01f,-9.42790955e-02f,-8.37735459e-02f,
+1.86055139e-01f,-9.97019783e-02f,+1.88071594e-01f,-1.76257610e-01f,+3.91866826e-02f,
-2.83330679e-02f,+1.86243400e-01f,-1.69170931e-01f,-4.24839370e-03f,+4.48595881e-02f,
+1.50183504e-02f,-1.35140881e-01f,+1.70594186e-01f,+1.60622131e-02f,+1.93666428e-01f,
-6.61339909e-02f,+1.07595980e-01f,+1.10834628e-01f,+4.92176861e-02f,+1.77236907e-02f,
-9.84350443e-02f,+8.14385712e-02f,-4.03274558e-02f,+9.79904681e-02f,-1.08696759e-01f,
+1.11938827e-01f,+1.38269439e-01f,-2.06978172e-02f,+1.48286402e-01f,-3.71736027e-02f,
-1.30376697e-01f,-3.54993343e-02f,+9.29988641e-03f,+7.69189075e-02f,+1.72037259e-01f,
+4.42950334e-03f,+1.64355144e-01f,+5.29859364e-02f,-7.78249204e-02f,-7.93962851e-02f,
+7.90962577e-02f,-6.01202548e-02f,-1.96806286e-02f,+2.65874229e-02f,-3.02706063e-02f,
+2.20755488e-02f,+1.74441934e-02f,+1.89358041e-01f,-1.26100153e-01f,-1.93027547e-03f,
-2.04891525e-02f,-1.20056622e-01f,+1.40493780e-01f,+9.28984508e-02f,+2.60441452e-02f,
-1.76283568e-01f,-5.12116104e-02f,+2.33637914e-02f,+8.68640691e-02f,+8.88302550e-02f,
+1.69478789e-01f,+5.94321303e-02f,-1.41540736e-01f,-1.78307872e-02f,-2.03038856e-01f,
-2.43562972e-03f,-2.84486301e-02f,+1.84924081e-01f,-5.43426462e-02f,-1.66847050e-01f,
+1.30222932e-01f,-1.99505258e-02f,-1.50789917e-01f,-2.96345534e-04f,+2.05464542e-01f,
-1.34529367e-01f,+1.17353544e-01f,-3.13225389e-02f,+2.99168611e-03f,+1.62579551e-01f,
+1.04771107e-01f,+4.28260714e-02f,-5.74600585e-02f,-7.36688226e-02f,-1.63719133e-01f,
+1.55072972e-01f,+1.11230455e-01f,+1.00542523e-01f,-1.22208111e-01f,+2.26980522e-02f,
+1.80705026e-01f,+1.60253599e-01f,-1.81010455e-01f,+1.78452745e-01f,+8.13336149e-02f,
+3.96857150e-02f,-7.48677328e-02f,+1.82104781e-01f,+1.44823417e-01f,-1.17690362e-01f,
+1.20296665e-01f,-1.83842182e-01f,+2.11387984e-02f,-1.87987626e-01f,+1.75124496e-01f,
+6.22952990e-02f,-1.38404360e-02f,+1.83921382e-02f,+4.17340212e-02f,+1.06091686e-01f,
-8.45214054e-02f,+9.12570953e-02f,+1.67942852e-01f,+1.42454788e-01f,+1.51017874e-01f,
+1.81829482e-02f,+1.77786633e-01f,+9.28647220e-02f,+2.38062046e-03f,+8.11160728e-02f,
-6.13033436e-02f,-1.91049635e-01f,+5.06949192e-03f,+9.27567109e-02f,+5.50960861e-02f,
+2.09964186e-01f,-1.16911516e-01f,+1.47623524e-01f,+1.16915867e-01f,+6.32746443e-02f,
+1.42150506e-01f,+1.52500987e-01f,-6.72083870e-02f,+4.80376258e-02f,-1.25645757e-01f,
+6.80491421e-03f,-1.64712638e-01f,+2.15994895e-01f,-8.87566805e-02f,-1.52248133e-03f,
+1.30873799e-01f,+1.96681768e-01f,-7.86345080e-02f,-1.91924736e-01f,-1.78365987e-02f,
+1.61646560e-01f,-1.14017725e-01f,+6.41657114e-02f,-6.89200982e-02f,-1.54773146e-02f,
+1.99614748e-01f,-1.77589729e-02f,+6.96533695e-02f,+2.30625674e-01f,-1.58605203e-01f,
+4.33896258e-02f,-2.70752367e-02f,+1.06006064e-01f,-2.01127268e-02f,-1.43563384e-02f,
+1.81406736e-01f,+1.77848175e-01f,-3.07653416e-02f,+1.64660558e-01f,+1.78926975e-01f,
+1.78155705e-01f,-1.49876997e-01f,+1.49044082e-01f,+1.97308823e-01f,-6.82652276e-03f,
-9.84613448e-02f,+9.26113315e-03f,+1.23675056e-01f,+1.71819434e-01f,+1.07346013e-01f,
+1.97212607e-01f,+2.96422988e-01f,+1.31005555e-01f,-8.54320526e-02f,-3.84540968e-02f,
-3.42740305e-02f,+1.11127801e-01f,-7.56191984e-02f,+4.51543787e-03f,+1.33167610e-01f,
-1.88779116e-01f,-1.17897876e-01f,-9.00696814e-02f,-4.22686078e-02f,+5.58100753e-02f,
-3.02254371e-02f,+1.33742601e-01f,+1.35595381e-01f,-3.13081853e-02f,+1.47978827e-01f,
-7.74967372e-02f,-2.30779834e-02f,+1.34042934e-01f,-3.73045430e-02f,-1.25783607e-01f,
+6.50432408e-02f,+7.24300519e-02f,+7.80659616e-02f,+1.98374293e-03f,-1.12402357e-01f,
+7.39559084e-02f,-3.36093688e-03f,+1.77364171e-01f,-8.74388516e-02f,-5.04736602e-02f,
+5.25886677e-02f,+1.00830905e-01f,+1.43845141e-01f,+6.76472485e-02f,-5.20560928e-02f,
+1.59059651e-02f,-7.26312399e-02f,-1.09519266e-01f,+1.12741672e-01f,-9.06158180e-04f,
+1.80970907e-01f,+1.01625295e-02f,+9.60554183e-02f,+1.93111245e-02f,+1.53122708e-01f,
-8.84285793e-02f,-1.87988188e-02f,-6.47097230e-02f,-3.12424023e-02f,-4.55260612e-02f,
+7.13542253e-02f,-6.78532422e-02f,-1.62479922e-01f,-3.90400104e-02f,-2.00731568e-02f,
+9.82646719e-02f,+1.52140200e-01f,-1.86507136e-01f,-1.58503190e-01f,-1.40098587e-01f,
-3.96351404e-02f,-1.30245060e-01f,-7.27082565e-02f,+9.77605879e-02f,+1.89407602e-01f,
+2.26214737e-01f,+1.71515152e-01f,-1.58230871e-01f,+4.94871587e-02f,+2.51862556e-01f,
-2.12672442e-01f,-7.38241302e-04f,-1.19776860e-01f,-1.60867572e-02f,+1.38575271e-01f,
-5.85806146e-02f,+1.59509003e-01f,+6.91654757e-02f,-1.07711111e-03f,-2.04959974e-01f,
-1.53858930e-01f,-1.20135061e-01f,+6.06996305e-02f,-4.43372391e-02f,+1.94128215e-01f,
+2.57250279e-01f,+1.30664736e-01f,-1.32464478e-02f,-1.63705677e-01f,-1.73486426e-01f,
+1.04668476e-01f,+9.13019329e-02f,+1.05755970e-01f,-4.62495908e-02f,+1.16363317e-01f,
+5.80637492e-02f,-1.61000878e-01f,+2.47368306e-01f,+3.03167850e-02f,+2.43953601e-01f,
+1.70248762e-01f,+1.27644703e-01f,+4.05266677e-04f,+1.02987699e-01f,+1.37031317e-01f,
-1.50833786e-01f,+9.20743793e-02f,-2.10757479e-01f,-1.15180455e-01f,+3.72414812e-02f,
+9.28526074e-02f,+1.55919284e-01f,+8.41597840e-02f,+1.94015115e-01f,-6.17114194e-02f,
+1.02787457e-01f,+1.07686944e-01f,+2.48835180e-02f,+4.34304401e-02f,+9.72350389e-02f,
+1.05322592e-01f,-2.33461354e-02f,-2.00550526e-01f,+1.12406261e-01f,+1.94384586e-02f,
-2.48977076e-02f,-1.92199320e-01f,+2.61288695e-02f,+2.61522382e-02f,-1.45792067e-01f,
-1.44075334e-01f,+1.67125743e-02f,-2.00615704e-01f,+8.96779746e-02f,+1.18648127e-01f,
+2.40161598e-01f,+1.30570412e-01f,+1.31445155e-01f,+1.71328038e-02f,-1.02991626e-01f,
-2.01589867e-01f,-4.39041369e-02f,+3.17454338e-02f,+2.87075993e-02f,+4.65929732e-02f,
+6.57921508e-02f,-3.24937664e-02f,-9.66330469e-02f,-1.70159921e-01f,-1.76470280e-02f,
-7.11621940e-02f,-2.87729949e-02f,-6.74027205e-03f,+1.32009521e-01f,-9.53936353e-02f,
+1.12774447e-01f,-2.15955377e-02f,+1.80066690e-01f,-3.90267372e-02f,-4.80892956e-02f,
-1.51345074e-01f,+1.37709334e-01f,-8.94442573e-02f,+6.39830381e-02f,-1.86174661e-02f,
+9.45243388e-02f,-1.32035881e-01f,-1.92819983e-02f,+1.15684792e-01f,-1.75643146e-01f,
+1.49319157e-01f,-8.39264691e-03f,+9.31134671e-02f,+2.97797620e-02f,-1.68072209e-01f,
+1.00984171e-01f,+1.29837170e-01f,+1.35788843e-01f,+1.48833379e-01f,-1.84352323e-01f,
-1.41435668e-01f,-3.08918357e-02f,-3.73807997e-02f,-1.04385220e-01f,-1.46869019e-01f,
-1.50238499e-01f,+8.80533308e-02f,+6.92235082e-02f,-7.65667856e-03f,+5.18683344e-02f,
-7.70561323e-02f,+1.14966288e-01f,-1.34961098e-01f,-1.79301098e-01f,+1.32802472e-01f,
+1.74385503e-01f,+2.63110250e-02f,+2.24920362e-02f,+1.47257462e-01f,+8.30320567e-02f,
+2.24192291e-02f,-1.29775047e-01f,+1.75919011e-01f,+1.21728763e-01f,+1.14088938e-01f,
-1.94982514e-01f,-1.38305590e-01f,-1.21644661e-01f,-8.22051987e-02f,+1.11982509e-01f,
+2.25789100e-02f,-9.50532481e-02f,+1.14601031e-01f,+6.33898526e-02f,+9.24111754e-02f,
-8.58405232e-03f,+1.28565595e-01f,+7.12645203e-02f,+1.27478540e-02f,-1.97093636e-01f,
+3.88418846e-02f,-1.31538033e-01f,+1.38667196e-01f,-6.70005083e-02f,+1.85395092e-01f,
-6.36585355e-02f,+1.50674641e-01f,-1.04398929e-01f,+7.52490684e-02f,-8.69575888e-02f,
-6.67570159e-02f,+1.78861901e-01f,+2.38190480e-02f,-1.59849171e-02f,-6.13602288e-02f,
-7.12091550e-02f,-1.21505857e-02f,-1.92229599e-01f,+1.93418947e-03f,-9.99459624e-02f,
-1.37113541e-01f,+1.17261231e-01f,-4.92230691e-02f,-1.20174073e-01f,-1.82558998e-01f,
+1.38594091e-01f,-5.40300161e-02f,+1.42583549e-02f,-1.61605235e-02f,-3.88613790e-02f,
+8.30395147e-02f,+1.87593207e-01f,-1.65234823e-02f,+1.61650702e-01f,+1.12626124e-02f,
-6.86092526e-02f,-1.73644677e-01f,-1.03392310e-01f,-1.43246919e-01f,-6.50478676e-02f,
-4.65485230e-02f,+1.78299516e-01f,+5.45464940e-02f,-1.17567033e-01f,-1.93947226e-01f,
+2.03453703e-03f,-1.60563439e-02f,+1.11340806e-01f,-1.41697079e-01f,+1.50544867e-01f,
+3.42593230e-02f,-8.48361105e-02f,+1.80537403e-01f,-4.94178757e-02f,-3.76998484e-02f,
+5.65302148e-02f,-1.88311413e-02f,-4.98954430e-02f,-1.63863137e-01f,-1.11435123e-01f,
+5.42171150e-02f,-1.82239085e-01f,+1.14961110e-01f,-3.25130671e-02f,+2.37266757e-02f,
-7.89771378e-02f,+1.75227672e-01f,-1.08463064e-01f,-1.86104715e-01f,+1.65130049e-01f,
+3.59279811e-02f,-1.27937654e-02f,+8.80624447e-03f,-7.10138902e-02f,+9.65859443e-02f,
+2.00198680e-01f,-2.76780520e-02f,-1.09328642e-01f,-4.13447879e-02f,+1.65114686e-01f,
+1.81627776e-02f,+1.86782211e-01f,-9.15442631e-02f,-3.22554028e-03f,+2.03703041e-03f,
-1.26485914e-01f,-1.40788466e-01f,+2.40455508e-01f,+1.43048376e-01f,-1.51369438e-01f,
-1.84203073e-01f,+1.93743724e-02f,-1.12351209e-01f,+1.93194807e-01f,+3.14734392e-02f,
-1.18808718e-02f,-1.16180055e-01f,+1.83328807e-01f,+2.13935614e-01f,+4.42457087e-02f,
-6.79247007e-02f,+1.34190723e-01f,+1.66997269e-01f,+1.12064481e-01f,+1.47657916e-01f,
+1.38418823e-01f,-6.01581438e-03f,-1.22944757e-01f,+5.70078157e-02f,+1.60006315e-01f,
+3.18648666e-02f,+2.35229284e-02f,+1.71706945e-01f,+1.43653005e-02f,-1.03295311e-01f,
-1.03290357e-01f,+3.66832390e-02f,+1.59604661e-02f,+3.16651464e-02f,-4.28047217e-02f,
-9.27226385e-04f,+1.34363949e-01f,-1.31992131e-01f,+1.40805706e-01f,-3.59518104e-03f,
-1.84610426e-01f,-8.52740929e-02f,-6.61319271e-02f,+1.68674588e-01f,-1.50458276e-01f,
-1.49505705e-01f,+8.36211145e-02f,+1.33871334e-02f,-1.64446324e-01f,+2.85512209e-01f,
+2.50329431e-02f,-9.15691853e-02f,+2.11407363e-01f,+1.54551178e-01f,+1.81499757e-02f,
+3.88354398e-02f,-2.52846833e-02f,+1.55532673e-01f,-6.06971346e-02f,+1.51634723e-01f,
+1.07871160e-01f,-8.26991200e-02f,-1.62091076e-01f,+1.49847448e-01f,+2.47212276e-01f,
+1.22147679e-01f,+1.50933921e-01f,+1.64214343e-01f,+3.08788139e-02f,-7.69685060e-02f,
-3.68951587e-04f,+1.17153913e-01f,-1.11260349e-02f,+6.48609772e-02f,+4.78387922e-02f,
-4.72574458e-02f,+1.00567870e-01f,+5.79779707e-02f,+1.59973815e-01f,+1.34023950e-01f,
-2.30201818e-02f,-2.79910415e-02f,-1.43452346e-01f,-5.12910746e-02f,+9.65448841e-02f,
-3.94991599e-02f,-8.42952281e-02f,+1.07871458e-01f,-1.92975670e-01f,+2.39447996e-01f,
+1.21550458e-02f,-2.65574250e-02f,+4.55880649e-02f,-2.01266274e-01f,-9.72402766e-02f,
+1.83216855e-02f,-1.51665002e-01f,+6.99452087e-02f,+8.41811746e-02f,+5.64582227e-03f,
-8.09368640e-02f,+1.43240646e-01f,-1.10409178e-01f,-1.99068025e-01f,+1.67062134e-01f,
+1.22523971e-01f,+6.90488443e-02f,-3.23877931e-02f,-1.16390377e-01f,-1.70108795e-01f,
+8.98928866e-02f,+4.61742617e-02f,+1.63161412e-01f,-1.36507988e-01f,-1.57845125e-01f,
+1.18547035e-02f,+3.53867635e-02f,+1.14638545e-01f,-3.55153829e-02f,+2.19409078e-01f,
-6.88589662e-02f,+1.91206530e-01f,+2.00649410e-01f,-1.74129128e-01f,+2.11678877e-01f,
+1.06460661e-01f,-2.33850434e-01f,-1.75872535e-01f,-1.88388731e-02f,+2.46640876e-01f,
+1.67783022e-01f,-9.22913924e-02f,+1.94651932e-01f,+5.47466539e-02f,+4.74549159e-02f,
+1.15663677e-01f,+2.61407584e-01f,+8.60541314e-02f,+2.58894991e-02f,+9.05612856e-02f,
-4.63345125e-02f,-1.75558105e-01f,+1.36294231e-01f,-2.78621495e-01f,+1.81287691e-01f,
-7.73380697e-02f,+2.96073675e-01f,-1.11449072e-02f,-1.61300436e-01f,+1.90392621e-02f,
+9.25310701e-02f,-4.89908755e-02f,-1.27554908e-02f,-6.61165938e-02f,+2.64446378e-01f,
+2.50827283e-01f,-1.01878859e-01f,-1.97235852e-01f,+7.05048665e-02f,-2.46322200e-01f,
-2.46574759e-01f,-2.75695711e-01f,+6.03658631e-02f,+1.04609981e-01f,+2.07439922e-02f,
+2.53076255e-01f,+2.38292709e-01f,-1.25951365e-01f,+1.48618057e-01f,+7.85709843e-02f,
-5.90820163e-02f,+9.94922221e-03f,-8.04576352e-02f,+1.14856981e-01f,-1.00614853e-01f,
-1.36097083e-02f,-1.40697107e-01f,-2.35616326e-01f,+1.11554943e-01f,-3.64006348e-02f,
+3.01311135e-01f,+1.07865535e-01f,+1.53409466e-01f,-1.55583858e-01f,+8.93213674e-02f,
-9.58327800e-02f,-1.49611428e-01f,-2.21806124e-01f,+8.12406763e-02f,+1.58247426e-01f,
+1.10869966e-01f,-2.24949792e-01f,-1.23742133e-01f,+5.80832809e-02f,-2.22562164e-01f,
+7.97146466e-03f,-4.86642085e-02f,+9.52818841e-02f,-1.84840232e-01f,+4.17192876e-01f,
+2.58110255e-01f,-2.07869615e-02f,+1.32238030e-01f,+5.13952800e-05f,-1.69025153e-01f,
-1.73189566e-01f,+2.98669398e-01f,-2.90530056e-01f,+3.56460869e-01f,-2.35903766e-02f,
-2.75153909e-02f,+8.40097964e-02f,+2.95152515e-01f,-6.83692917e-02f,+3.72252107e-01f,
-8.45417902e-02f,+2.19211783e-02f,-7.21333697e-02f,-7.35656824e-03f,+3.42583805e-01f,
-9.14079845e-02f,+1.84953555e-01f,+2.65963562e-02f,+3.00720762e-02f,+3.72513771e-01f,
+3.09998870e-01f,+1.30495839e-02f,-2.44063273e-01f,+3.47186953e-01f,-4.78496291e-02f,
+2.50228141e-02f,-1.97874561e-01f,+1.34584218e-01f,+1.01257667e-01f,+2.12783650e-01f,
+2.83514529e-01f,+2.58763641e-01f,-1.39316857e-01f,+9.13464278e-02f,-1.80940941e-01f,
-3.38985741e-01f,+1.33851022e-01f,+3.29582728e-02f,+1.08273119e-01f,+5.70147783e-02f,
-9.05802771e-02f,-1.13661513e-02f,-4.66996580e-02f,-3.05365950e-01f,+3.41963768e-02f,
+8.88177156e-02f,+5.15880063e-02f,+1.03901312e-01f,-2.32595503e-02f,+2.01132953e-01f,
+1.70614481e-01f,-3.22272003e-01f,-1.44484684e-01f,+2.38951921e-01f,+1.40428737e-01f,
-1.45638824e-01f,-2.66467065e-01f,-2.02104911e-01f,+4.59690429e-02f,-2.23010555e-01f,
-3.27953286e-02f,+1.73030775e-02f,+3.11172567e-02f,-5.49729243e-02f,+2.82376915e-01f,
+1.95679456e-01f,+2.43035182e-02f,+1.45249709e-01f,+1.19497655e-02f,+5.60881160e-02f,
-1.05173476e-01f,-6.90056384e-02f,-2.19032258e-01f,+5.37359565e-02f,-1.78768799e-01f,
-2.53627561e-02f,+3.12740952e-02f,+2.61137784e-01f,+1.11464627e-01f,+3.97106409e-02f,
-6.94542229e-02f,+6.21883832e-02f,-2.28109106e-01f,+1.82001367e-02f,-5.37486002e-02f,
-1.72503367e-01f,+1.13851085e-01f,+1.06723560e-02f,+8.89692903e-02f,+2.86483198e-01f,
+1.83423206e-01f,-1.90922529e-01f,-1.92689061e-01f,-2.92609893e-02f,+3.78605463e-02f,
-1.36117309e-01f,-1.73664331e-01f,+2.26136029e-01f,-8.89408514e-02f,+8.94914791e-02f,
+1.80925667e-01f,+3.18907022e-01f,+3.68901268e-02f,+6.40512481e-02f,+2.45132223e-02f,
-1.35223553e-01f,-7.13387551e-03f,+1.58898979e-02f,+1.70448005e-01f,+3.41063514e-02f,
-7.42209703e-02f,+5.08511290e-02f,-2.60722995e-01f,-5.15884021e-03f,+1.51872039e-02f,
+7.62404576e-02f,+1.56364113e-01f,+1.88066944e-01f,+6.96178377e-02f,+2.91251898e-01f,
-7.11745545e-02f,-2.44007349e-01f,-2.48482600e-01f,+1.64291531e-01f,+2.57743984e-01f,
-1.77512273e-01f,-7.56001174e-02f,+9.20229331e-02f,+1.87073827e-01f,-1.51664302e-01f,
-1.18968223e-04f,+2.08194941e-01f,-1.51404575e-01f,-3.28879990e-02f,+4.92020436e-02f,
+8.56267195e-03f,-4.72136028e-02f,+1.34493709e-01f,+1.48233458e-01f,+7.85538554e-02f,
-6.16110079e-02f,+1.72173873e-01f,-2.98892967e-02f,+1.49192974e-01f,+8.69470239e-02f,
+3.51317786e-02f,-2.00936660e-01f,+1.36589199e-01f,+1.76922217e-01f,+1.56689525e-01f,
-1.07339427e-01f,+1.79293409e-01f,-6.89659417e-02f,-1.07206166e-01f,+5.83899990e-02f,
+2.20771898e-02f,-1.55701071e-01f,+1.74661696e-01f,+5.45914434e-02f,+5.66720171e-03f,
-6.51307777e-02f,+8.08655769e-02f,-9.12271664e-02f,-1.37635440e-01f,+1.78540081e-01f,
+3.15099675e-03f,+9.92494896e-02f,-1.74773261e-01f,-1.41331017e-01f,-1.00661516e-02f,
-2.11388692e-01f,-1.79448351e-01f,-4.42146175e-02f,-8.27717409e-02f,-3.43654561e-03f,
+1.45701036e-01f,-4.03804667e-02f,+1.35528371e-01f,+2.77910698e-02f,+9.22615305e-02f,
+1.44934684e-01f,-5.59813492e-02f,-1.35441110e-01f,+9.04803798e-02f,-1.95280775e-01f,
-9.66536701e-02f,-1.41373202e-01f,-1.14904001e-01f,-1.12751737e-01f,-9.18107405e-02f,
+1.39016271e-01f,-1.71142921e-01f,+5.24066612e-02f,+1.47363305e-01f,-3.38985473e-02f,
+7.79315531e-02f,+2.40981020e-02f,+1.13807797e-01f,-8.19463655e-02f,-1.26634061e-01f,
}; 
//k2c_tensor dense_30_kernel = {&dense_30_kernel_array[0],2,5880,{84,70, 1, 1, 1}};
dense_30_kernel.ndim=2;
dense_30_kernel.numel=5880;
dense_30_kernel.shape[0]=84;
dense_30_kernel.shape[1]=70;
dense_30_kernel.shape[2]=1;
dense_30_kernel.shape[3]=1;
dense_30_kernel.shape[4]=1;
for(i=0;i<5880;i++){
#pragma HLS unroll factor = 128
//#pragma HLS unroll
	dense_30_kernel.array[i] = dense_30_kernel_array[i];
}


float dense_30_bias_array[70] = {
-5.41666115e-04f,-2.43807342e-02f,-8.98293103e-04f,+1.70839019e-02f,+5.97166456e-02f,
+5.29247411e-02f,-1.75855905e-02f,+5.15716486e-02f,+1.07900538e-02f,+1.42593915e-02f,
+2.70019639e-02f,+5.52767441e-02f,+1.76338684e-02f,+3.50919254e-02f,-1.49228657e-02f,
-7.77025009e-03f,-1.63561236e-02f,+7.33033419e-02f,+2.88407616e-02f,+2.03388166e-02f,
-1.61647312e-02f,+3.88861634e-02f,-1.17783705e-02f,+1.74301732e-02f,+3.29862647e-02f,
+6.12054486e-03f,+0.00000000e+00f,-1.43343722e-02f,+1.66892391e-02f,+4.20301557e-02f,
+3.53102759e-02f,+1.68538243e-02f,+1.18735190e-02f,+2.87521034e-02f,+3.37675889e-03f,
+1.78535692e-02f,+1.95901631e-03f,-8.07290152e-03f,-1.48276370e-02f,+6.47203550e-02f,
+1.52812423e-02f,+6.15103431e-02f,+1.02641266e-02f,+2.29633544e-02f,-8.62887688e-03f,
-2.33895965e-02f,-5.34351729e-03f,+5.02841966e-03f,+6.29236875e-03f,-5.77822700e-03f,
+1.95945390e-02f,-8.34443606e-03f,-1.25187524e-02f,+3.00723850e-03f,+1.37789128e-02f,
+4.40021157e-02f,+6.12556934e-02f,-4.43425914e-03f,-2.03958414e-02f,+5.79448529e-02f,
-1.79832280e-02f,-2.46542003e-02f,+3.65882623e-03f,+5.89084327e-02f,+6.48356751e-02f,
-1.18914852e-02f,+3.33551727e-02f,+1.06893694e-02f,+3.30237043e-03f,+9.95913520e-03f,
}; 
//k2c_tensor dense_30_bias = {&dense_30_bias_array[0],1,70,{70, 1, 1, 1, 1}};
dense_30_bias.ndim=1;
dense_30_bias.numel=70;
dense_30_bias.shape[0]=1;
dense_30_bias.shape[1]=1;
dense_30_bias.shape[2]=1;
dense_30_bias.shape[3]=1;
dense_30_bias.shape[4]=1;
for(i=0;i<70;i++){
//#pragma HLS unroll factor = 16
#pragma HLS unroll
	dense_30_bias.array[i] = dense_30_bias_array[i];
}


float dense_30_fwork[5964] = {0}; 

 
float dense_31_output_array[10] = {0}; 
//k2c_tensor dense_31_output = {&dense_31_output_array[0],1,10,{10, 1, 1, 1, 1}};
dense_31_output.ndim=1;
dense_31_output.numel=10;
dense_31_output.shape[0]=10;
dense_31_output.shape[1]=1;
dense_31_output.shape[2]=1;
dense_31_output.shape[3]=1;
dense_31_output.shape[4]=1;
for(i=0;i<10;i++){
#pragma HLS unroll factor = 16
	dense_31_output.array[i] = dense_31_output_array[i];
}


float dense_31_kernel_array[700] = {
-4.10778016e-01f,+2.59099990e-01f,+7.69857317e-02f,-4.48606350e-02f,-8.33014101e-02f,
-2.61941344e-01f,-2.75765479e-01f,-2.18294933e-01f,+2.57688642e-01f,+2.09471285e-01f,
-2.17708841e-01f,-2.64325976e-01f,+7.18386173e-02f,+3.16559166e-01f,-3.17948386e-02f,
-1.32854104e-01f,-3.55083048e-01f,-2.08874550e-02f,+1.40598983e-01f,-1.42028674e-01f,
+1.58270180e-01f,+9.74793732e-02f,+2.91680425e-01f,-1.68164536e-01f,+1.42517000e-01f,
+9.86745059e-02f,-1.02682792e-01f,-1.33578017e-01f,-1.01174146e-01f,+1.91720888e-01f,
-1.38451979e-01f,+8.56119320e-02f,+2.13295311e-01f,+3.11295569e-01f,+1.10878780e-01f,
-1.70778483e-01f,+5.38694896e-02f,-3.20491083e-02f,+2.27799281e-01f,-2.03278512e-01f,
+2.84503341e-01f,+7.36918077e-02f,-3.07578109e-02f,-1.72766708e-02f,-4.47773784e-02f,
-2.76771903e-01f,+1.02474608e-01f,+1.89244315e-01f,+2.01632783e-01f,-1.05424546e-01f,
+2.01795161e-01f,-2.53474414e-01f,-2.94224303e-02f,-8.17356855e-02f,-8.98978710e-02f,
-6.24667034e-02f,+3.34128648e-01f,-2.35491261e-01f,+1.02264076e-01f,-2.30184615e-01f,
-2.15789437e-01f,+2.05204248e-01f,-2.70300470e-02f,+1.80034459e-01f,+7.84561187e-02f,
+2.18497723e-01f,-8.23547691e-02f,+1.84097201e-01f,-1.38575435e-01f,-1.14093229e-01f,
-5.42535670e-02f,+1.50418943e-02f,-2.44749546e-01f,+1.26734272e-01f,+1.28020197e-01f,
-9.07284149e-04f,+2.74267942e-01f,+8.52620229e-02f,-1.43246666e-01f,-2.49938205e-01f,
+1.80123612e-01f,+1.46505296e-01f,+2.49072790e-01f,+1.41577527e-01f,+1.48327976e-01f,
-6.46109059e-02f,-2.28693634e-01f,+1.50593426e-02f,+5.60686104e-02f,-1.44837171e-01f,
-3.50937068e-01f,+7.26681799e-02f,+2.16220170e-01f,+1.25938937e-01f,+4.23851348e-02f,
+2.43578374e-01f,-1.30138174e-01f,-1.79374397e-01f,+2.56331682e-01f,-2.01962486e-01f,
+2.39812151e-01f,-2.20075458e-01f,+2.73000062e-01f,+2.95724243e-01f,-3.84581052e-02f,
+2.38423616e-01f,-2.31690487e-04f,+1.68742910e-01f,+2.41932556e-01f,+7.35401511e-02f,
+2.97700703e-01f,-5.17478883e-02f,-3.42587307e-02f,+3.98195721e-02f,-2.89756775e-01f,
-3.69480252e-03f,+8.02877173e-02f,+1.34795859e-01f,+1.53348282e-01f,-1.63642153e-01f,
-5.44761010e-02f,-2.79843479e-01f,+3.09924662e-01f,-6.80476576e-02f,-5.76102510e-02f,
+7.43119940e-02f,-1.17250979e-01f,+1.34691954e-01f,+1.10039443e-01f,-3.03200036e-02f,
-1.55688733e-01f,-9.50053185e-02f,-2.84380347e-01f,+2.06737593e-01f,-2.79021829e-01f,
-2.74484083e-02f,+2.13748083e-01f,-8.56251940e-02f,-1.73703685e-01f,+2.51666456e-01f,
-2.74519414e-01f,+2.20609978e-01f,+2.09125876e-02f,+1.16203941e-01f,+3.58977497e-01f,
+1.97262168e-02f,-2.28978336e-01f,-8.27208310e-02f,+2.36429751e-01f,-1.41614184e-01f,
-4.00156200e-01f,-9.29570347e-02f,+1.29174083e-01f,+1.75222695e-01f,+3.15732419e-01f,
-8.40212032e-02f,-3.42738450e-01f,+6.77333847e-02f,-8.93433914e-02f,+1.12492524e-01f,
-2.86673546e-01f,+1.30854234e-01f,-1.89664006e-01f,+1.04282543e-01f,+1.62372440e-01f,
-3.59403342e-02f,+9.08494592e-02f,-1.07068010e-01f,+5.58294170e-02f,-5.60233509e-03f,
+2.04771087e-01f,-4.43760213e-03f,+2.62457550e-01f,+1.65863156e-01f,+1.91346481e-01f,
-2.40209058e-01f,+2.89712906e-01f,-1.92179129e-01f,-1.15753524e-02f,-4.82251868e-02f,
+9.42120235e-03f,-2.61602134e-01f,+2.22153008e-01f,+2.64897756e-02f,+5.72199933e-02f,
-2.08879784e-01f,-1.25582842e-02f,+2.63924599e-01f,+1.57704264e-01f,-8.03535525e-03f,
+4.09089029e-01f,+2.49660015e-01f,-7.71180838e-02f,-1.25196621e-01f,-3.61998491e-02f,
-2.69413501e-01f,+2.04889968e-01f,-1.88555177e-02f,-3.51200193e-01f,+1.03864342e-01f,
-1.29493345e-02f,-2.27402765e-02f,-1.10607617e-01f,+8.83363932e-02f,+2.72775352e-01f,
+1.28871605e-01f,-2.09623471e-01f,+1.16179623e-01f,-2.60986146e-02f,+7.82290027e-02f,
-1.28066063e-01f,+1.07805550e-01f,-3.31346467e-02f,-2.35584646e-01f,+1.81893781e-01f,
-2.54842043e-01f,+2.76041955e-01f,+2.16778845e-01f,+9.16900188e-02f,-2.20932022e-01f,
-1.81487635e-01f,-1.00227550e-01f,-5.06899096e-02f,-3.92131461e-03f,+1.33355901e-01f,
+2.24717483e-02f,-2.32807070e-01f,-2.67420143e-01f,+3.86399239e-01f,+1.53761506e-01f,
-2.60165721e-01f,+1.02831908e-01f,+1.88879728e-01f,+6.23866543e-03f,+4.58494294e-03f,
-1.37907133e-01f,-2.00981900e-01f,+1.25937954e-01f,+3.58512551e-01f,-2.57846028e-01f,
+2.11400509e-01f,-1.07668683e-01f,-2.27147445e-01f,-5.81616834e-02f,-2.87607342e-01f,
-5.30709922e-02f,+1.64289474e-01f,-2.67380923e-01f,+1.51964784e-01f,+6.48800507e-02f,
-2.47391045e-01f,-2.41668925e-01f,+3.09696853e-01f,-1.30485043e-01f,+2.46064365e-02f,
-2.06349909e-01f,-1.23392165e-01f,-1.45744562e-01f,+3.08558136e-01f,+2.51381755e-01f,
+6.82022572e-02f,+2.51133204e-01f,+7.00455904e-03f,+1.15100384e-01f,-1.78797573e-01f,
+1.07750893e-02f,-1.36269748e-01f,+8.89143348e-03f,+1.56306326e-02f,-2.29483753e-01f,
-2.70991892e-01f,+4.40604687e-02f,+2.53697693e-01f,+1.08211115e-01f,-1.80943251e-01f,
+1.61103398e-01f,-2.32305229e-01f,-1.91166520e-01f,-1.24101825e-01f,+2.19280988e-01f,
-6.83589429e-02f,-2.23995641e-01f,+1.87131405e-01f,+1.22793525e-01f,+1.49787664e-02f,
-1.93554968e-01f,-1.61599144e-01f,+1.87871233e-01f,+3.66101235e-01f,+2.32177246e-02f,
+5.08060634e-01f,+2.53342628e-01f,-3.11935455e-01f,-2.28081658e-01f,-3.53374600e-01f,
-2.07993343e-01f,+2.91762352e-01f,+1.06296569e-01f,-4.13630426e-01f,+2.22428203e-01f,
-9.56895575e-02f,+1.99410528e-01f,-3.44109029e-01f,-2.19258130e-01f,-1.14173174e-01f,
+1.29540712e-01f,+3.18143159e-01f,-1.64260328e-01f,-2.15506390e-01f,-8.66057202e-02f,
+3.70858200e-02f,+2.27383867e-01f,+2.08666354e-01f,+1.07696302e-01f,+4.47979867e-02f,
-3.66742685e-02f,-6.61358163e-02f,-9.79738832e-02f,+2.58490115e-01f,+1.61060944e-01f,
-3.50363731e-01f,-4.64481264e-02f,+2.35300109e-01f,+9.59727690e-02f,+2.13205948e-01f,
-1.86931744e-01f,-2.00111002e-01f,+2.20057026e-01f,+3.48229825e-01f,+1.18280672e-01f,
+2.91344911e-01f,+1.92185566e-01f,-4.65650037e-02f,-2.54857689e-01f,+9.55644548e-02f,
+1.88583493e-01f,+1.64745659e-01f,+2.65780807e-01f,-7.38813961e-03f,+6.69796467e-02f,
-1.88302845e-01f,-1.16829656e-01f,-8.91210977e-04f,+2.93053389e-01f,+9.07369554e-02f,
+1.66837886e-01f,-1.49897844e-01f,-1.74866438e-01f,+1.26423016e-01f,-2.46744409e-01f,
-2.40947474e-02f,+1.49082333e-01f,+2.26787582e-01f,-5.61024733e-02f,+2.06734166e-01f,
+2.65393078e-01f,-3.20584625e-01f,-1.21771172e-01f,+8.86633024e-02f,+8.85092914e-02f,
+2.18669325e-01f,-2.41880625e-01f,-1.77292377e-01f,-5.51832691e-02f,-3.41377139e-01f,
-2.20139399e-01f,-1.01504892e-01f,-4.32484262e-02f,-4.53399941e-02f,-2.91856527e-02f,
+3.81142870e-02f,-1.55926004e-01f,-2.83800066e-01f,-2.66818590e-02f,-2.54054904e-01f,
+8.52797702e-02f,+1.57977149e-01f,+1.75875410e-01f,+1.89961731e-01f,+1.09357193e-01f,
+5.09897359e-02f,+1.41976714e-01f,+8.00150782e-02f,-1.92466527e-01f,+2.27879226e-01f,
-1.73123389e-01f,-2.47257203e-01f,-8.26497972e-02f,-1.46578595e-01f,+1.40192300e-01f,
-1.42706500e-03f,-9.73592401e-02f,+2.35823944e-01f,-1.57981336e-01f,-7.04150349e-02f,
-7.96371624e-02f,+3.02264899e-01f,+1.70021236e-01f,-1.24890229e-03f,+4.54203524e-02f,
+3.72804821e-01f,-9.98451114e-02f,-3.74308348e-01f,-2.82360643e-01f,-2.10837722e-01f,
+2.00455084e-01f,+6.64289147e-02f,-1.91358954e-01f,-3.05683196e-01f,-7.51945227e-02f,
+2.79609203e-01f,-2.27707885e-02f,+1.55172333e-01f,-2.11101636e-01f,-1.10731110e-01f,
-2.30455309e-01f,+1.89286754e-01f,-2.06658065e-01f,-2.13621482e-01f,+1.34750471e-01f,
-1.11027367e-01f,-1.28825545e-01f,+2.42140219e-01f,+3.51227313e-01f,+1.68054670e-01f,
-2.44021282e-01f,-2.52200395e-01f,-2.57521898e-01f,-8.76543745e-02f,+2.14471206e-01f,
+3.02847892e-01f,-2.75299013e-01f,-1.83769837e-01f,+2.85639819e-02f,+1.00755394e-02f,
-2.27553025e-01f,+1.42039031e-01f,-2.78328300e-01f,-7.62148798e-02f,+1.90262526e-01f,
-2.34473169e-01f,+5.26524037e-02f,-4.93139997e-02f,+1.32910848e-01f,-1.28367722e-01f,
+9.02316272e-02f,+2.92954482e-02f,-2.58565962e-01f,+1.46223381e-01f,+2.54370391e-01f,
-2.91976869e-01f,-7.82655403e-02f,+2.08426729e-01f,-1.30513078e-02f,+7.94429407e-02f,
+6.54468909e-02f,-3.72628391e-01f,+1.87571734e-01f,+1.16605178e-01f,-3.92375737e-02f,
-1.64626807e-01f,+2.67205387e-01f,+8.07650387e-02f,-2.09954098e-01f,+7.70501792e-02f,
+3.53532583e-02f,+1.88803941e-01f,+1.67384773e-01f,-4.03188616e-02f,-4.65028593e-03f,
-2.32008055e-01f,-7.35586584e-02f,-1.68823123e-01f,+3.27469073e-02f,-7.54178837e-02f,
-1.40102923e-01f,+4.04514447e-02f,-7.25722313e-02f,-2.26753920e-01f,+1.75017267e-01f,
+3.19690228e-01f,+6.51303828e-02f,-2.16700897e-01f,-1.96804598e-01f,-3.25474769e-01f,
+1.30250365e-01f,-9.78152305e-02f,+1.63894519e-01f,-1.42384827e-01f,-4.28905860e-02f,
-2.05004409e-01f,+1.34975761e-01f,-1.04928270e-01f,-2.56784141e-01f,+4.52062637e-02f,
+2.04597995e-01f,-2.26757258e-01f,+1.20540261e-01f,-5.93789108e-02f,-1.04748279e-01f,
+2.54744776e-02f,-4.53838408e-02f,+3.39048296e-01f,+3.24420214e-01f,+3.40292454e-02f,
-2.34095097e-01f,-2.02886418e-01f,+2.64954627e-01f,+2.20462739e-01f,+2.49982297e-01f,
+1.00296922e-01f,-8.84539187e-02f,-1.57384455e-01f,+1.87259644e-01f,+3.47239114e-02f,
+2.04666689e-01f,-2.15592012e-01f,-2.37806126e-01f,+2.41616532e-01f,+2.58305222e-01f,
-1.87280819e-01f,+2.13689748e-02f,-1.41253576e-01f,-9.73580182e-02f,+3.43236059e-01f,
-2.07448915e-01f,-3.96234989e-01f,+1.51575387e-01f,+1.93779498e-01f,+7.85001274e-03f,
+1.20812729e-01f,-2.09669232e-01f,+1.45260811e-01f,-1.22730816e-02f,+8.15800726e-02f,
-2.07905039e-01f,-2.72078753e-01f,+1.08024135e-01f,+2.92864680e-01f,-7.20817074e-02f,
+2.80448884e-01f,-1.80108681e-01f,-2.64323413e-01f,-2.94797242e-01f,+1.55537993e-01f,
-2.47160103e-02f,+1.73748031e-01f,+1.35506978e-02f,-6.27562031e-02f,+1.23185620e-01f,
+3.73808861e-01f,-1.17392009e-02f,+1.52904252e-02f,-3.90007198e-02f,-1.19545892e-01f,
+2.52757128e-02f,+2.59529173e-01f,+1.48787484e-01f,-9.27045196e-02f,+1.31494537e-01f,
+1.88059777e-01f,+1.88918084e-01f,+2.70226806e-01f,+1.45188853e-01f,+2.62105405e-01f,
+1.18387245e-01f,+7.35683888e-02f,-2.22637698e-01f,-2.01913461e-01f,-1.80937350e-01f,
-3.49589400e-02f,+2.22152948e-01f,-1.26953468e-01f,+1.53158456e-01f,+3.29081491e-02f,
+1.97710767e-02f,+1.89050928e-01f,-5.18786237e-02f,+8.76230597e-02f,-1.80765539e-01f,
-1.30573381e-02f,+1.10710762e-01f,-9.67822596e-02f,+3.59897353e-02f,-6.42264076e-03f,
+1.10028751e-01f,-1.68894619e-01f,+1.59098133e-01f,+7.25333244e-02f,+8.82733464e-02f,
+1.31231621e-01f,+6.44780556e-03f,-8.65130574e-02f,+4.50688228e-03f,-9.60504562e-02f,
-1.79280162e-01f,+2.04335317e-01f,-1.95989460e-01f,+1.49458721e-01f,+1.05850354e-01f,
-1.41989022e-01f,+4.63434942e-02f,-1.20057724e-02f,+1.84452072e-01f,+1.39127493e-01f,
+1.76866472e-01f,+6.34561107e-03f,+2.67178893e-01f,-2.23629951e-01f,+1.69079050e-01f,
+3.00079063e-02f,+1.93323180e-01f,+5.86071834e-02f,+2.02320427e-01f,+2.30398849e-01f,
+2.17316344e-01f,-3.34900349e-01f,+1.52292792e-02f,+1.35231391e-01f,-5.21342009e-02f,
-2.39488304e-01f,-5.39673772e-03f,+2.92656600e-01f,+8.20456818e-02f,+4.78551202e-02f,
+1.60784684e-02f,-1.03636220e-01f,-1.74002368e-02f,+2.52478570e-01f,-1.60515442e-01f,
+1.52930781e-01f,-2.62108654e-01f,-9.74000618e-02f,+1.55283600e-01f,-2.32000515e-01f,
+1.29603297e-01f,+1.57881200e-01f,-1.27720580e-01f,-7.70596862e-02f,+6.51159231e-03f,
+1.50634795e-01f,-3.54961902e-02f,+2.01161221e-01f,+1.72837049e-01f,+2.35101044e-01f,
+8.02069530e-02f,+1.46596655e-01f,-1.20975584e-01f,+7.81532452e-02f,+1.35324061e-01f,
-1.77345704e-03f,+2.61497319e-01f,+1.28422827e-01f,-1.05455592e-01f,+1.51263177e-02f,
+1.40452594e-01f,-2.15443969e-01f,+1.76485509e-01f,-4.94118743e-02f,+1.81076258e-01f,
-3.81913602e-01f,-2.65132278e-01f,+2.67034583e-02f,+1.75540030e-01f,+4.38246697e-01f,
+1.75287366e-01f,-3.23170185e-01f,-1.37805361e-02f,+3.19163114e-01f,-6.74434379e-02f,
-2.65996128e-01f,-1.64119482e-01f,+3.05194259e-01f,+2.24228594e-02f,-1.55263543e-01f,
-5.40356748e-02f,-1.83893561e-01f,-3.70336585e-02f,+3.13522577e-01f,+9.41543840e-03f,
-2.02679724e-01f,-2.43094549e-01f,-9.16174427e-02f,+7.53859729e-02f,-5.52673824e-02f,
-2.04656303e-01f,-1.20689899e-01f,+6.05856702e-02f,-6.41469434e-02f,-2.48802394e-01f,
-3.53046983e-01f,+4.97268997e-02f,+1.01585023e-01f,+3.60839397e-01f,+2.57017255e-01f,
+1.64375171e-01f,-1.27000988e-01f,-2.59740144e-01f,+6.66175112e-02f,-1.41340792e-01f,
}; 
//k2c_tensor dense_31_kernel = {&dense_31_kernel_array[0],2,700,{70,10, 1, 1, 1}};
dense_31_kernel.ndim=2;
dense_31_kernel.numel=700;
dense_31_kernel.shape[0]=70;
dense_31_kernel.shape[1]=10;
dense_31_kernel.shape[2]=1;
dense_31_kernel.shape[3]=1;
dense_31_kernel.shape[4]=1;
for(i=0;i<700;i++){
//#pragma HLS unroll factor = 16
#pragma HLS unroll
	dense_31_kernel.array[i] = dense_31_kernel_array[i];
}


float dense_31_bias_array[10] = {
+2.96791606e-02f,-6.36675395e-03f,+1.38879390e-02f,+2.43084263e-02f,+2.08274163e-02f,
-1.07453987e-02f,+5.25713041e-02f,-7.20918737e-03f,+9.72352270e-03f,-6.65181037e-03f,
}; 
//k2c_tensor dense_31_bias = {&dense_31_bias_array[0],1,10,{10, 1, 1, 1, 1}};
dense_31_bias.ndim=1;
dense_31_bias.numel=10;
dense_31_bias.shape[0]=10;
dense_31_bias.shape[1]=1;
dense_31_bias.shape[2]=1;
dense_31_bias.shape[3]=1;
dense_31_bias.shape[4]=1;
for(i=0;i<10;i++){
#pragma HLS unroll factor = 16
	dense_31_bias.array[i] = dense_31_bias_array[i];
}


float dense_31_fwork[770] = {0}; 

 
float dense_32_kernel_array[10] = {
-7.35660434e-01f,-4.23227727e-01f,+8.03415775e-01f,+3.32635939e-01f,+1.86455160e-01f,
+4.25241351e-01f,-8.77878428e-01f,+3.14027160e-01f,+4.70608890e-01f,+4.61627310e-03f,
}; 
//k2c_tensor dense_32_kernel = {&dense_32_kernel_array[0],2,10,{10, 1, 1, 1, 1}};
dense_32_kernel.ndim=2;
dense_32_kernel.numel=10;
dense_32_kernel.shape[0]=10;
dense_32_kernel.shape[1]=1;
dense_32_kernel.shape[2]=1;
dense_32_kernel.shape[3]=1;
dense_32_kernel.shape[4]=1;
for(i=0;i<10;i++){
#pragma HLS unroll factor = 16
	dense_32_kernel.array[i] = dense_32_kernel_array[i];
}


float dense_32_bias_array[1] = {
+1.96657237e-03f,}; 
// k2c_tensor dense_32_bias = {&dense_32_bias_array[0],1,1,{1,1,1,1,1}};
dense_32_bias.ndim=1;
dense_32_bias.numel=1;
dense_32_bias.shape[0]=1;
dense_32_bias.shape[1]=1;
dense_32_bias.shape[2]=1;
dense_32_bias.shape[3]=1;
dense_32_bias.shape[4]=1;
for(i=0;i<1;i++){
#pragma HLS unroll factor = 16
	dense_32_bias.array[i] = dense_32_bias_array[i];
}


float dense_32_fwork[20] = {0}; 


//#pragma HLS STREAM variable=layer21_out depth=10201
k2c_pad2d(&conv2d_31_padded_input,input_9_input,conv2d_31_fill, 
	conv2d_31_pad);

//#pragma HLS STREAM variable=layer2_out depth=2500
k2c_conv2d(&conv2d_31_output,&conv2d_31_padded_input,&conv2d_31_kernel, 
	&conv2d_31_bias,conv2d_31_stride,conv2d_31_dilation);

//#pragma HLS STREAM variable=layer4_out depth=625
k2c_maxpool2d(&max_pooling2d_18_output,&conv2d_31_output,max_pooling2d_18_pool_size, 
	max_pooling2d_18_stride); 


k2c_pad2d(&conv2d_32_padded_input,&max_pooling2d_18_output,conv2d_32_fill, 
	conv2d_32_pad); 


k2c_conv2d(&conv2d_32_output,&conv2d_32_padded_input,&conv2d_32_kernel, 
	&conv2d_32_bias,conv2d_32_stride,conv2d_32_dilation);


k2c_maxpool2d(&max_pooling2d_19_output,&conv2d_32_output,max_pooling2d_19_pool_size, 
	max_pooling2d_19_stride);


k2c_pad2d(&conv2d_33_padded_input,&max_pooling2d_19_output,conv2d_33_fill, 
	conv2d_33_pad);


k2c_conv2d(&conv2d_33_output,&conv2d_33_padded_input,&conv2d_33_kernel, 
	&conv2d_33_bias,conv2d_33_stride,conv2d_33_dilation);


k2c_pad2d(&conv2d_34_padded_input,&conv2d_33_output,conv2d_34_fill,
	conv2d_34_pad);


k2c_conv2d(&conv2d_34_output,&conv2d_34_padded_input,&conv2d_34_kernel, 
	&conv2d_34_bias,conv2d_34_stride,conv2d_34_dilation);


k2c_flatten(&flatten_8_output,&conv2d_34_output);


k2c_dense(&dense_29_output,&flatten_8_output,&dense_29_kernel, 
	&dense_29_bias,1,dense_29_fwork);


k2c_dense(&dense_30_output,&dense_29_output,&dense_30_kernel, 
	&dense_30_bias,1,dense_30_fwork);


k2c_dense(&dense_31_output,&dense_30_output,&dense_31_kernel, 
	&dense_31_bias,1,dense_31_fwork);


k2c_dense(dense_32_output,&dense_31_output,&dense_32_kernel, 
	&dense_32_bias,0,dense_32_fwork);

 } 

void braintumer_initialize() { 

} 

void braintumer_terminate() { 

} 


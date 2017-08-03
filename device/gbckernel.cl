// Copyright (C) 2013-2014 Altera Corporation, San Jose, California, USA. All rights reserved. 
// Permission is hereby granted, free of charge, to any person obtaining a copy of this 
// software and associated documentation files (the "Software"), to deal in the Software 
// without restriction, including without limitation the rights to use, copy, modify, merge, 
// publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to 
// whom the Software is furnished to do so, subject to the following conditions: 
// The above copyright notice and this permission notice shall be included in all copies or 
// substantial portions of the Software. 
//  
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, 
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES 
// OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND 
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT 
// HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, 
// WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR 
// OTHER DEALINGS IN THE SOFTWARE. 
//  
// This agreement shall be governed in all respects by the laws of the State of California and 
// by the laws of the United States of America. 

 // ACL kernel for adding two input vectors

#define BLOCK_SIZE 16

//__attribute__((num_compute_units(8)))

__attribute((reqd_work_group_size(BLOCK_SIZE,BLOCK_SIZE,1)))
__attribute((num_simd_work_items(4)))

__kernel void gbc(__global float* restrict matrix, // input-data, e.g. 80000x192 
		  __global float* restrict gbcresult,    // output-data, 32x80000
		  int width,                             // 192
		  int startseed                          // 0, 32, 64, ...
		  )
{
  // For each working id, calculate the correlation between
  // global_row id and global_col+startseed
  // such that:
  //    A = global_col+startseed; (the seeds)
  //    B = global_row (the whole image)


  __local float A_local[BLOCK_SIZE*BLOCK_SIZE];
  __local float B_local[BLOCK_SIZE*BLOCK_SIZE];

  int global_row = get_global_id(0); // 0 .. 79999
  int global_col = get_global_id(1); // 0 .. 31
  
  int local_row = get_local_id(0); // rows
  int local_col = get_local_id(1); // columns
  
  //  float old = gbcresult[global_row*width+global_col];
  float running_sum = 0;
  int a, b;
  
  for(a=0;a<width;a+=BLOCK_SIZE) {
    A_local[local_col*BLOCK_SIZE+local_row] = matrix[(startseed+global_col)*width+a+local_row];
    B_local[local_row*BLOCK_SIZE+local_col] = matrix[global_row*width+local_col+a];
    barrier(CLK_LOCAL_MEM_FENCE);
        #pragma unroll
    for(int k=0;k<BLOCK_SIZE;++k) {
      running_sum += A_local[local_col*BLOCK_SIZE+k]*B_local[local_row*BLOCK_SIZE+k];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  
  if(running_sum<0)
    running_sum = 0;

  gbcresult[global_row*BLOCK_SIZE+global_col] += running_sum;
}


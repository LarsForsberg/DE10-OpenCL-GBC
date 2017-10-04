#define BLOCK_SIZE 16

__attribute((reqd_work_group_size(BLOCK_SIZE,BLOCK_SIZE,1)))
__attribute((num_simd_work_items(4)))

__kernel void gbc(__global float* restrict matrix,       // input-data, e.g. 80000x192
		              __global float* restrict accumulator,  // output-data, 16x80000
		              int width,                             // 192
		              int start_row                          // 0, 16, 32, ...
		  )
{
  int i, j;

  __local float A_local[BLOCK_SIZE*BLOCK_SIZE];
  __local float B_local[BLOCK_SIZE*BLOCK_SIZE];

  int global_col = get_global_id(0); // 0 .. 79999
  int global_row = get_global_id(1); // 0 .. 15
  int num_rows = get_global_size(1);

  int local_col = get_local_id(0); // rows
  int local_row = get_local_id(1); // columns

  float acc = accumulator[global_row*BLOCK_SIZE+global_col];
  float sum = 0;


  for(i=0;i<width;i+=BLOCK_SIZE) {
    A_local[local_row*BLOCK_SIZE+local_col] = matrix[(start_row+local_row)*width+local_col+i];
    B_local[local_row*BLOCK_SIZE+local_col] = matrix[global_row*width+local_col+i];
    barrier(CLK_LOCAL_MEM_FENCE);
    #pragma unroll
    for(int j=0;j<BLOCK_SIZE;++j) {
      sum += A_local[local_col*BLOCK_SIZE+j]*B_local[local_row*BLOCK_SIZE+j];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if(sum<0)
    sum = 0;
  acc += sum;

  accumulator[global_row*BLOCK_SIZE+global_col] = acc;
}

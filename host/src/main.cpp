#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "CL/opencl.h"
#include "AOCLUtils/aocl_utils.h"
#include "nifti1.h"
#include "fslio.h"


#define BLOCK_SIZE 16

short xsize, ysize, zsize, vsize;

using namespace aocl_utils;

cl_platform_id platform = NULL;
cl_device_id device = NULL;
cl_program program = NULL;
cl_command_queue queue = NULL;
cl_context context = NULL;
cl_kernel kernel = NULL;
cl_mem matrixmem = NULL;
cl_mem gbcresultmem = NULL;


float* matrix;
float* matrix2;
float* gbcresult;
float* finalvector;
float* imagebuffer;
float* mask;

char* inputfilename;
char* maskfilename;
char* outputfilename;

bool init_opencl(int nvoxelstotal, int ntimestotal);
void normalise(float* matrix, int width, int height);
void calc(float* matrix, int startseed, int ntimes, int nvoxels);
float* getImage(char* filename);
void writeImage();

int main(int argc,char* argv[]) {
  int i,j;
  int x,y,z,t;
  int startseed;
  float value;

  size_t global[2];
  size_t local[2];
  cl_int status;
  cl_event event;
  int fpga = 1;
  int nvoxels = 0;

  float* data;

  if(argc<3 || argc>4) {
    printf("Usage: gbc input.nii output.nii\n");
    printf("       gbc input.nii mask.nii output.nii\n");
    return 1;
  }
  if(argc==3) {
    inputfilename = argv[1];
    outputfilename = argv[2];
    printf("Input: %s\n",inputfilename);
    printf("Output: %s\n",outputfilename);
  }
  if(argc==4) {
    inputfilename = argv[1];
    maskfilename = argv[2];
    outputfilename = argv[3];
    printf("Input: %s\n",inputfilename);
    printf("Output: %s\n",outputfilename);
    printf("Mask: %s\n",maskfilename);
  }




  if(argc==4) {
    mask = getImage(maskfilename);

    for(i=0;i<xsize*ysize*zsize;i++) {
      if(mask[i] > 0)
	nvoxels++;
    }
  }

  data = getImage(inputfilename);

  if(argc==3) {
    mask = (float*) malloc(sizeof(float)*xsize*ysize*zsize);
    for(i=0;i<xsize*ysize*xsize;i++) {
      mask[i] = 0;
    }
  }



  imagebuffer = (float*) malloc(sizeof(float)*xsize*ysize*zsize);

  printf("fMRI dimensions: %d %d %d %d\n",xsize,ysize,zsize,vsize);
  if(argc==3) {
    printf("Calculating mask.\n");
    for(t=0;t<vsize;t++) {
      for(z=0;z<zsize;z++) {
	for(y=0;y<ysize;y++) {
	  for(x=0;x<xsize;x++) {
	    mask[x+y*xsize+z*xsize*ysize] += fabs(data[x+y*xsize+z*xsize*ysize+t*xsize*ysize*zsize]);
	  }
	}
      }
    }

    for(i=0;i<(xsize*ysize*zsize);i++) {
      if(mask[i] > 0) {
	nvoxels++;
      }
    }
  }

  printf("NVOXELS: %d\n",nvoxels);

  int nvoxelstotal = nvoxels+(16-nvoxels%BLOCK_SIZE);
  int ntimes = vsize;
  int ntimestotal = ntimes+(16-ntimes%BLOCK_SIZE);

  printf("Initializing OpenCL\n");
  if(!init_opencl(nvoxelstotal,ntimestotal))
    return -1;

  printf("Dimensions matrix 1: %dx%d\n",nvoxels,ntimes);
  printf("Dimensions matrix 2: %dx%d\n",nvoxelstotal,ntimestotal);

  matrix = (float*) malloc(sizeof(float)*nvoxels*ntimes);
  matrix2 = (float*) malloc(sizeof(float)*nvoxelstotal*ntimestotal);
  gbcresult = (float*) malloc(sizeof(float)*nvoxelstotal*BLOCK_SIZE);
  finalvector = (float*) malloc(sizeof(float)*nvoxels);

  for(i=0;i<nvoxelstotal;i++) {
    for(j=0;j<BLOCK_SIZE;j++)
      gbcresult[i*BLOCK_SIZE+j] = 0;
  }

  printf("Creating matrix 1.\n");
  i=0;
  for(x=0;x<xsize;x++) {
    for(y=0;y<ysize;y++) {
      for(z=0;z<zsize;z++) {
	value = mask[x+y*xsize+z*xsize*ysize];
	if(value>0) {
	  for(t=0;t<vsize;t++) {
	    value = data[x+y*xsize+z*xsize*ysize+t*xsize*ysize*zsize];
	    matrix[i++] = value;
	  }
	}
      }
    }
  }

  printf("Normalising matrix 1.\n");
  normalise(matrix,ntimes,nvoxels);
  printf("Creating matrix 2.\n");

  for(x=0;x<nvoxelstotal;x++) {
    for(t=0;t<ntimestotal;t++) {
      if(x<nvoxels && t<ntimes)
	matrix2[x*ntimestotal+t] = matrix[x*ntimes+t];
      else
	matrix2[x*ntimestotal+t] = 0;
    }
  }
  free(matrix);

  printf("Writing to global memory.\n");
  clEnqueueWriteBuffer(queue, matrixmem, CL_TRUE, 0, sizeof(float)*nvoxelstotal*ntimestotal, matrix2, 0, NULL, NULL);
  clEnqueueWriteBuffer(queue, gbcresultmem, CL_TRUE, 0, sizeof(float)*nvoxelstotal*BLOCK_SIZE, gbcresult, 0, NULL, NULL);

  startseed = 0;
  int width = ntimestotal;
  startseed = 0;

  clSetKernelArg(kernel, 0, sizeof(cl_mem), &matrixmem);
  clSetKernelArg(kernel, 1, sizeof(cl_mem), &gbcresultmem);
  clSetKernelArg(kernel, 2, sizeof(int), &width);
  clSetKernelArg(kernel, 3, sizeof(int), &startseed);

  // Let's run
  printf("Running kernel.\n");
  global[0] = nvoxelstotal;
  global[1] = BLOCK_SIZE;
  local[0] = BLOCK_SIZE;
  local[1] = BLOCK_SIZE;
  int num_of_iterations = nvoxelstotal/BLOCK_SIZE;

  printf("Computing %d blocks.\n",num_of_iterations);
  //  num_of_iterations = 1;
  for(i=0;i<num_of_iterations;i++) {
    printf("Iteration %d of %d. ",i+1, num_of_iterations);
    if(fpga==1) {
      clSetKernelArg(kernel, 3, sizeof(int), &startseed);
      status = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global, local, 0, NULL, &event);
      clWaitForEvents(1, &event);
      cl_ulong time_ns = getStartEndTime(event);
      printf("Kernel time: %0.3f s\n", double(time_ns) * 1e-9);
    } else {
      calc(matrix2, startseed, ntimestotal, nvoxelstotal);
      printf("\n");
    }
    startseed += BLOCK_SIZE;
  }
  if(fpga==1) {
    printf("Reading output from global memory.\n");
    status = clEnqueueReadBuffer(queue, gbcresultmem, CL_TRUE, 0, sizeof(float)*nvoxelstotal*BLOCK_SIZE, gbcresult, 1, &event, NULL);
    clWaitForEvents(1, &event);
  }

  printf("Calculating final image.\n");

  for(i=0;i<nvoxels;i++) {
    finalvector[i] = 0;
    for(j=0;j<BLOCK_SIZE;j++) {
      finalvector[i] += gbcresult[i*BLOCK_SIZE+j];
    }
    finalvector[i]--;
  }

  writeImage();
  cleanup();
  printf("Done!\n");
}

void calc(float* matrix, int startseed, int ntimes, int nvoxels)
{
  int x,y,z;
  float sum;

  for(x=0;x<BLOCK_SIZE;x++) {
    for(y=0;y<nvoxels;y++) {
      sum = 0;
      for(z=0;z<ntimes;z++) {
	sum += matrix[(startseed+x)*ntimes+z]*matrix[y*ntimes+z];
      }
      if(sum<0)
	sum = 0;
      gbcresult[y*BLOCK_SIZE+x] += sum;
    }
  }
}

void normalise(float* matrix, int width, int height)
{
  int row,col;

  double sum, sq_sum;
  double value;
  double mean;
  double variance;
  double sd;

  for(row=0;row<height;row++) {
    sum = 0;
    sq_sum = 0;
    for(col=0;col<width;col++) {
      value = matrix[row*width+col];
      sum += value;
      sq_sum += value*value;
    }
    mean = sum/width;
    variance = sq_sum/width-mean*mean;
    sd = sqrt(variance);
    for(col=0;col<width;col++) {
      value =  matrix[row*width+col];
      matrix[row*width+col] = (float) ((value-mean)/sd)/sqrt(width);
    }
  }

  printf("Done normalising.\n");
}



bool init_opencl(int nvoxelstotal, int ntimestotal) {
  cl_uint numofdevices;
  cl_int status;

  platform = findPlatform("Intel(R) FPGA");
  if(platform == NULL) {
    printf("ERROR: Unable to find Intel OpenCL platform.\n");
    return false;
  }

  clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, &device, &numofdevices);

  context = clCreateContext(0, 1, &device, NULL, NULL, &status);
  program = createProgramFromBinary(context, "gbckernel.aocx", &device, numofdevices);

  status = clBuildProgram(program, 0, NULL, "", NULL, NULL);

  queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &status);

  kernel = clCreateKernel(program, "gbc", &status);

  matrixmem = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float)*nvoxelstotal*ntimestotal, NULL, &status);
  gbcresultmem = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float)*nvoxelstotal*BLOCK_SIZE, NULL, &status);


  return true;
}

float* getImage(char* filename)
{
  FSLIO *imageIO;
  int size;

  imageIO = FslOpen(filename,"r");

  FslGetDim(imageIO, &xsize, &ysize, &zsize, &vsize);

  size = xsize*ysize*zsize*vsize;
  float* imagebuffer = (float*) malloc(sizeof(float)*size);
  FslReadVolumes(imageIO, imagebuffer, vsize);
  float* buffer = (float*) imagebuffer;
  FslClose(imageIO);

  return buffer;
}

void writeImage()
{
  int x,y,z,voxel;

  voxel = 0;
  for(x=0;x<xsize;x++) {
    for(y=0;y<ysize;y++)
      for(z=0;z<zsize;z++) {
	if(mask[x+y*xsize+z*xsize*ysize]>0) {
	  imagebuffer[x+y*xsize+z*xsize*ysize] = finalvector[voxel];
	  voxel++;
	} else
	  imagebuffer[x+y*xsize+z*xsize*ysize] = 0;
      }
  }

  nifti_image *image;
  float* tmp;

  image = nifti_image_read(inputfilename,1);

  nifti_set_filenames(image,outputfilename,0,1);
  image->dim[4] = 1;
  image->ndim = 3;
  image->nt = 1;
  image->nvox = image->dim[1]*image->dim[2]*image->dim[3];

  tmp = (float*) image->data;
  image->data = imagebuffer;
  nifti_image_write(image);
  image->data = tmp;
  nifti_image_free(image);
}

void cleanup()
{
  clReleaseKernel(kernel);
  clReleaseCommandQueue(queue);
  clReleaseMemObject(matrixmem);
  clReleaseMemObject(gbcresultmem);
  clReleaseProgram(program);
  clReleaseContext(context);
  free(matrix2);
  free(gbcresult);
}

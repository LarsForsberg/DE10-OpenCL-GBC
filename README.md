# DE10GBC (Global Brain Connectivity on Terasic DE10-Standard)

## Introduction

Functional brain imaging is a neuroimaging technique where the functionality of the brain is measured over time. Resting State fMRI (RS-fMRI) is a method where the brain's functionality is investigated during rest, i.e. when the subject is not performing an explicit task. The data consists of a series of 3D volumes, where the data points within a volume are called "voxels" (instead of "pixels").

A common research field within RS-fMRI is functional connectivity, where the temporal correlation between a "seed voxel" and all other voxels are measured. The result for a given seed voxel is a "seed map", showing the correlation level with all other voxels.

For a given "seed map", the sum of all voxels correlating positively with the seed voxel will then give the degree of connectivity for the seed voxel. A low value indicates a low connectivity, whereas a high value indicates a high level of connectivity. By calculating the degree of connectivity for each voxel, a "Global Brain Connectivity" (GBC) map is obtained. A commonly used tool for GBC is AFNI's 3dTcorrMap.

This tool (DE10GBC) is an alternative GBC tool for running on a Terasic DE10-Standard, where the calculations are accelerated on the FPGA. A speed comparison show that DE10GBC on the Terasic DE10-Standard is faster than 3dTcorrMap on a MacBook Pro 2016 quad core, even when running the OpenMP version of 3dTCorrMap with 8 threads.

## Terasic DE10-Standard


The Terasic DE10-Standard is a great low-powered development platform built around the largest Cyclone-V SoC with a dual-core ARM processor and a FPGA device with 110K LE. Terasic provides a BSP (Board Support Package) for Intel FPGA SDK OpenCL, making it a perfect entry board for OpenCL on FPGA. The current price is $350 ($259 academic price):

<http://www.terasic.com.tw/cgi-bin/page/archive.pl?Language=English&CategoryNo=167&No=1081>



## Installation

### Compiling from source

The following text assumes you have installed a valid OpenCL environment on a Linux machine for the DE10-Standard developer board. Start with copying the gbc folder to the de10\_standard/examples folder on your Linux developer machine. Go to the de10\_standard/examples folder. To compile the host software, type "make". To synthesise the OpenCL device software into FPGA logic, type "./synthesise". This will generate the necessary software in the bin directory. You will have two files in the bin directory:

  - gbc (the host file)
  - gbckernel.aocx (the device file) 

Compiling from source is optional. You can also use the precompiled binaries found in the bin directory of this distribution.

### Installing the binaries

Copy the bin directory to a directory of your choice on the DE10-Standard, e.g. ~/gbc/bin.

## Running

You need a NIfTI 4D image with an fMRI timeseries and optionally a mask image to run this software. You can obtain an example from public databases such as ABIDE. The NIfTI file must not be compressed, so nii.gz files must first be uncompressed before running. Here is a direct link to an example file from ABIDE, that you can use as indata file (after uncompressing it):

<https://s3.amazonaws.com/fcp-indi/data/Projects/ABIDE_Initiative/Outputs/cpac/filt_global/func_preproc/KKI_0050822_func_preproc.nii.gz>

Here are two examples of how to run the software:

- gbc indata.nii outdata.nii
- gbc indata.nii mask.nii outdata.nii

In the first example, the mask will be automatically calculated by only including non-zero voxels in the indata.nii file. In the second example, a mask.nii file is used as the mask. The indata.nii file is the 4D fMRI dataset and the outdata.nii file is the 3D output. The second example corresponds to the following expression in 3dTcorrMap:

- 3dTcorrMap -input indata.nii -mask mask.nii -Sexpr 'step(r)*r' outdata.nii

## YouTube comparison video

This YouTube video compares the speed between running DE10GBC and AFNI's 3dTcorrMap. The result shows that the DE10GBC custom software requires 392 seconds whereas the OpenMP version of 3dTcorrMap running with 8 threads on a MacBook Pro (2016) requires 423 seconds. The end results are approximately the same.

[![IMAGE ALT TEXT](http://img.youtube.com/vi/N8zh7ErCZ8o/0.jpg)](https://www.youtube.com/watch?v=N8zh7ErCZ8o "Terasic DE10 Standard running OpenCL")

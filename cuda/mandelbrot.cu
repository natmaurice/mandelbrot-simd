#include "mandelbrot-cuda.h"
#include <bits/time.h>
#include <cuda.h>
#include <time.h>
#include <stdio.h>


__global__
void mandelbrot_kernel(int32_t* mat, int nrl, int nrh, int ncl, int nch, int stride,
		       float minx, float miny, float maxx, float maxy, int max_iterations) {


    int width = nch - ncl + 1;
    int height = nrh - nrl + 1;

    float stepx = (maxx - minx) / width;
    float stepy = (maxy - miny) / height;
    
    for (int ri = threadIdx.y + blockDim.y * blockIdx.y; ri < height; ri += blockDim.y * gridDim.y) {
	int32_t* line = mat + ri * stride;
	const int i = ri - nrl;
	const float y0 = miny + i * stepy;
	
	for (int rj = threadIdx.x + blockDim.x * blockIdx.x; rj < width; rj += blockDim.x * gridDim.x) {
	    const int j = rj - ncl;

	    int it = 0;

	    const float x0 = minx + j * stepx;
	    float x = 0, y = 0;
	    float x2 = 0, y2 = 0;

	    int final_it = max_iterations;
	    int mask = 0xffffffff;
	    
	    for (it = 0; it < max_iterations; it++) {
		
		y = 2 * x * y + y0;
		x = x2 - y2 + x0;
		x2 = x * x;
		y2 = y * y;
		
		if (x2 + y2 > 2*2) {
		    // Don't modify final_it if this condition has already been encountered
		    final_it = (it & mask) | (final_it & (~mask));
		    mask = 0x0;
		}
	    }
	    line[j] = final_it;
	}
    }
}

__global__
void mandelbrot_kernel_f16(int32_t* mat, int nrl, int nrh, int ncl, int nch, int stride,
			   float minx, float miny, float maxx, float maxy, int max_iterations) {


    int width = nch - ncl + 1;
    int height = nrh - nrl + 1;

    float stepx = (maxx - minx) / width;
    float stepy = (maxy - miny) / height;
    
    for (int ri = threadIdx.y + blockDim.y * blockIdx.y; ri < height; ri += blockDim.y * gridDim.y) {
	int32_t* line = mat + ri * stride;
	const int i = ri - nrl;
	const float y0 = miny + i * stepy;
	
	for (int rj = threadIdx.x + blockDim.x * blockIdx.x; rj < width; rj += blockDim.x * gridDim.x) {
	    const int j = rj - ncl;

	    int it = 0;

	    const float x0 = minx + j * stepx;
	    float x = 0, y = 0;
	    float x2 = 0, y2 = 0;

	    int final_it = max_iterations;
	    int mask = 0xffffffff;
	    
	    for (it = 0; it < max_iterations; it++) {
		
		y = 2 * x * y + y0;
		x = x2 - y2 + x0;
		x2 = x * x;
		y2 = y * y;
		
		if (x2 + y2 > 2*2) {
		    // Don't modify final_it if this condition has already been encountered
		    final_it = (it & mask) | (final_it & (~mask));
		    mask = 0x0;
		}
	    }
	    line[j] = final_it;
	}
    }
}



__global__
void mandelbrot_kernel_v2(int32_t* mat, int nrl, int nrh, int ncl, int nch, int stride,
			  float minx, float miny, float maxx, float maxy, int max_iterations) {


    int width = nch - ncl + 1;
    int height = nrh - nrl + 1;

    float stepx = (maxx - minx) / width;
    float stepy = (maxy - miny) / height;
    
    for (int ri = threadIdx.y + blockDim.y * blockIdx.y; ri < height; ri += blockDim.y * gridDim.y) {
	int32_t* line = mat + ri * stride;
	const int i = ri - nrl;
	const float y0 = miny + i * stepy;
	
	for (int rj = threadIdx.x + blockDim.x * blockIdx.x; rj < width; rj += blockDim.x * gridDim.x) {
	    const int j = rj - ncl;

	    int it = 0;

	    const float x0 = minx + j * stepx;
	    float x = 0, y = 0;
	    float x2 = 0, y2 = 0;

	    int final_it = max_iterations;
	    int mask = 0xffffffff;
	    
	    for (it = 0; it < max_iterations; it++) {
		
		y = 2 * x * y + y0;
		x = x2 - y2 + x0;
		x2 = x * x;
		y2 = y * y;
		
		if (x2 + y2 > 2*2) {
		    break;
		}
	    }
	    line[j] = it;
	}
    }
}


__global__
void mandelbrot_kernel_lu2(int32_t* mat, int nrl, int nrh, int ncl, int nch, int stride,
			   float minx, float miny, float maxx, float maxy, int max_iterations) {


    int width = nch - ncl + 1;
    int height = nrh - nrl + 1;

    float stepx = (maxx - minx) / width;
    float stepy = (maxy - miny) / height;

    constexpr int ELEM_PER_IT = 2;
    
    for (int ri = threadIdx.y + blockDim.y * blockIdx.y; ri < height; ri += blockDim.y * gridDim.y) {
	int32_t* line = mat + ri * stride;
	const int i = ri - nrl;
	const float vy0 = miny + i * stepy;
	
	for (int rj = (threadIdx.x + blockDim.x * blockIdx.x) * ELEM_PER_IT;
	     rj < width;
	     rj += (blockDim.x * gridDim.x) * ELEM_PER_IT) {

	    
	    const int j = rj - ncl;

	    int it = 0;
	    int final_it0 = max_iterations, final_it1 = max_iterations;
	    
	    const float vx0 = minx + j * stepx;
	    const float vx1 = minx + ((j + 1) * stepx);
	    
	    float x0 = 0, y0 = 0;
	    float xb0 = 0, yb0 = 0;
	    float x1 = 0, y1 = 0;
	    float xb1 = 0, yb1 = 0;

	    int mask0, mask1;
	    mask0 = mask1 = 0xffffffff;
	    
	    for (it = 0; it < max_iterations; it++) {
		
		y0 = 2 * x0 * y0 + vy0;
		y1 = 2 * x1 * y1 + vy0;
		
		x0 = xb0 - yb0 + vx0;
		x1 = xb1 - yb1 + vx1;

		xb0 = x0 * x0;
		xb1 = x1 * x1;

		yb0 = y0 * y0;
		yb1 = y1 * y1;
		
		if (xb0 + yb0 > 2*2) {
		    // Don't modify final_it if this condition has already been encountered
		    final_it0 = (it & mask0) | (final_it0 & (~mask0));
		    mask0 = 0x0;
		}
		if (xb1 + xb1 > 2*2) {
		    final_it1 = (it & mask1) | (final_it1 & (~mask1));
		    mask1 = 0x0;
		}
	    }
	    line[j    ] = final_it0;
	    line[j + 1] = final_it1;
	}
    }
}




struct Mandelbrot_V0 {
    static void Execute(dim3 nblocks, dim3 threadsPerBlock, 
			    int32_t* dev_mat, int nrl, int nrh, int ncl, int nch, int stride,
			    float minx, float miny, float maxx, float maxy, int max_iterations) {
    
	mandelbrot_kernel<<<nblocks, threadsPerBlock>>>(dev_mat, nrl, nrh, ncl, nch, stride, minx, miny, maxx, maxy, max_iterations);
    }
};


struct Mandelbrot_V2 {
    static void Execute(dim3 nblocks, dim3 threadsPerBlock, 
			    int32_t* dev_mat, int nrl, int nrh, int ncl, int nch, int stride,
			    float minx, float miny, float maxx, float maxy, int max_iterations) {
    
	mandelbrot_kernel_v2<<<nblocks, threadsPerBlock>>>(dev_mat, nrl, nrh, ncl, nch, stride, minx, miny, maxx, maxy, max_iterations);
    }
};


struct Mandelbrot_LU2 {
    static void Execute(dim3 nblocks, dim3 threadsPerBlock, 
			    int32_t* dev_mat, int nrl, int nrh, int ncl, int nch, int stride,
			    float minx, float miny, float maxx, float maxy, int max_iterations) {
    
	mandelbrot_kernel_lu2<<<nblocks, threadsPerBlock>>>(dev_mat, nrl, nrh, ncl, nch, stride, minx, miny, maxx, maxy, max_iterations);
    }
};


template <class Kernel>    
void mandelbrot_generic(int32_t** mat, int nrl, int nrh, int ncl, int nch,
			float minx, float miny, float maxx, float maxy, int max_iterations) {
    
    int height = nrh - nrl + 1;
    int width = nch - ncl + 1;

    
    dim3 threadsPerBlock(16, 16);
    dim3 nblocks(width / threadsPerBlock.x, height / threadsPerBlock.y);
    //dim3 nblocks(width / threadPerBlock.x, height / threadPerBlock.y);
    
    int stride = width;
    if (height > 1) {
	stride = mat[1] - mat[0];
    }

    int32_t* dev_mat;
    
    cudaMalloc((void**)&dev_mat, height*stride*sizeof(int32_t));


    struct timespec ts_start, ts_end;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts_start);
    
    Kernel::Execute(nblocks, threadsPerBlock, dev_mat, nrl, nrh, ncl, nch, stride, minx, miny, maxx, maxy, max_iterations);
    cudaDeviceSynchronize();
    
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts_end);

    double elapsed_ms = (ts_end.tv_sec - ts_start.tv_sec) * 1000 + (ts_end.tv_nsec - ts_start.tv_nsec) / 1e6;
    
    
    cudaMemcpy(mat[0], dev_mat, height*stride*sizeof(int32_t), cudaMemcpyDeviceToHost);

    long pixel_count = width * height;
    
    float pixel_per_s = pixel_count / (elapsed_ms / 1000.0);
    float gpixels = pixel_per_s / 1e9;
    
    printf("Elapsed GPU time = %f\n", elapsed_ms);
    printf("Bandwidth = %f GPix/s\n", gpixels);


    
    cudaFree(dev_mat);
}

void mandelbrot_cuda(int32_t **mat, int nrl, int nrh, int ncl, int nch,
                     float minx, float miny, float maxx, float maxy,
                     int max_iterations) {
    
    mandelbrot_generic<Mandelbrot_V0>(mat, nrl, nrh, ncl, nch, minx, miny, maxx, maxy, max_iterations);
}


void mandelbrot_cuda_v2(int32_t **mat, int nrl, int nrh, int ncl, int nch,
                     float minx, float miny, float maxx, float maxy,
                     int max_iterations) {
    
    mandelbrot_generic<Mandelbrot_V2>(mat, nrl, nrh, ncl, nch, minx, miny, maxx, maxy, max_iterations);
}


void mandelbrot_cuda_lu2(int32_t **mat, int nrl, int nrh, int ncl, int nch,
                     float minx, float miny, float maxx, float maxy,
                     int max_iterations) {
    mandelbrot_generic<Mandelbrot_LU2>(mat, nrl, nrh, ncl, nch, minx, miny, maxx, maxy, max_iterations);
}
#ifndef MANDELBROT_CUDA_MANDELBROT_H
#define MANDELBROT_CUDA_MANDELBROT_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

void mandelbrot_cuda(int32_t **mat, int nrl, int nrh, int ncl, int nch,
                     float minx, float miny, float maxx, float maxy,
                     int max_iterations);

    
void mandelbrot_cuda_v2(int32_t **mat, int nrl, int nrh, int ncl, int nch,
			float minx, float miny, float maxx, float maxy,
			int max_iterations);


void mandelbrot_cuda_lu2(int32_t **mat, int nrl, int nrh, int ncl, int nch,
			 float minx, float miny, float maxx, float maxy,
			 int max_iterations);

    

    
#ifdef __cplusplus
}
#endif // __cplusplus

#endif // MANDELBROT_CUDA_MANDELBROT_H

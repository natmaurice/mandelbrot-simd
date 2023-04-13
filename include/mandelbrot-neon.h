#ifndef MANDELBROT_NEON_H_
#define MANDELBROT_NEON_H_

#ifdef __ARM_NEON

#include <stdint.h>

void mandelbrot_neon(int32_t** mat, int nrl, int nrh, int ncl, int nch,
		       float minx, float miny, float maxx, float maxy, int max_iterations);

void mandelbrot_parallel_neon(int32_t** mat, int nrl, int nrh, int ncl, int nch,
			      float minx, float miny, float maxx, float maxy, int max_iterations);

#endif // __ARM_NEON
       // 
#endif // MANDELBROT_NEON_H_

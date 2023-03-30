#ifndef MANDELBROT_AVX512_H_
#define MANDELBROT_AVX512_H_

#include <stdint.h>


void mandelbrot_avx512(int32_t** mat, int nrl, int nrh, int ncl, int nch,
		       float minx, float miny, float maxx, float maxy, int max_iterations);

void mandelbrot_avx512_v2(int32_t** mat, int nrl, int nrh, int ncl, int nch,
			  float minx, float miny, float maxx, float maxy, int max_iterations);

void mandelbrot_avx512_lu2_pl(int32_t** mat, int nrl, int nrh, int ncl, int nch,
			      float minx, float miny, float maxx, float maxy, int max_iterations);

void mandelbrot_avx512_lu4(int32_t** mat, int nrl, int nrh, int ncl, int nch,
			   float minx, float miny, float maxx, float maxy, int max_iterations);

void mandelbrot_avx512_lu4_pl(int32_t** mat, int nrl, int nrh, int ncl, int nch,
			      float minx, float miny, float maxx, float maxy, int max_iterations);

void mandelbrot_parallel_avx512(int32_t** mat, int nrl, int nrh, int ncl, int nch, float minx,
				float maxx, float miny, float maxy, int max_iterations);

void mandelbrot_parallel_avx512_lu4(int32_t** mat, int nrl, int nrh, int ncl, int nch,
				    float minx, float miny, float maxx, float maxy, int max_iterations);


#endif // MANDELBROT_AVX512_H_

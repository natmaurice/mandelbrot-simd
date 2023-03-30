#ifndef MANDELBROT_H_
#define MANDELBROT_H_


#include <stdint.h>

void mandelbrot_scalar(int32_t** mat, int nrl, int nrh, int ncl, int nch,
		       float minx, float miny, float maxx, float maxy, int max_iterations);

void mandelbrot_parallel_scalar(int32_t** mat, int nrl, int nrh, int ncl, int nch,
		       float minx, float miny, float maxx, float maxy, int max_iterations);

#endif // MANDELBROT_H_

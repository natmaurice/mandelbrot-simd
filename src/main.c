#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>

#include <nrtype.h>
#include <nralloc2.h>
#include <nrio.h>
#include <nrset2.h>

#include <omp.h>



#ifdef __x86_64__
#include <immintrin.h>
#endif // __x86_64__


#include "mandelbrot.h"

#ifdef __AVX512F__
#include "mandelbrot-avx512.h"
#endif // __AVX512F__


#ifdef MANDELBROT_USE_CUDA
#include "mandelbrot-cuda.h"
#endif // MANDELBROT_USE_CUDA

void color_image(rgb8** rgbmat, int32_t** it_mat, int nrl, int nrh, int ncl, int nch, int max_iteration) {

    const int r = 0;
    const int g = 0;
    
    for (int row = nrl; row <= nrh; row++) {
	rgb8* restrict dstline = rgbmat[row];
	const int32_t* srcline = it_mat[row];
	
	for (int col = ncl; col <= nch; col++) {
	    int32_t it = srcline[col];
	    int b = (it * 255 / max_iteration);
	    
	    rgb8 rgb;
	    rgb.r = 0;
	    rgb.g = 0;
	    rgb.b = b;
	    
	    
	    dstline[col] = rgb;
	}
    }
}

int main(int argc, char** argv) {

    int nrl = 0;
    int nrh = 8191;
    int ncl = 0;
    int nch = 8191;
    
    int32_t** it_mat = si32matrix(nrl, nrh, ncl, nch);
    rgb8** rgbmat = rgb8matrix(nrl, nrh, ncl, nch);
    
    const float MINX = -2.0;
    const float MAXX = 0.47;
    const float MINY = -1.12;
    const float MAXY = 1.12;
    const int MAX_ITERATIONS = 500;

    
    omp_set_num_threads(16);
    //set_ui8matrix(mat, nrl, nrh, ncl, nch, 255);

    struct timespec ts_start, ts_end;

    // "Warmup" + ensure physical allocation of arrays
    //mandelbrot_scalar(it_mat, nrl, nrh, ncl, nch, MINX, MINY, MAXX, MAXY, MAX_ITERATIONS);
    
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts_start);
    
    mandelbrot_parallel_scalar(it_mat, nrl, nrh, ncl, nch, MINX, MINY, MAXX, MAXY, MAX_ITERATIONS);
    //mandelbrot_avx512(it_mat, nrl, nrh, ncl, nch, MINX, MINY, MAXX, MAXY, MAX_ITERATIONS);        
    //mandelbrot_avx512_lu2(it_mat, nrl, nrh, ncl, nch, MINX, MINY, MAXX, MAXY, MAX_ITERATIONS);    
    //mandelbrot_avx512_lu2_pl(it_mat, nrl, nrh, ncl, nch, MINX, MINY, MAXX, MAXY, MAX_ITERATIONS);    
    //mandelbrot_avx512_lu4(it_mat, nrl, nrh, ncl, nch, MINX, MINY, MAXX, MAXY, MAX_ITERATIONS);
    //mandelbrot_avx512_lu4_pl(it_mat, nrl, nrh, ncl, nch, MINX, MINY, MAXX, MAXY, MAX_ITERATIONS);    
    //mandelbrot_parallel_avx512(it_mat, nrl, nrh, ncl, nch, MINX, MINY, MAXX, MAXY, MAX_ITERATIONS);
    //mandelbrot_parallel_avx512_lu4(it_mat, nrl, nrh, ncl, nch, MINX, MINY, MAXX, MAXY, MAX_ITERATIONS);
    //mandelbrot_cuda(it_mat, nrl, nrh, ncl, nch, MINX, MINY, MAXX, MAXY, MAX_ITERATIONS);
    //mandelbrot_cuda_v2(it_mat, nrl, nrh, ncl, nch, MINX, MINY, MAXX, MAXY, MAX_ITERATIONS);
    //mandelbrot_cuda_lu2(it_mat, nrl, nrh, ncl, nch, MINX, MINY, MAXX, MAXY, MAX_ITERATIONS);
    
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts_end);

    color_image(rgbmat, it_mat, nrl, nrh, ncl, nch, MAX_ITERATIONS);
    

    long pixel_count = (nrh - nrl + 1) * (nch - ncl + 1);
    
    float elapsed_ms = (ts_end.tv_sec - ts_start.tv_sec) * 1000 + (float)((ts_end.tv_nsec - ts_start.tv_nsec)) / 1000000;
    float pixel_per_s = pixel_count / (elapsed_ms / 1000.0);
    float gpixels = pixel_per_s / 1e9;
    
    printf("Elapsed ms = %f\n", elapsed_ms);
    printf("Bandwidth = %f GPix/s\n", gpixels);
    
    //SavePGM_ui8matrix(mat, nrl, nrh, ncl, nch, "mandelbrot.pgm");
    SavePPM_rgb8matrix(rgbmat, nrl, nrh, ncl, nch, "mandelbrot.ppm");

    
    free_si32matrix(it_mat, nrl, nrh, ncl, nch);
    free_rgb8matrix(rgbmat, nrl, nrh, ncl, nch);
}

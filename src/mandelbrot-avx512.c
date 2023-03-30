#include "mandelbrot-avx512.h"

#include <omp.h>
#include <immintrin.h>
#include "utils.h"


#ifdef __AVX512F__

void mandelbrot_avx512(int32_t** mat, int nrl, int nrh, int ncl, int nch,
		       float minx, float miny, float maxx, float maxy, int max_iterations) {

    const int width = nch - ncl + 1;
    const int height = nrh - nrl + 1;

    const float stepx = (maxx - minx) / width;
    const float stepy = (maxy - miny) / height;
    
    for (int row = nrl; row <= nrh; row++) {
        int32_t* restrict line = mat[row];

	const float y0 = miny + row * stepy;
	for (int col = ncl; col < nch; col += 16) {
	    const float x0 = minx + col * stepx;

	    __m512 vx0 = _mm512_set_ps(x0 + 15*stepx, x0 + 14*stepx, x0 + 13*stepx, x0 + 12*stepx,
				       x0 + 11*stepx, x0 + 10*stepx, x0 + 9 *stepx, x0 + 8*stepx,
				       x0 + 7 *stepx, x0 + 6 *stepx, x0 + 5 *stepx, x0 + 4*stepx,
				       x0 + 3 *stepx, x0 + 2 *stepx, x0 + 1 *stepx, x0);
	    __m512 vy0 = _mm512_set1_ps(y0);
	    
	    __m512 x = _mm512_set1_ps(0.0);
	    __m512 y = _mm512_set1_ps(0.0);
	    __m512 x2 = _mm512_set1_ps(0.0);
	    __m512 y2 = _mm512_set1_ps(0.0);
	    __m512i vit = _mm512_setzero_si512();
	    
	    const __m512 two = _mm512_set1_ps(2.0);
	    const __m512 four = _mm512_set1_ps(4.0);
	    const __m512i incr1i = _mm512_set1_epi32(1);
	    
	    for (int it = 0; it < max_iterations; it++) {
		
		// y = 2 * x * y + y0;
		y = _mm512_add_ps(_mm512_mul_ps(two, _mm512_mul_ps(x, y)), vy0); 

		// x = x2 - y2 + x0;
		x = _mm512_add_ps(_mm512_sub_ps(x2, y2), vx0);
		
		x2 = _mm512_mul_ps(x, x);
		y2 = _mm512_mul_ps(y, y);

		__m512 tot = _mm512_add_ps(x2, y2);
		
		__mmask16 mask = _mm512_cmple_ps_mask(tot, four);
		
		vit = _mm512_mask_add_epi32(vit, mask, vit, incr1i);

		if (!mask) {
		    break;
		}
	    }

	    _mm512_storeu_epi32(line + col, vit);
	}
    }    
}

void mandelbrot_avx512_v2(int32_t** mat, int nrl, int nrh, int ncl, int nch,
		       float minx, float miny, float maxx, float maxy, int max_iterations) {

    const int width = nch - ncl + 1;
    const int height = nrh - nrl + 1;

    const float stepx = (maxx - minx) / width;
    const float stepy = (maxy - miny) / height;

    const __m512 vstepx = _mm512_set1_ps(16 * stepx);
    const __m512 vminx = _mm512_set1_ps(minx);
    
    for (int row = nrl; row <= nrh; row++) {
        int32_t* restrict line = mat[row];

	const float y0 = miny + row * stepy;

	__m512 vx0 = _mm512_set_ps(15*stepx, 14*stepx, 13*stepx, 12*stepx, 11*stepx, 10*stepx,
				   9*stepx, 8*stepx, 7*stepx, 6*stepx, 5*stepx, 4*stepx,
				   3*stepx, 2*stepx, stepx, 0.0);
	vx0 = _mm512_add_ps(vx0, vminx);
	
	for (int col = ncl; col < nch; col += 16) {
	    //const float x0 = minx + col * stepx;

	    __m512 vy0 = _mm512_set1_ps(y0);
	    
	    __m512 x = _mm512_set1_ps(0.0);
	    __m512 y = _mm512_set1_ps(0.0);
	    __m512 x2 = _mm512_set1_ps(0.0);
	    __m512 y2 = _mm512_set1_ps(0.0);
	    __m512i vit = _mm512_setzero_si512();
	    
	    const __m512 two = _mm512_set1_ps(2.0);
	    const __m512 four = _mm512_set1_ps(4.0);
	    const __m512i incr1i = _mm512_set1_epi32(1);
	    
	    for (int it = 0; it < max_iterations; it++) {
		
		// y = 2 * x * y + y0;
		//y = _mm512_add_ps(_mm512_mul_ps(two, _mm512_mul_ps(x, y)), vy0); 
		y = _mm512_fmadd_ps(_mm512_mul_ps(x, y), two, vy0);
		
		// x = x2 - y2 + x0;
		x = _mm512_add_ps(_mm512_sub_ps(x2, y2), vx0);
		
		x2 = _mm512_mul_ps(x, x);
		y2 = _mm512_mul_ps(y, y);

		__m512 tot = _mm512_add_ps(x2, y2);
		
		__mmask16 mask = _mm512_cmple_ps_mask(tot, four);
		
		vit = _mm512_mask_add_epi32(vit, mask, vit, incr1i);

		if (!mask) {
		    break;
		}

	    }

	    _mm512_storeu_epi32(line + col, vit);
	    
	    vx0 = _mm512_add_ps(vx0, vstepx);
	}
    }    
}

void mandelbrot_avx512_lu2(int32_t** mat, int nrl, int nrh, int ncl, int nch,
			   float minx, float miny, float maxx, float maxy, int max_iterations) {

    const int width = nch - ncl + 1;
    const int height = nrh - nrl + 1;

    const float stepx = (maxx - minx) / width;
    const float stepy = (maxy - miny) / height;

    const __m512 vstepx = _mm512_set1_ps(32 * stepx);
    const __m512 vminx = _mm512_set1_ps(minx);
    const __m512 zerof = _mm512_set1_ps(0.0);
    
    for (int row = nrl; row <= nrh; row++) {
        int32_t* restrict line = mat[row];

	const float y0 = miny + row * stepy;

	__m512 vx0 = _mm512_set_ps(15*stepx, 14*stepx, 13*stepx, 12*stepx, 11*stepx, 10*stepx,
				   9*stepx, 8*stepx, 7*stepx, 6*stepx, 5*stepx, 4*stepx,
				   3*stepx, 2*stepx, stepx, 0.0);
	vx0 = _mm512_add_ps(vx0, vminx);
	__m512 vx1 = _mm512_add_ps(vx0, _mm512_set1_ps(16*stepx));
	
	for (int col = ncl; col < nch; col += 32) {
	    //const float x0 = minx + col * stepx;	    
	    
	    __m512 vy0 = _mm512_set1_ps(y0);
	    
	    __m512 x0 = zerof, x1 = zerof;
	    __m512 y0 = zerof, y1 = zerof;
	    __m512 xb0 = zerof, xb1 = zerof;
	    __m512 yb0 = zerof, yb1 = zerof;
	    __m512i vit0 = _mm512_setzero_si512();
	    __m512i vit1 = _mm512_setzero_si512();
	    
	    const __m512 two = _mm512_set1_ps(2.0);
	    const __m512 four = _mm512_set1_ps(4.0);
	    const __m512i incr1i = _mm512_set1_epi32(1);
	    
	    for (int it = 0; it < max_iterations; it++) {
		
		// y0 = 2 * x0 * y0 + y0;
		//y0 = _mm512_add_ps(_mm512_mul_ps(two, _mm512_mul_ps(x0, y0)), vy0); 
		y0 = _mm512_fmadd_ps(_mm512_mul_ps(x0, y0), two, vy0);
		y1 = _mm512_fmadd_ps(_mm512_mul_ps(x1, y1), two, vy0);
		
		// x0 = xb0 - yb0 + x0;
		x0 = _mm512_add_ps(_mm512_sub_ps(xb0, yb0), vx0);
		x1 = _mm512_add_ps(_mm512_sub_ps(xb1, yb1), vx1);
		
		xb0 = _mm512_mul_ps(x0, x0);
		xb1 = _mm512_mul_ps(x1, x1);
		yb0 = _mm512_mul_ps(y0, y0);
		yb1 = _mm512_mul_ps(y1, y1);

		__m512 tot0 = _mm512_add_ps(xb0, yb0);
		__m512 tot1 = _mm512_add_ps(xb1, yb1);
		
		__mmask16 mask0 = _mm512_cmple_ps_mask(tot0, four);
		__mmask16 mask1 = _mm512_cmple_ps_mask(tot1, four);
		
		vit0 = _mm512_mask_add_epi32(vit0, mask0, vit0, incr1i);
		vit1 = _mm512_mask_add_epi32(vit1, mask1, vit1, incr1i);
		
		if (!(mask0 | mask1)) {
		    break;
		}

	    }

	    _mm512_storeu_epi32(line + col, vit0);
	    _mm512_storeu_epi32(line + col + 16, vit1);
	    
	    vx0 = _mm512_add_ps(vx0, vstepx);
	    vx1 = _mm512_add_ps(vx1, vstepx);
	}
    }    
}

void mandelbrot_avx512_lu2_pl(int32_t** mat, int nrl, int nrh, int ncl, int nch,
			   float minx, float miny, float maxx, float maxy, int max_iterations) {

    const int width = nch - ncl + 1;
    const int height = nrh - nrl + 1;

    const float stepx = (maxx - minx) / width;
    const float stepy = (maxy - miny) / height;

    const __m512 vstepx = _mm512_set1_ps(16 * stepx);
    const __m512 vminx = _mm512_set1_ps(minx);
    const __m512 zerof = _mm512_set1_ps(0.0);
    
    for (int row = nrl; row <= nrh; row++) {
        int32_t* restrict line = mat[row];

	const float y = miny + row * stepy;

	__m512 vx0 = _mm512_set_ps(15*stepx, 14*stepx, 13*stepx, 12*stepx, 11*stepx, 10*stepx,
				   9*stepx, 8*stepx, 7*stepx, 6*stepx, 5*stepx, 4*stepx,
				   3*stepx, 2*stepx, stepx, 0.0);
	vx0 = _mm512_add_ps(vx0, vminx);
	__m512 vx1 = _mm512_add_ps(vx0, _mm512_set1_ps(16*stepx));
	
	    
	const __m512 vy0 = _mm512_set1_ps(y);
	    
	__m512 x0 = zerof, x1 = zerof;
	__m512 y0 = zerof, y1 = zerof;
	__m512 xb0 = zerof, xb1 = zerof;
	__m512 yb0 = zerof, yb1 = zerof;
	__m512i vit0 = _mm512_setzero_si512();
	__m512i vit1 = _mm512_setzero_si512();
	    
	const __m512 two = _mm512_set1_ps(2.0);
	const __m512 four = _mm512_set1_ps(4.0);
	const __m512i incr1i = _mm512_set1_epi32(1);

	int off0 = 0, off1 = 16;
	int it0 = 0, it1 = 0;
	    
	while (MAX(off0, off1) < width) {
	    y0 = _mm512_fmadd_ps(_mm512_mul_ps(x0, y0), two, vy0);
	    y1 = _mm512_fmadd_ps(_mm512_mul_ps(x1, y1), two, vy0);
		
	    // x0 = xb0 - yb0 + x0;
	    x0 = _mm512_add_ps(_mm512_sub_ps(xb0, yb0), vx0);
	    x1 = _mm512_add_ps(_mm512_sub_ps(xb1, yb1), vx1);
		
	    xb0 = _mm512_mul_ps(x0, x0);
	    xb1 = _mm512_mul_ps(x1, x1);
	    yb0 = _mm512_mul_ps(y0, y0);
	    yb1 = _mm512_mul_ps(y1, y1);

	    __m512 tot0 = _mm512_add_ps(xb0, yb0);
	    __m512 tot1 = _mm512_add_ps(xb1, yb1);
		
	    __mmask16 mask0 = _mm512_cmple_ps_mask(tot0, four);
	    __mmask16 mask1 = _mm512_cmple_ps_mask(tot1, four);
		
	    vit0 = _mm512_mask_add_epi32(vit0, mask0, vit0, incr1i);
	    vit1 = _mm512_mask_add_epi32(vit1, mask1, vit1, incr1i);
	    
		
	    it0++;
	    it1++;
		
	    if (!mask0 || it0 >= max_iterations) {
		_mm512_storeu_epi32(line + off0, vit0);
		off0 = MAX(off0, off1) + 16;		    
		
		__m512 vminx0 = _mm512_max_ps(vx0, vx1);
		vx0 = _mm512_add_ps(vminx0, vstepx);

		x0 = y0 = xb0 = yb0 = zerof;
		vit0 = _mm512_setzero_si512();
		it0 = 0;
	    }

	    if (!mask1 || it1 >= max_iterations) {
		_mm512_storeu_epi32(line + off1, vit1);
		off1 = MAX(off0, off1) + 16;
		    
		__m512 vminx1 = _mm512_max_ps(vx0, vx1);
		vx1 = _mm512_add_ps(vminx1, vstepx);

		x1 = y1 = xb1 = yb1 = zerof;
		vit1 = _mm512_setzero_si512();
		it1 = 0;
	    }
	}
	
    }    
}


void mandelbrot_avx512_lu4(int32_t** mat, int nrl, int nrh, int ncl, int nch,
			   float minx, float miny, float maxx, float maxy, int max_iterations) {

    const int width = nch - ncl + 1;
    const int height = nrh - nrl + 1;

    const float stepx = (maxx - minx) / width;
    const float stepy = (maxy - miny) / height;

    const __m512 vstepx = _mm512_set1_ps(64 * stepx);
    const __m512 vminx = _mm512_set1_ps(minx);
    const __m512 zerof = _mm512_set1_ps(0.0);
    
    for (int row = nrl; row <= nrh; row++) {
        int32_t* restrict line = mat[row];

	const float y0 = miny + row * stepy;

	__m512 vx0 = _mm512_set_ps(15*stepx, 14*stepx, 13*stepx, 12*stepx, 11*stepx, 10*stepx,
				   9*stepx, 8*stepx, 7*stepx, 6*stepx, 5*stepx, 4*stepx,
				   3*stepx, 2*stepx, stepx, 0.0);
	vx0 = _mm512_add_ps(vx0, vminx);
	__m512 vx1 = _mm512_add_ps(vx0, _mm512_set1_ps(16*stepx));
	__m512 vx2 = _mm512_add_ps(vx0, _mm512_set1_ps(32*stepx));
	__m512 vx3 = _mm512_add_ps(vx0, _mm512_set1_ps(48*stepx));

	
	for (int col = ncl; col < nch; col += 64) {
	    //const float x0 = minx + col * stepx;	    
	    
	    __m512 vy0 = _mm512_set1_ps(y0);

	    __m512 x0, x1, x2, x3;
	    __m512 y0, y1, y2, y3;
	    __m512 xb0, xb1, xb2, xb3;
	    __m512 yb0, yb1, yb2, yb3;
	    __m512i vit0, vit1, vit2, vit3;
	    
	    x0 = x1 = x2 = x3 = zerof;
	    y0 = y1 = y2 = y3 = zerof;
	    xb0 = xb1 = xb2 = xb3 = zerof;
	    yb0 = yb1 = yb2 = yb3 = zerof;
	    
	    vit0 = vit1 = vit2 = vit3 = _mm512_setzero_si512();
	    
	    const __m512 two = _mm512_set1_ps(2.0);
	    const __m512 four = _mm512_set1_ps(4.0);
	    const __m512i incr1i = _mm512_set1_epi32(1);
	    
	    for (int it = 0; it < max_iterations; it++) {
		
		// y0 = 2 * x0 * y0 + y0;
		//y0 = _mm512_add_ps(_mm512_mul_ps(two, _mm512_mul_ps(x0, y0)), vy0); 
		y0 = _mm512_fmadd_ps(_mm512_mul_ps(x0, y0), two, vy0);
		y1 = _mm512_fmadd_ps(_mm512_mul_ps(x1, y1), two, vy0);
		y2 = _mm512_fmadd_ps(_mm512_mul_ps(x2, y2), two, vy0);
		y3 = _mm512_fmadd_ps(_mm512_mul_ps(x3, y3), two, vy0);
		
		// x0 = xb0 - yb0 + x0;
		x0 = _mm512_add_ps(_mm512_sub_ps(xb0, yb0), vx0);
		x1 = _mm512_add_ps(_mm512_sub_ps(xb1, yb1), vx1);
		x2 = _mm512_add_ps(_mm512_sub_ps(xb2, yb2), vx2);
		x3 = _mm512_add_ps(_mm512_sub_ps(xb3, yb3), vx3);
		
		xb0 = _mm512_mul_ps(x0, x0);
		xb1 = _mm512_mul_ps(x1, x1);
		xb2 = _mm512_mul_ps(x2, x2);
		xb3 = _mm512_mul_ps(x3, x3);
		
		yb0 = _mm512_mul_ps(y0, y0);
		yb1 = _mm512_mul_ps(y1, y1);		
		yb2 = _mm512_mul_ps(y2, y2);
		yb3 = _mm512_mul_ps(y3, y3);

		__m512 tot0 = _mm512_add_ps(xb0, yb0);
		__m512 tot1 = _mm512_add_ps(xb1, yb1);
		__m512 tot2 = _mm512_add_ps(xb2, yb2);
		__m512 tot3 = _mm512_add_ps(xb3, yb3);
		
		__mmask16 mask0 = _mm512_cmple_ps_mask(tot0, four);
		__mmask16 mask1 = _mm512_cmple_ps_mask(tot1, four);
		__mmask16 mask2 = _mm512_cmple_ps_mask(tot2, four);
		__mmask16 mask3 = _mm512_cmple_ps_mask(tot3, four);
		
		vit0 = _mm512_mask_add_epi32(vit0, mask0, vit0, incr1i);
		vit1 = _mm512_mask_add_epi32(vit1, mask1, vit1, incr1i);
		vit2 = _mm512_mask_add_epi32(vit2, mask2, vit2, incr1i);
		vit3 = _mm512_mask_add_epi32(vit3, mask3, vit3, incr1i);
		
		if (!(mask0 | mask1 | mask2 | mask3)) {
		    break;
		}

	    }

	    _mm512_storeu_epi32(line + col, vit0);
	    _mm512_storeu_epi32(line + col + 16, vit1);
	    _mm512_storeu_epi32(line + col + 32, vit2);
	    _mm512_storeu_epi32(line + col + 48, vit3);
	    
	    vx0 = _mm512_add_ps(vx0, vstepx);
	    vx1 = _mm512_add_ps(vx1, vstepx);
	    vx2 = _mm512_add_ps(vx2, vstepx);
	    vx3 = _mm512_add_ps(vx3, vstepx);
	}
    }    
}


void mandelbrot_avx512_lu4_pl(int32_t** mat, int nrl, int nrh, int ncl, int nch,
			   float minx, float miny, float maxx, float maxy, int max_iterations) {

    const int width = nch - ncl + 1;
    const int height = nrh - nrl + 1;

    const float stepx = (maxx - minx) / width;
    const float stepy = (maxy - miny) / height;

    const __m512 vstepx = _mm512_set1_ps(16 * stepx);
    const __m512 vminx = _mm512_set1_ps(minx);
    const __m512 zerof = _mm512_set1_ps(0.0);
    
    for (int row = nrl; row <= nrh; row++) {
        int32_t* restrict line = mat[row];

	const float y = miny + row * stepy;

	__m512 vx0 = _mm512_set_ps(15*stepx, 14*stepx, 13*stepx, 12*stepx, 11*stepx, 10*stepx,
				   9*stepx, 8*stepx, 7*stepx, 6*stepx, 5*stepx, 4*stepx,
				   3*stepx, 2*stepx, stepx, 0.0);
	vx0 = _mm512_add_ps(vx0, vminx);
	__m512 vx1 = _mm512_add_ps(vx0, _mm512_set1_ps(16*stepx));
	__m512 vx2 = _mm512_add_ps(vx0, _mm512_set1_ps(32*stepx));
	__m512 vx3 = _mm512_add_ps(vx0, _mm512_set1_ps(48*stepx));
	
	    
	const __m512 vy0 = _mm512_set1_ps(y);

	__m512 x0, x1, x2, x3;
	__m512 y0, y1, y2, y3;
	__m512 xb0, xb1, xb2, xb3;
	__m512 yb0, yb1, yb2, yb3;
	__m512i vit0, vit1, vit2, vit3;
	    
	x0 = x1 = x2 = x3 = zerof;
	y0 = y1 = y2 = y3 = zerof;
	xb0 = xb1 = xb2 = xb3 = zerof;
	yb0 = yb1 = yb2 = yb3 = zerof;
	    
	vit0 = vit1 = vit2 = vit3 = _mm512_setzero_si512();
	    
	    
	const __m512 two = _mm512_set1_ps(2.0);
	const __m512 four = _mm512_set1_ps(4.0);
	const __m512i incr1i = _mm512_set1_epi32(1);

	int off0 = 0, off1 = 16, off2 = 32, off3 = 48;
	int it0 = 0, it1 = 0, it2 = 0, it3 = 0;

	int maxx_off = off3;	
	__m512 vmaxx = vx3;
	    
	while (maxx_off < width) {
	    y0 = _mm512_fmadd_ps(_mm512_mul_ps(x0, y0), two, vy0);
	    y1 = _mm512_fmadd_ps(_mm512_mul_ps(x1, y1), two, vy0);
	    y2 = _mm512_fmadd_ps(_mm512_mul_ps(x2, y2), two, vy0);
	    y3 = _mm512_fmadd_ps(_mm512_mul_ps(x3, y3), two, vy0);
		
	    // x0 = xb0 - yb0 + x0;
	    x0 = _mm512_add_ps(_mm512_sub_ps(xb0, yb0), vx0);
	    x1 = _mm512_add_ps(_mm512_sub_ps(xb1, yb1), vx1);
	    x2 = _mm512_add_ps(_mm512_sub_ps(xb2, yb2), vx2);
	    x3 = _mm512_add_ps(_mm512_sub_ps(xb3, yb3), vx3);
		
	    xb0 = _mm512_mul_ps(x0, x0);
	    xb1 = _mm512_mul_ps(x1, x1);
	    xb2 = _mm512_mul_ps(x2, x2);
	    xb3 = _mm512_mul_ps(x3, x3);
		
	    yb0 = _mm512_mul_ps(y0, y0);
	    yb1 = _mm512_mul_ps(y1, y1);		
	    yb2 = _mm512_mul_ps(y2, y2);
	    yb3 = _mm512_mul_ps(y3, y3);

	    __m512 tot0 = _mm512_add_ps(xb0, yb0);
	    __m512 tot1 = _mm512_add_ps(xb1, yb1);
	    __m512 tot2 = _mm512_add_ps(xb2, yb2);
	    __m512 tot3 = _mm512_add_ps(xb3, yb3);
		
	    __mmask16 mask0 = _mm512_cmple_ps_mask(tot0, four);
	    __mmask16 mask1 = _mm512_cmple_ps_mask(tot1, four);
	    __mmask16 mask2 = _mm512_cmple_ps_mask(tot2, four);
	    __mmask16 mask3 = _mm512_cmple_ps_mask(tot3, four);
		
	    vit0 = _mm512_mask_add_epi32(vit0, mask0, vit0, incr1i);
	    vit1 = _mm512_mask_add_epi32(vit1, mask1, vit1, incr1i);
	    vit2 = _mm512_mask_add_epi32(vit2, mask2, vit2, incr1i);
	    vit3 = _mm512_mask_add_epi32(vit3, mask3, vit3, incr1i);
	    
		
	    it0++;
	    it1++;
	    it2++;
	    it3++;

	    
	    if (!mask0 || it0 >= max_iterations) {
		_mm512_storeu_epi32(line + off0, vit0);
		
		maxx_off = off0 = maxx_off + 16;		
		vmaxx = vx0 = _mm512_add_ps(vmaxx, vstepx);

		x0 = y0 = xb0 = yb0 = zerof;
		vit0 = _mm512_setzero_si512();
		it0 = 0;
	    }

	    if (!mask1 || it1 >= max_iterations) {
		_mm512_storeu_epi32(line + off1, vit1);
		
		maxx_off = off1 = maxx_off + 16;
		vmaxx = vx1 = _mm512_add_ps(vmaxx, vstepx);

		x1 = y1 = xb1 = yb1 = zerof;
		vit1 = _mm512_setzero_si512();
		it1 = 0;
	    }

	    if (!mask2 || it2 >= max_iterations) {
		_mm512_storeu_epi32(line + off2, vit2);
		
		maxx_off = off2 = maxx_off + 16;		
		vmaxx = vx2 = _mm512_add_ps(vmaxx, vstepx);

		x2 = y2 = xb2 = yb2 = zerof;
		vit2 = _mm512_setzero_si512();
		it2 = 0;
	    }
	    
	    if (!mask3 || it3 >= max_iterations) {
		_mm512_storeu_epi32(line + off3, vit3);
		
		maxx_off = off3 = maxx_off + 16;		
		vmaxx = vx3 = _mm512_add_ps(vmaxx, vstepx);

		x3 = y3 = xb3 = yb3 = zerof;
		vit3 = _mm512_setzero_si512();
		it3 = 0;
	    }
	}
    }    
}


void mandelbrot_parallel_avx512(int32_t** mat, int nrl, int nrh, int ncl, int nch, float minx,
				float maxx, float miny, float maxy, int max_iterations) {
    
    const int width = nch - ncl + 1;
    const int height = nrh - nrl + 1;

    const float stepx = (maxx - minx) / width;
    const float stepy = (maxy - miny) / height;

    #pragma omp parallel for
    for (int row = nrl; row <= nrh; row++) {
        int32_t* restrict line = mat[row];

	const float y0 = miny + row * stepy;
	for (int col = ncl; col < nch; col += 16) {
	    const float x0 = minx + col * stepx;

	    __m512 vx0 = _mm512_set_ps(x0 + 15*stepx, x0 + 14*stepx, x0 + 13*stepx, x0 + 12*stepx,
				       x0 + 11*stepx, x0 + 10*stepx, x0 + 9 *stepx, x0 + 8*stepx,
				       x0 + 7 *stepx, x0 + 6 *stepx, x0 + 5 *stepx, x0 + 4*stepx,
				       x0 + 3 *stepx, x0 + 2 *stepx, x0 + 1 *stepx, x0);
	    __m512 vy0 = _mm512_set1_ps(y0);
	    
	    __m512 x = _mm512_set1_ps(0.0);
	    __m512 y = _mm512_set1_ps(0.0);
	    __m512 x2 = _mm512_set1_ps(0.0);
	    __m512 y2 = _mm512_set1_ps(0.0);
	    __m512i vit = _mm512_setzero_si512();
	    
	    const __m512 two = _mm512_set1_ps(2.0);
	    const __m512 four = _mm512_set1_ps(4.0);
	    const __m512i incr1i = _mm512_set1_epi32(1);
	    
	    for (int it = 0; it < max_iterations; it++) {
		
		// y = 2 * x * y + y0;
		y = _mm512_add_ps(_mm512_mul_ps(two, _mm512_mul_ps(x, y)), vy0); 

		// x = x2 - y2 + x0;
		x = _mm512_add_ps(_mm512_sub_ps(x2, y2), vx0);
		
		x2 = _mm512_mul_ps(x, x);
		y2 = _mm512_mul_ps(y, y);

		__m512 tot = _mm512_add_ps(x2, y2);
		
		__mmask16 mask = _mm512_cmple_ps_mask(tot, four);
		
		vit = _mm512_mask_add_epi32(vit, mask, vit, incr1i);

		if (!mask) {
		    break;
		}
	    }

	    _mm512_storeu_epi32(line + col, vit);
	}
    }    
    
}

void mandelbrot_parallel_avx512_lu4(int32_t** mat, int nrl, int nrh, int ncl, int nch,
				    float minx, float miny, float maxx, float maxy, int max_iterations) {

    const int width = nch - ncl + 1;
    const int height = nrh - nrl + 1;

    const float stepx = (maxx - minx) / width;
    const float stepy = (maxy - miny) / height;

    const __m512 vstepx = _mm512_set1_ps(64 * stepx);
    const __m512 vminx = _mm512_set1_ps(minx);
    const __m512 zerof = _mm512_set1_ps(0.0);

    #pragma omp parallel for
    for (int row = nrl; row <= nrh; row++) {
        int32_t* restrict line = mat[row];

	const float y0 = miny + row * stepy;

	__m512 vx0 = _mm512_set_ps(15*stepx, 14*stepx, 13*stepx, 12*stepx, 11*stepx, 10*stepx,
				   9*stepx, 8*stepx, 7*stepx, 6*stepx, 5*stepx, 4*stepx,
				   3*stepx, 2*stepx, stepx, 0.0);
	vx0 = _mm512_add_ps(vx0, vminx);
	__m512 vx1 = _mm512_add_ps(vx0, _mm512_set1_ps(16*stepx));
	__m512 vx2 = _mm512_add_ps(vx0, _mm512_set1_ps(32*stepx));
	__m512 vx3 = _mm512_add_ps(vx0, _mm512_set1_ps(48*stepx));

	
	for (int col = ncl; col < nch; col += 64) {
	    //const float x0 = minx + col * stepx;	    
	    
	    __m512 vy0 = _mm512_set1_ps(y0);

	    __m512 x0, x1, x2, x3;
	    __m512 y0, y1, y2, y3;
	    __m512 xb0, xb1, xb2, xb3;
	    __m512 yb0, yb1, yb2, yb3;
	    __m512i vit0, vit1, vit2, vit3;
	    
	    x0 = x1 = x2 = x3 = zerof;
	    y0 = y1 = y2 = y3 = zerof;
	    xb0 = xb1 = xb2 = xb3 = zerof;
	    yb0 = yb1 = yb2 = yb3 = zerof;
	    
	    vit0 = vit1 = vit2 = vit3 = _mm512_setzero_si512();
	    
	    const __m512 two = _mm512_set1_ps(2.0);
	    const __m512 four = _mm512_set1_ps(4.0);
	    const __m512i incr1i = _mm512_set1_epi32(1);
	    
	    for (int it = 0; it < max_iterations; it++) {
		
		// y0 = 2 * x0 * y0 + y0;
		//y0 = _mm512_add_ps(_mm512_mul_ps(two, _mm512_mul_ps(x0, y0)), vy0); 
		y0 = _mm512_fmadd_ps(_mm512_mul_ps(x0, y0), two, vy0);
		y1 = _mm512_fmadd_ps(_mm512_mul_ps(x1, y1), two, vy0);
		y2 = _mm512_fmadd_ps(_mm512_mul_ps(x2, y2), two, vy0);
		y3 = _mm512_fmadd_ps(_mm512_mul_ps(x3, y3), two, vy0);
		
		// x0 = xb0 - yb0 + x0;
		x0 = _mm512_add_ps(_mm512_sub_ps(xb0, yb0), vx0);
		x1 = _mm512_add_ps(_mm512_sub_ps(xb1, yb1), vx1);
		x2 = _mm512_add_ps(_mm512_sub_ps(xb2, yb2), vx2);
		x3 = _mm512_add_ps(_mm512_sub_ps(xb3, yb3), vx3);
		
		xb0 = _mm512_mul_ps(x0, x0);
		xb1 = _mm512_mul_ps(x1, x1);
		xb2 = _mm512_mul_ps(x2, x2);
		xb3 = _mm512_mul_ps(x3, x3);
		
		yb0 = _mm512_mul_ps(y0, y0);
		yb1 = _mm512_mul_ps(y1, y1);		
		yb2 = _mm512_mul_ps(y2, y2);
		yb3 = _mm512_mul_ps(y3, y3);

		__m512 tot0 = _mm512_add_ps(xb0, yb0);
		__m512 tot1 = _mm512_add_ps(xb1, yb1);
		__m512 tot2 = _mm512_add_ps(xb2, yb2);
		__m512 tot3 = _mm512_add_ps(xb3, yb3);
		
		__mmask16 mask0 = _mm512_cmple_ps_mask(tot0, four);
		__mmask16 mask1 = _mm512_cmple_ps_mask(tot1, four);
		__mmask16 mask2 = _mm512_cmple_ps_mask(tot2, four);
		__mmask16 mask3 = _mm512_cmple_ps_mask(tot3, four);
		
		vit0 = _mm512_mask_add_epi32(vit0, mask0, vit0, incr1i);
		vit1 = _mm512_mask_add_epi32(vit1, mask1, vit1, incr1i);
		vit2 = _mm512_mask_add_epi32(vit2, mask2, vit2, incr1i);
		vit3 = _mm512_mask_add_epi32(vit3, mask3, vit3, incr1i);
		
		if (!(mask0 | mask1 | mask2 | mask3)) {
		    break;
		}

	    }

	    _mm512_storeu_epi32(line + col, vit0);
	    _mm512_storeu_epi32(line + col + 16, vit1);
	    _mm512_storeu_epi32(line + col + 32, vit2);
	    _mm512_storeu_epi32(line + col + 48, vit3);
	    
	    vx0 = _mm512_add_ps(vx0, vstepx);
	    vx1 = _mm512_add_ps(vx1, vstepx);
	    vx2 = _mm512_add_ps(vx2, vstepx);
	    vx3 = _mm512_add_ps(vx3, vstepx);
	}
    }    
}


#endif // __AVX512F__



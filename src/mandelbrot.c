#include "mandelbrot.h"

void mandelbrot_scalar(int32_t** mat, int nrl, int nrh, int ncl, int nch,
		       float minx, float miny, float maxx, float maxy, int max_iterations) {

    int width = nch - ncl + 1;
    int height = nrh - nrl + 1;

    float stepx = (maxx - minx) / width;
    float stepy = (maxy - miny) / height;
    
    for (int row = nrl; row <= nrh; row++) {
        int32_t* restrict line = mat[row];

	const float y0 = miny + row * stepy;
	for (int col = ncl; col < nch; col++) {
	    const float x0 = minx + col * stepx;

	    int it = 0;

	    float x = 0, y =0;
	    float x2 = 0, y2 = 0;
	    for (it = 0; it < max_iterations; it++) {
		y = 2 * x * y + y0;
		x = x2 - y2 + x0;
		x2 = x * x;
		y2 = y * y;

	       		
		if (x2 + y2 > 2*2) {
		    break;
		}
	    }
 
	    line[col] = it;
	}
    }    
}

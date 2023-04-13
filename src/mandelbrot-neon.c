#ifdef __ARM_NEON

#include "mandelbrot-neon.h"

#include <arm_neon.h>
#include "utils.h"


void mandelbrot_neon(int32_t** mat, int nrl, int nrh, int ncl, int nch,
		     float minx, float miny, float maxx, float maxy, int max_iterations) {
    const int width = nch - ncl + 1;
    const int height = nrh - nrl + 1;


    const float stepx = (maxx - minx) / width;
    const float stepy = (maxy - miny) / height;

    const _Alignas(16) float Fdata[] = {0.0, 1.0, 2.0, 3.0};
    
    float32x4_t vstepx = vdupq_n_f32(stepx);
    const float32x4_t vj = vld1q_f32(Fdata);
    const float32x4_t vminx = vdupq_n_f32(minx);    
    
    const float32x4_t vincrx = vmulq_f32(vstepx, vj);
    
    for (int row = nrl; row <= nrh; row++) {
	int32_t* restrict line = mat[row];

	float32x4_t vx0 = vaddq_f32(vx0, vminx);
	
	const float y0 = miny + row * stepy;
	for (int col = ncl; col < nch; col += 16) {
	    const float x0 = minx + col * stepx;

	    float32x4_t vy0 = vdupq_n_f32(y0);
	    float32x4_t x = vdupq_n_f32(0.0);
	    float32x4_t y = vdupq_n_f32(0.0);
	    float32x4_t x2 = vdupq_n_f32(0.0);
	    float32x4_t y2 = vdupq_n_f32(0.0);
	    uint32x4_t vit = vdupq_n_u32(0);

	    const float32x4_t two = vdupq_n_f32(2.0);
	    const float32x4_t four = vdupq_n_f32(4.0);
	    const uint32x4_t incr1i = vdupq_n_u32(1);

	    for (int it = 0; it < max_iterations; it++) {

		// y = 2 * x * y + y0
		y = vaddq_f32(vmulq_f32(two, vmulq_f32(x, y)), vy0);

		// x = x2 - y2 + x0
		x = vaddq_f32(vsubq_f32(x2, y2), vx0);

		x2 = vmulq_f32(x, x); // x^2
		y2 = vmulq_f32(y, y); // y^2

		float32x4_t tot = vaddq_f32(x2, y2);
		uint32x4_t vmask = vcleq_f32(tot, four);

		unsigned cnt;
		cnt = vgetq_lane_u8(vcntq_u8(vreinterpretq_u8_u64(vmask)), 0); // not efficient
		
		vit = vaddq_u32(vit, vandq_u32(vmask, vdupq_n_u32(1)));
		if (!cnt) {
		    break;
		}
	    }
	    vst1q_s32(line + col, vreinterpretq_s32_u32(vit)); // Number of iterations should be positive and lower than 2**31
	    
	    vx0 = vaddq_f32(vx0, vstepx);
	}
	
    }
}

void mandelbrot_parallel_neon(int32_t** mat, int nrl, int nrh, int ncl, int nch,
			      float minx, float miny, float maxx, float maxy, int max_iterations) {
    
}

#endif // __ARM_NEON

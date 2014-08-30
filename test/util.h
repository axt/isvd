/* util.h
 *
 * Copyright (C) 2008,2009 Attila Axt
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or (at
 * your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
 */

#ifndef __ISVD_UTIL_H__
#define __ISVD_UTIL_H__

#include <stdio.h>
#include <sys/time.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_linalg.h>

/**
 * Simple code profiling macros. To enable profiling you must define
 * the PROFILE macro, before you include this header:
 * #define PROFILE
 */
#if defined PROFILE

#	define __PROFILE_START {\
			struct timeval tv1,tv2; \
			int sec; double msec;\
			gettimeofday(&tv1,NULL);
#	define __PROFILE_END gettimeofday(&tv2,NULL);\
				sec=(tv2.tv_sec-tv1.tv_sec );\
				msec=(tv2.tv_usec - tv1.tv_usec)/1000.0;\
                    		if(msec<0 && sec != 0) {sec--,msec += 1000;}\
				printf("__PROFILE__ [%d] %ds%fms\n",__LINE__,sec,msec);}

#	define __PROFILE__(x) __PROFILE_START x; __PROFILE_END
#else
#	define __PROFILE__(x)  x;
#	define __PROFILE_START
#	define __PROFILE_END
#endif


/**
 * Maximum displayed rows/columns by isvd_display_matrix & isvd_display_vector.
 */
#define DISPLAY_ROWS	6
#define DISPLAY_COLS	8



#undef __BEGIN_DECLS
#undef __END_DECLS
#ifdef __cplusplus
#	define __BEGIN_DECLS extern "C" {
#	define __END_DECLS }
#else
#	define __BEGIN_DECLS           /* empty */
#	define __END_DECLS             /* empty */
#endif

__BEGIN_DECLS

/* Initializes the matrix given as parameter with random values */
void isvd_init_random_matrix(gsl_matrix *A, unsigned long s);

/* Initializes the vector given as parameter with random values */
void isvd_init_random_vector(gsl_vector *A, unsigned long s);

/* Displays the matrix given as parameter on the screen */
void isvd_display_matrix(gsl_matrix *A, const char* name);

/* Displays the vector given as parameter on the screen */
void isvd_display_vector(gsl_vector *A, const char* name);

/* Calculate the reconstruction of matrix A = U*diag(S)*V' */
int isvd_SV_reconstruct(gsl_matrix *U, gsl_matrix *V, gsl_vector *S, gsl_matrix *A);

/* Calculate the L_2 operator-norm of the matrix given as parameter */
double isvd_matrix_l2norm(gsl_matrix *A);

__END_DECLS

#endif

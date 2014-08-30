/* isvd_float.h
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

/* Header file for isvd library using single precision
  
  The isvd library implements Incremental Singular Value Decomposition. The library is 
  intended to be used #TODO#
   
  The main source used to develop this library was:

	- M. Gu, Studies in numerical linear algebra, Ph.D. thesis, Yale University, New Haven, CT, USA (1993).

  This software is supposed to be an add-on module for the GNU Scientific Library
*/
#ifndef __ISVD_FLOAT_H__
#define __ISVD_FLOAT_H__


#include <gsl/gsl_matrix_float.h>
#include <gsl/gsl_vector_float.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_linalg.h>


#undef __BEGIN_DECLS
#undef __END_DECLS

#ifdef __cplusplus
# define __BEGIN_DECLS extern "C" {
# define __END_DECLS }
#else
# define __BEGIN_DECLS          /* empty */
# define __END_DECLS            /* empty */
#endif

/* TODO */
extern int gsl_linalg_SV_decomp_float (gsl_matrix_float * A, gsl_matrix_float * V, gsl_vector_float * S, gsl_vector_float * work);
extern int gsl_linalg_SV_decomp_mod_float (gsl_matrix_float * A, gsl_matrix_float * X, gsl_matrix_float * V, gsl_vector_float * S, gsl_vector_float * work);



__BEGIN_DECLS

typedef gsl_matrix_float gsl_matrix_float_pv;
typedef gsl_vector_float gsl_vector_float_pv;

/* TODO */
typedef struct 
{
	size_t N;					/* The size of the column vectors*/
	size_t M;   				/* Number of columns */

	gsl_matrix_float_pv *U1;	/*M   x M 	[temp1 subman 1 1 M M]*/
	gsl_matrix_float_pv *V1;	/*N   x M 	[temp2 submat 0 0 M M]*/
	gsl_vector_float *S1;		/*M*/

	gsl_matrix_float *U2;		/*N   x M+1*/
	gsl_matrix_float *V2;		/*M+1 x M+1*/
	gsl_vector_float *S2;		/*M+1*/

	gsl_matrix_float *L1;		/*M+1 x M+1*/
	gsl_matrix_float *L2;		/*M   x M*/

	gsl_matrix_float *X;		/*M   x M*/
	gsl_matrix_float *C;		/*M+1 x M*/
	gsl_matrix_float *LL;		/*M   x M+1*/

	gsl_matrix_float_pv *MM;	/*M   x M	[LL submat 0 0 M M] */
	gsl_vector_float_pv *z1t;	/*M		[L1 subrow 0 0 M]*/

	gsl_matrix_float *W;		/*M+1 x M+1*/
	gsl_matrix_float *D;		/*M+1 x M+1*/

	gsl_vector_float *work1;	/*M+1*/
	gsl_vector_float *work2;	/*M*/
	gsl_matrix_float *temp1;	/*M+1 x M+1*/
	gsl_matrix_float *temp2;	/*N   x M+1*/

	gsl_vector_float_pv *bb;	/*M			[temp2 col M]*/
	gsl_matrix_float_pv *V11;	/*M  x M	[V2 submat 0 0 M M]*/
	gsl_vector_float_pv *x;		/*M			[V2 subcol M]*/
	gsl_vector_float_pv *u1;	/*M			[V2 subrow M]*/
	gsl_vector_float_pv *l2;	/*M			[LL col M]*/
	gsl_vector_float_pv *d;		/*M+1		[D diag]*/

} isvd_float_workspace;

/* TODO */
isvd_float_workspace *isvd_float_alloc(const size_t N, const size_t M);

/* TODO */
void isvd_float_free(isvd_float_workspace *w);

/* TODO */
int isvd_float_update (isvd_float_workspace *w, gsl_vector_float *b);

/* TODO */
int isvd_float_downdate (isvd_float_workspace *w);

/* TODO */
int isvd_float_step (isvd_float_workspace *w, gsl_vector_float *b);

/* TODO */
int isvd_float_initialize(isvd_float_workspace *w, gsl_matrix_float *A);

__END_DECLS


#endif






/* isvd.h
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

/* Header file for isvd library using single precision calculation #TODO#
  
  The isvd library implements Incremental Singular Value Decomposition. The library is 
  intended to be used #TODO#
   
  The main source used to develop this library was:

	- M. Gu, Studies in numerical linear algebra, Ph.D. thesis, Yale University, New Haven, CT, USA (1993).

  This software is supposed to be an add-on module for the GNU Scientific Library
*/
#ifndef __ISVD_H__
#define __ISVD_H__


#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
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


__BEGIN_DECLS

typedef gsl_matrix gsl_matrix_pv;
typedef gsl_vector gsl_vector_pv;

/* TODO */
typedef struct 
{
	size_t N;				/* The size of the column vectors*/
	size_t M;   			/* Number of columns */

	gsl_matrix_pv *U1;	/*M   x M 	[temp1 subman 1 1 M M]*/
	gsl_matrix_pv *V1;	/*N   x M 	[temp2 submat 0 0 M M]*/
	gsl_vector *S1;		/*M*/

	gsl_matrix *U2;		/*N   x M+1*/
	gsl_matrix *V2;		/*M+1 x M+1*/
	gsl_vector *S2;		/*M+1*/

	gsl_matrix *L1;		/*+1 x M+1*/
	gsl_matrix *L2;		/*M   x M*/

	gsl_matrix *X;		/*M   x M*/
	gsl_matrix *C;		/*M+1 x M*/
	gsl_matrix *LL;		/*M   x M+1*/

	gsl_matrix_pv *MM;	/*M   x M	[LL submat 0 0 M M] */
	gsl_vector_pv *z1t;	/*M		[L1 subrow 0 0 M]*/

	gsl_matrix *W;		/*M+1 x M+1*/
	gsl_matrix *D;		/*M+1 x M+1*/

	gsl_vector *work1;	/*M+1*/
	gsl_vector *work2;	/*M*/
	gsl_matrix *temp1;	/*M+1 x M+1*/
	gsl_matrix *temp2;	/*N   x M+1*/

	gsl_vector_pv *bb;	/*M		[temp2 col M]*/
	gsl_matrix_pv *V11;	/*M  x M	[V2 submat 0 0 M M]*/
	gsl_vector_pv *x;	/*M		[V2 subcol M]*/
	gsl_vector_pv *u1;	/*M		[V2 subrow M]*/
	gsl_vector_pv *l2;	/*M		[LL col M]*/
	gsl_vector_pv *d;	/*M+1		[D diag]*/

} isvd_workspace;

/* TODO */
isvd_workspace *isvd_alloc(const size_t N, const size_t M);

/* TODO */
void isvd_free(isvd_workspace *w);

/* TODO */
int isvd_update (isvd_workspace *w, gsl_vector *b);

/* TODO */
int isvd_downdate (isvd_workspace *w);

/* TODO */
int isvd_step (isvd_workspace *w, gsl_vector *b);

/* TODO */
int isvd_initialize(isvd_workspace *w, gsl_matrix *A);

__END_DECLS


#endif






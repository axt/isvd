/* update_float.h
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

#include "../src/isvd_float.h"
#include "util_float.h"

#define N	(1000)
#define M	(20)

#define EPS 	(1e-6)

int main(int argc, char **argv)
{
	gsl_matrix_float *A, *A1, *A2, *V;
	gsl_matrix_float_view vA;
	gsl_vector_float_view vb;
	gsl_vector_float *s, *temp, *b;
	isvd_float_workspace *w;
	double dnorm, anorm;

	A = gsl_matrix_float_calloc(N,M+1);
	A1 = gsl_matrix_float_calloc(N,M+1);
	A2 = gsl_matrix_float_calloc(N,M+1);
	V = gsl_matrix_float_calloc(M+1,M+1);
	s = gsl_vector_float_calloc(M+1);
	temp = gsl_vector_float_calloc(M+1);
	b = gsl_vector_float_calloc(N);

	vA = gsl_matrix_float_submatrix(A,0,1,N,M);
	vb = gsl_matrix_float_column(A,0);

	isvd_float_init_random_matrix(A,0);

	w = isvd_float_alloc(N,M);
	isvd_float_initialize(w, &vA.matrix);

	gsl_vector_float_memcpy(b, &vb.vector);

	isvd_float_update(w, b);

	gsl_linalg_SV_decomp_float(A,V,s,temp);

	isvd_float_SV_reconstruct(A,V,s,A1);
	isvd_float_SV_reconstruct(w->U2,w->V2,w->S2,A2);



	anorm = isvd_float_matrix_l2norm(A1);
	gsl_matrix_float_sub(A1,A2);

	dnorm = isvd_float_matrix_l2norm(A1);

	isvd_float_free(w);

	gsl_matrix_float_free(A);
	gsl_matrix_float_free(V);
	gsl_vector_float_free(s);
	gsl_vector_float_free(temp);

	printf("Error norms: %g [%g/%g]\n", dnorm/anorm, dnorm, anorm);

	if( (dnorm/anorm) < EPS)
		return EXIT_SUCCESS;
	else
		return EXIT_FAILURE;
}


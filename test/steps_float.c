/* steps_float.c
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

#define N	(400)
#define M	(80)

#define EPS 	(1e-4)

int main(int argc, char **argv)
{
	size_t i;
	isvd_float_workspace *w;
	gsl_matrix_float *A1, *A2, *V, *R;
	gsl_vector_float *s, *temp, *b;
	double dnorm, anorm;
	
	A1 = gsl_matrix_float_calloc(N,M);
	A2 = gsl_matrix_float_calloc(N,M);
	R = gsl_matrix_float_calloc(N,M);
	V = gsl_matrix_float_calloc(M,M);
	s = gsl_vector_float_calloc(M);
	temp = gsl_vector_float_calloc(M);
	b = gsl_vector_float_calloc(N);

	isvd_float_init_random_matrix(A1,1);
	isvd_float_init_random_matrix(A2,2);

	w = isvd_float_alloc(N,M);
	isvd_float_initialize(w, A1);
	
	for(i=0; i < M; i++)
	{
		gsl_vector_float_view v = gsl_matrix_float_column(A2,M-i-1);
		gsl_vector_float_memcpy(b, &v.vector);
		isvd_float_step(w, b);
	}

	isvd_float_SV_reconstruct(w->U1,w->V1,w->S1, R);

	anorm = isvd_float_matrix_l2norm(A2);
	gsl_matrix_float_sub(A2,R);
	dnorm = isvd_float_matrix_l2norm(A2);

	isvd_float_free(w);

	gsl_matrix_float_free(A1);
	gsl_matrix_float_free(A2);
	gsl_matrix_float_free(R);
	gsl_matrix_float_free(V);
	gsl_vector_float_free(s);
	gsl_vector_float_free(b);
	gsl_vector_float_free(temp);

	printf("Error norms: %g [%g/%g]\n", dnorm/anorm, dnorm, anorm);

	if( (dnorm/anorm) < EPS)
		return EXIT_SUCCESS;
	else
		return EXIT_FAILURE;
}

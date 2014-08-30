/* steps.c
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

#include "../src/isvd.h"
#include "util.h"

#define N	(400)
#define M	(80)

#define EPS 	(1e-12)

int main(int argc, char **argv)
{
	size_t i;
	isvd_workspace *w;
	gsl_matrix *A1, *A2, *V, *R;
	gsl_vector *s, *temp, *b;
	double dnorm, anorm;
	
	A1 = gsl_matrix_calloc(N,M);
	A2 = gsl_matrix_calloc(N,M);
	R = gsl_matrix_calloc(N,M);
	V = gsl_matrix_calloc(M,M);
	s = gsl_vector_calloc(M);
	temp = gsl_vector_calloc(M);
	b = gsl_vector_calloc(N);

	isvd_init_random_matrix(A1,1);
	isvd_init_random_matrix(A2,2);

	w = isvd_alloc(N,M);
	isvd_initialize(w, A1);
	
	for(i=0; i < M; i++)
	{
		gsl_vector_view v = gsl_matrix_column(A2,M-i-1);
		gsl_vector_memcpy(b, &v.vector);
		isvd_step(w, b);
	}

	isvd_SV_reconstruct(w->U1,w->V1,w->S1, R);

	anorm = isvd_matrix_l2norm(A2);
	gsl_matrix_sub(A2,R);
	dnorm = isvd_matrix_l2norm(A2);

	isvd_free(w);

	gsl_matrix_free(A1);
	gsl_matrix_free(A2);
	gsl_matrix_free(R);
	gsl_matrix_free(V);
	gsl_vector_free(s);
	gsl_vector_free(b);
	gsl_vector_free(temp);

	printf("Error norms: %g [%g/%g]\n", dnorm/anorm, dnorm, anorm);

	if( (dnorm/anorm) < EPS)
		return EXIT_SUCCESS;
	else
		return EXIT_FAILURE;
}

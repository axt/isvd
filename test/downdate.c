/* downdate.c
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

#define N	(1000)
#define M	(20)

#define EPS 	(1e-15)

int main(int argc, char **argv)
{
	gsl_matrix *A, *A1, *A2, *V;
	gsl_matrix_view vA;
	gsl_vector *s, *temp;
	isvd_workspace *w;
	double dnorm, anorm;

	A = gsl_matrix_calloc(N,M+1);
	A1 = gsl_matrix_calloc(N,M);
	A2 = gsl_matrix_calloc(N,M);
	V = gsl_matrix_calloc(M,M);
	s = gsl_vector_calloc(M);
	temp = gsl_vector_calloc(M);

	vA = gsl_matrix_submatrix(A,0,0,N,M);


	isvd_init_random_matrix(A,0);

	w = isvd_alloc(N,M);

	gsl_matrix_memcpy(w->U2, A);
	gsl_linalg_SV_decomp(w->U2, w->V2, w->S2, w->work1);
	isvd_downdate(w);

	gsl_linalg_SV_decomp(&vA.matrix,V,s,temp);

	isvd_SV_reconstruct(&vA.matrix,V,s,A1);
	isvd_SV_reconstruct(w->U1,w->V1,w->S1,A2);

	anorm = isvd_matrix_l2norm(A1);
	gsl_matrix_sub(A1,A2);
	dnorm = isvd_matrix_l2norm(A1);

	isvd_free(w);

	gsl_matrix_free(A);
	gsl_matrix_free(V);
	gsl_vector_free(s);
	gsl_vector_free(temp);

	printf("Error norms: %g [%g/%g]\n", dnorm/anorm, dnorm, anorm);

	if( (dnorm/anorm) < EPS)
		return EXIT_SUCCESS;
	else
		return EXIT_FAILURE;
}


/* util.c
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

#include "util.h"

void isvd_init_random_matrix(gsl_matrix *A, unsigned long s)
{
	size_t i,j;
	const size_t rows    = A->size1;
	const size_t columns = A->size2;

	gsl_rng *r = gsl_rng_alloc(gsl_rng_taus);
	gsl_rng_set(r, s);
	for(i=0; i<rows; ++i) {
		for(j=0; j<columns; ++j) {
			double rnd = gsl_rng_uniform(r);
			gsl_matrix_set(A,i,j,rnd);
		}
	}

	gsl_rng_free(r);
}

void isvd_init_random_vector(gsl_vector *A, unsigned long s)
{
	size_t i;

	const size_t columns = A->size;	

	gsl_rng *r = gsl_rng_alloc(gsl_rng_taus);
	gsl_rng_set(r, s);
	for(i=0; i < columns; ++i) {
		double rnd = gsl_rng_uniform(r);
		gsl_vector_set(A,i,rnd);
	}

	gsl_rng_free(r);
}

void isvd_display_matrix(gsl_matrix *A, const char* name)
{
	size_t i,j;
	printf("%s = \n---\n", name);

	for(i=0; i < A->size1; ++i) {
		for(j=0; j < A->size2; ++j) {
			printf("%+.2g\t", gsl_matrix_get(A, i,j));
			if(j>DISPLAY_COLS) {
				printf("[%d columns not displayed]", (int) A->size2-DISPLAY_COLS);
				break;
			}
		}
		if(i>DISPLAY_ROWS) {
			printf("\n[%d rows not displayed]", (int)A->size1-DISPLAY_ROWS);
			break;
		}
		printf("\n");
	}
	printf("\n---\n");
}

void isvd_display_vector(gsl_vector *A, const char* name)
{
	size_t i;
	printf("%s = \n", name);

	for(i=0; i < A->size; ++i) {
		printf("%+.2g\t", gsl_vector_get(A, i));
		if(i>DISPLAY_COLS) {
			printf("[%d columns not displayed]", (int)A->size-DISPLAY_COLS);
			break;
		}
	}

	printf("\n\n");
}


int isvd_SV_reconstruct(gsl_matrix *U, gsl_matrix *V, gsl_vector *S, gsl_matrix *A)
{
	const size_t N = U->size1;
	const size_t M = U->size2;
	gsl_matrix *SS, *QQ;
	gsl_vector_view s;

	if (V->size1 != M)
		GSL_ERROR ("square matrix V must match second dimension of matrix U", GSL_EBADLEN);
	else if (V->size1 != V->size2)
		GSL_ERROR ("matrix V must be square", GSL_ENOTSQR);
	else if (S->size != M)
		GSL_ERROR ("length of vector S must match second dimension of matrix U", GSL_EBADLEN);
	else if (A->size1 != N)
		GSL_ERROR ("first dimension of matrix A must match first dimension of matrix U", GSL_EBADLEN);
	else if (A->size2 != M)
		GSL_ERROR ("second dimension of matrix A must match second dimension of matrix U", GSL_EBADLEN);

	SS = gsl_matrix_calloc(M,M);
	s = gsl_matrix_diagonal(SS);
	gsl_vector_memcpy(&s.vector, S);

	QQ = gsl_matrix_calloc(N,M);

	gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, U, SS, 0.0, QQ);
	gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, QQ, V, 0.0, A);

	gsl_matrix_free(QQ);

	return GSL_SUCCESS;
}



double isvd_matrix_l2norm(gsl_matrix *A) 
{
	size_t i;
	double d = 0.0;

	for (i=0; i < A->size1; i++)
	{
		gsl_vector_view row = gsl_matrix_row (A, i);
		d += gsl_blas_dnrm2 (&row.vector);
	}

	return d;
}

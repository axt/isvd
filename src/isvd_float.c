/* isvd.c
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

/* TODO */    
#include <gsl/gsl_errno.h>
#include <string.h>
#include "isvd_float.h"
#include "gsl_linalg_float/gsl_linalg_float.h"

gsl_vector_float* persistent_vector_view(gsl_vector_float_view vv) 
{
	gsl_vector_float *v;

	v = calloc(1, sizeof(gsl_vector_float));
	memcpy(v, &vv.vector, sizeof(gsl_vector_float));

	return v;
}

gsl_matrix_float* persistent_matrix_view(gsl_matrix_float_view mv) 
{
	gsl_matrix_float *m;

	m = calloc(1, sizeof(gsl_matrix_float));
	memcpy(m, &mv.matrix, sizeof(gsl_matrix_float));

	return m;
}


/* TODO
  param N ...
  param M ...
  return GSL_SUCCESS if successful
 */
isvd_float_workspace *isvd_float_alloc(const size_t N, const size_t M)
{
	isvd_float_workspace *w;

	if(M < 1) 
		GSL_ERROR_NULL("can not allocate workspace for M < 1 ", GSL_EINVAL);
	if(N < M) 
		GSL_ERROR_NULL("can not allocate workspace for N < M ", GSL_EINVAL);
	
	w = calloc(1, sizeof(isvd_float_workspace));

	if(!w)
		GSL_ERROR_NULL("failed to allocate memory for workspace", GSL_ENOMEM);

	w->M = M;
	w->N = N;

	w->U2 = gsl_matrix_float_calloc(N,M+1);
	if(!w->U2)
	{
		isvd_float_free(w);
		GSL_ERROR_NULL("failed to allocate memory for matrix U2", GSL_ENOMEM);
	}

	w->V2 = gsl_matrix_float_calloc(M+1,M+1);
	if(!w->V2)
	{
		isvd_float_free(w);
		GSL_ERROR_NULL("failed to allocate memory for matrix V1", GSL_ENOMEM);
	}

	w->S1 = gsl_vector_float_calloc(M);
	if(!w->S1)
	{
		isvd_float_free(w);
		GSL_ERROR_NULL("failed to allocate memory for vector S1", GSL_ENOMEM);
	}
	
	w->S2 = gsl_vector_float_calloc(M+1);
	if(!w->S2)
	{
		isvd_float_free(w);
		GSL_ERROR_NULL("failed to allocate memory for vector S2", GSL_ENOMEM);
	}


	w->L1 = gsl_matrix_float_calloc(M+1,M+1);
	if(!w->L1)
	{
		isvd_float_free(w);
		GSL_ERROR_NULL("failed to allocate memory for vector L1", GSL_ENOMEM);
	}
	w->L2 = gsl_matrix_float_calloc(M,M);
	if(!w->L2)
	{
		isvd_float_free(w);
		GSL_ERROR_NULL("failed to allocate memory for vector L2", GSL_ENOMEM);
	}

	w->X = gsl_matrix_float_calloc(M,M);
	if(!w->X)
	{
		isvd_float_free(w);
		GSL_ERROR_NULL("failed to allocate memory for vector X", GSL_ENOMEM);
	}
	w->C = gsl_matrix_float_calloc(M+1,M);
	if(!w->C)
	{
		isvd_float_free(w);
		GSL_ERROR_NULL("failed to allocate memory for vector C", GSL_ENOMEM);
	}


	w->LL = gsl_matrix_float_calloc(M,M+1);
	if(!w->LL)
	{
		isvd_float_free(w);
		GSL_ERROR_NULL("failed to allocate memory for vector LL", GSL_ENOMEM);
	}

	w->MM = persistent_matrix_view(gsl_matrix_float_submatrix(w->LL,0,0,M,M));

	w->z1t = persistent_vector_view(gsl_matrix_float_subrow (w->L1, 0, 0, M));

	w->W = gsl_matrix_float_calloc(M+1,M+1);
	if(!w->W)
	{
		isvd_float_free(w);
		GSL_ERROR_NULL("failed to allocate memory for vector W", GSL_ENOMEM);
	}

	w->D = gsl_matrix_float_calloc(M+1,M+1);
	if(!w->D)
	{
		isvd_float_free(w);
		GSL_ERROR_NULL("failed to allocate memory for vector D", GSL_ENOMEM);
	}

	gsl_matrix_float_set_all(w->D,0.0);

	w->work1 = gsl_vector_float_calloc(M+1);
	if(!w->work1)
	{
		isvd_float_free(w);
		GSL_ERROR_NULL("failed to allocate memory for vector work1", GSL_ENOMEM);
	}

	w->work2 = gsl_vector_float_calloc(M);
	if(!w->work2)
	{
		isvd_float_free(w);
		GSL_ERROR_NULL("failed to allocate memory for vector work1", GSL_ENOMEM);
	}

	w->temp1 = gsl_matrix_float_calloc(M+1,M+1);
	if(!w->temp1)
	{
		isvd_float_free(w);
		GSL_ERROR_NULL("failed to allocate memory for matrix temp1", GSL_ENOMEM);
	}
	
	w->temp2 = gsl_matrix_float_calloc(N,M+1);
	if(!w->temp2)
	{
		isvd_float_free(w);
		GSL_ERROR_NULL("failed to allocate memory for matrix temp2", GSL_ENOMEM);
	}

	gsl_matrix_float_set_all(w->temp1, 0.0);
	gsl_matrix_float_set(w->temp1,0,0,1.0);

	w->V1 = persistent_matrix_view(gsl_matrix_float_submatrix(w->temp1,1,1,M,M));
	w->U1 = persistent_matrix_view(gsl_matrix_float_submatrix(w->temp2,0,0,N,M));
	w->bb = persistent_vector_view(gsl_matrix_float_column(w->temp2,M));

	w->V11 = persistent_matrix_view(gsl_matrix_float_submatrix (w->V2, 0,0, M,M));
	w->x = persistent_vector_view(gsl_matrix_float_subcolumn (w->V2, M, 0, M));
	w->u1 = persistent_vector_view(gsl_matrix_float_subrow (w->V2, M, 0, M));

	w->l2 = persistent_vector_view(gsl_matrix_float_column(w->LL,M));
	w->d = persistent_vector_view(gsl_matrix_float_diagonal(w->D));

	return w;
}

void isvd_float_free(isvd_float_workspace *w)
{
	if(!w)
		GSL_ERROR_VOID("workspace is null", GSL_EINVAL);
	
	if(w->U1)
		free(w->U1);
	if(w->V1)
		free(w->V1);
	if(w->S1)
		gsl_vector_float_free(w->S1);
	if(w->U2)
		gsl_matrix_float_free(w->U2);
	if(w->V2)
		gsl_matrix_float_free(w->V2);
	if(w->S2)
		gsl_vector_float_free(w->S2);
	if(w->L1)
		gsl_matrix_float_free(w->L1);
	if(w->L2)
		gsl_matrix_float_free(w->L2);
	if(w->X)
		gsl_matrix_float_free(w->X);
	if(w->C)
		gsl_matrix_float_free(w->C);
	if(w->LL)
		gsl_matrix_float_free(w->LL);
	if(w->MM)
		free(w->MM);
	if(w->z1t)
		free(w->z1t);
	if(w->W)
		gsl_matrix_float_free(w->W);
	if(w->D)
		gsl_matrix_float_free(w->D);
	if(w->work1)
		gsl_vector_float_free(w->work1);
	if(w->work2)
		gsl_vector_float_free(w->work2);
	if(w->temp1)
		gsl_matrix_float_free(w->temp1);
	if(w->temp2)
		gsl_matrix_float_free(w->temp2);
	if(w->bb)
		free(w->bb);
	if(w->V11)
		free(w->V11);
	if(w->x)
		free(w->x);
	if(w->u1)
		free(w->u1);
	if(w->l2)
		free(w->l2);
	if(w->d)
		free(w->d);

	free(w);
}

int isvd_float_update(isvd_float_workspace *w, gsl_vector_float *b)
{
	size_t i;
	double eta;
	if(!w)
		GSL_ERROR("workspace must be allocated", GSL_EINVAL);

	if (w->N < w->M)
		GSL_ERROR("svd update of NxM matrix, N < M, is not implemented", GSL_EUNIMPL);

	if (b->size != w->N)
		GSL_ERROR("length of b is not conform with the size of the workspace", GSL_EBADLEN);

	gsl_matrix_float_set_all(w->L1, 0.0);

	gsl_blas_sgemv(CblasTrans, 1.0, w->U1, b, 0.0, w->z1t);
	gsl_blas_sgemv(CblasNoTrans, -1.0, w->U1, w->z1t, 1.0, b);

	for(i = 0; i < w->M; i++) 
		gsl_matrix_float_set(w->L1,i+1,i,gsl_vector_float_get(w->S1,i));

	eta = -1.0*gsl_blas_snrm2(b);
	gsl_matrix_float_set(w->L1,0,w->M,eta);

	gsl_blas_sscal (1.0/eta, b);

	gsl_linalg_SV_decomp_float(w->L1, w->W, w->S2, w->work1);
	gsl_blas_sgemm(CblasNoTrans, CblasNoTrans, 1.0, w->temp1, w->L1, 0.0, w->V2);

	gsl_vector_float_memcpy(w->bb, b);

	gsl_blas_sgemm(CblasNoTrans, CblasNoTrans, 1.0, w->temp2, w->W, 0.0, w->U2);
	return GSL_SUCCESS;
}

int isvd_float_downdate(isvd_float_workspace *w)
{
	double mu;
	double factor = 1.0;
	
	if(!w)
		GSL_ERROR("workspace must be allocated", GSL_EINVAL);

	if (w->N < w->M)
		GSL_ERROR("svd downdate of NxM matrix, N < M, is not implemented", GSL_EUNIMPL);

	mu = gsl_matrix_float_get(w->V2,w->M,w->M);

	if(mu < 0)
	{
		mu *= -1.0;
		factor = -1.0;
	}
	
	gsl_matrix_float_set_all(w->L2,0.0);
	gsl_blas_sger(1.0/(1.0+mu), w->u1, w->u1, w->L2);

	gsl_matrix_float_set_identity(w->MM);
	gsl_matrix_float_sub(w->MM, w->L2);

	gsl_matrix_float_set_all(w->L2, 0.0);
	gsl_blas_sgemm(CblasNoTrans, CblasNoTrans, factor, w->V11, w->MM, 0.0, w->X);
	gsl_blas_sger(1.0, w->x, w->u1, w->L2);
	gsl_matrix_float_sub(w->X,w->L2);

	gsl_vector_float_memcpy(w->l2, w->u1);
	gsl_vector_float_scale(w->l2, -1.0*factor);

	gsl_vector_float_memcpy(w->d, w->S2);
	gsl_blas_sgemm(CblasNoTrans, CblasTrans, 1.0, w->D, w->LL, 0.0, w->C);

	gsl_linalg_SV_decomp_float(w->C, w->L2, w->S1, w->work2);

	gsl_blas_sgemm(CblasNoTrans, CblasNoTrans, 1.0, w->X, w->L2, 0.0, w->V1);
	gsl_blas_sgemm(CblasNoTrans, CblasNoTrans, factor, w->U2, w->C, 0.0, w->U1);

	return GSL_SUCCESS;
}


int isvd_float_step (isvd_float_workspace *w, gsl_vector_float *b) 
{
	int ret;
	ret = isvd_float_update(w,b);

	if(ret == GSL_SUCCESS)
		ret = isvd_float_downdate(w);
	else
	    GSL_ERROR("error in update", ret);

	if(ret != GSL_SUCCESS)
		GSL_ERROR("error in downdate", ret);
	return ret;
}

int isvd_float_initialize(isvd_float_workspace *w, gsl_matrix_float *A) 
{
	int ret;
	if(!w)
		GSL_ERROR_NULL("workspace must be allocated", GSL_EINVAL);
	

	if(A)
	{
		if(w->N != A->size1 && w->M != A->size2)
			GSL_ERROR_NULL("workspace and matrix sizes are not compatible ", GSL_EINVAL);
		ret = gsl_matrix_float_memcpy(w->U1, A);
		if(ret != GSL_SUCCESS)
		    GSL_ERROR("error in matrix copy", ret);
	}
	return gsl_linalg_SV_decomp_mod_float(w->U1, w->X, w->V1, w->S1, w->work2);
}




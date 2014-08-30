/* linalg_float/svd.c
 * 
 * Copyright (C) 1996, 1997, 1998, 1999, 2000, 2004 Gerard Jungman, Brian Gough
 * 
 * Converting functions to use single precision by Attila Axt, 2009
 * 
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or (at
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



#include <stdlib.h>
#include <string.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_vector_float.h>
#include <gsl/gsl_matrix_float.h>
#include <gsl/gsl_blas.h>

#include <gsl/gsl_linalg.h>

#include "gsl_linalg_float.h"
#include "../isvd_float.h"

#include "givens.h"
#include "svdstep.h"

/* Factorise a general M x N matrix A into,
 *
 *   A = U D V^T
 *
 * where U is a column-orthogonal M x N matrix (U^T U = I), 
 * D is a diagonal N x N matrix, 
 * and V is an N x N orthogonal matrix (V^T V = V V^T = I)
 *
 * U is stored in the original matrix A, which has the same size
 *
 * V is stored as a separate matrix (not V^T). You must take the
 * transpose to form the product above.
 *
 * The diagonal matrix D is stored in the vector S,  D_ii = S_i
 */

int
gsl_linalg_SV_decomp_float (gsl_matrix_float * A, gsl_matrix_float * V, gsl_vector_float * S, 
                      gsl_vector_float * work)
{
  size_t a, b, i, j;

  const size_t M = A->size1;
  const size_t N = A->size2;
  const size_t K = GSL_MIN (M, N);

  if (M < N)
    {
      GSL_ERROR ("svd of MxN matrix, M<N, is not implemented", GSL_EUNIMPL);
    }
  else if (V->size1 != N)
    {
      GSL_ERROR ("square matrix V must match second dimension of matrix A",
                 GSL_EBADLEN);
    }
  else if (V->size1 != V->size2)
    {
      GSL_ERROR ("matrix V must be square", GSL_ENOTSQR);
    }
  else if (S->size != N)
    {
      GSL_ERROR ("length of vector S must match second dimension of matrix A",
                 GSL_EBADLEN);
    }
  else if (work->size != N)
    {
      GSL_ERROR ("length of workspace must match second dimension of matrix A",
                 GSL_EBADLEN);
    }

  /* Handle the case of N = 1 (SVD of a column vector) */

  if (N == 1)
    {
      gsl_vector_float_view column = gsl_matrix_float_column (A, 0);
      float norm = gsl_blas_snrm2 (&column.vector);

      gsl_vector_float_set (S, 0, norm); 
      gsl_matrix_float_set (V, 0, 0, 1.0);
      
      if (norm != 0.0)
        {
          gsl_blas_sscal (1.0/norm, &column.vector);
        }

      return GSL_SUCCESS;
    }
  
  {
    gsl_vector_float_view f = gsl_vector_float_subvector (work, 0, K - 1);
    
    /* bidiagonalize matrix A, unpack A into U S V */
    
    gsl_linalg_bidiag_decomp_float (A, S, &f.vector);
    gsl_linalg_bidiag_unpack2_float (A, S, &f.vector, V);

    /* apply reduction steps to B=(S,Sd) */
    
    chop_small_elements_float (S, &f.vector);
    
    /* Progressively reduce the matrix until it is diagonal */
    
    b = N - 1;
    
    while (b > 0)
      {
        float fbm1 = gsl_vector_float_get (&f.vector, b - 1);

        if (fbm1 == 0.0 || gsl_isnan (fbm1))
          {
            b--;
            continue;
          }
        
        /* Find the largest unreduced block (a,b) starting from b
           and working backwards */
        
        a = b - 1;
        
        while (a > 0)
          {
            float fam1 = gsl_vector_float_get (&f.vector, a - 1);

            if (fam1 == 0.0 || gsl_isnan (fam1))
              {
                break;
              }
            
            a--;
          }
        
        {
          const size_t n_block = b - a + 1;
          gsl_vector_float_view S_block = gsl_vector_float_subvector (S, a, n_block);
          gsl_vector_float_view f_block = gsl_vector_float_subvector (&f.vector, a, n_block - 1);
          
          gsl_matrix_float_view U_block =
            gsl_matrix_float_submatrix (A, 0, a, A->size1, n_block);
          gsl_matrix_float_view V_block =
            gsl_matrix_float_submatrix (V, 0, a, V->size1, n_block);
          
          qrstep_float (&S_block.vector, &f_block.vector, &U_block.matrix, &V_block.matrix);
          
          /* remove any small off-diagonal elements */
          
          chop_small_elements_float (&S_block.vector, &f_block.vector);
        }
      }
  }
  /* Make singular values positive by reflections if necessary */
  
  for (j = 0; j < K; j++)
    {
      float Sj = gsl_vector_float_get (S, j);
      
      if (Sj < 0.0)
        {
          for (i = 0; i < N; i++)
            {
              float Vij = gsl_matrix_float_get (V, i, j);
              gsl_matrix_float_set (V, i, j, -Vij);
            }
          
          gsl_vector_float_set (S, j, -Sj);
        }
    }
  
  /* Sort singular values into decreasing order */
  
  for (i = 0; i < K; i++)
    {
      float S_max = gsl_vector_float_get (S, i);
      size_t i_max = i;
      
      for (j = i + 1; j < K; j++)
        {
          float Sj = gsl_vector_float_get (S, j);
          
          if (Sj > S_max)
            {
              S_max = Sj;
              i_max = j;
            }
        }
      
      if (i_max != i)
        {
          /* swap eigenvalues */
          gsl_vector_float_swap_elements (S, i, i_max);
          
          /* swap eigenvectors */
          gsl_matrix_float_swap_columns (A, i, i_max);
          gsl_matrix_float_swap_columns (V, i, i_max);
        }
    }
  
  return GSL_SUCCESS;
}

/* Modified algorithm which is better for M>>N */

int
gsl_linalg_SV_decomp_mod_float (gsl_matrix_float * A,
                          gsl_matrix_float * X,
                          gsl_matrix_float * V, gsl_vector_float * S, gsl_vector_float * work)
{
  size_t i, j;

  const size_t M = A->size1;
  const size_t N = A->size2;

  if (M < N)
    {
      GSL_ERROR ("svd of MxN matrix, M<N, is not implemented", GSL_EUNIMPL);
    }
  else if (V->size1 != N)
    {
      GSL_ERROR ("square matrix V must match second dimension of matrix A",
                 GSL_EBADLEN);
    }
  else if (V->size1 != V->size2)
    {
      GSL_ERROR ("matrix V must be square", GSL_ENOTSQR);
    }
  else if (X->size1 != N)
    {
      GSL_ERROR ("square matrix X must match second dimension of matrix A",
                 GSL_EBADLEN);
    }
  else if (X->size1 != X->size2)
    {
      GSL_ERROR ("matrix X must be square", GSL_ENOTSQR);
    }
  else if (S->size != N)
    {
      GSL_ERROR ("length of vector S must match second dimension of matrix A",
                 GSL_EBADLEN);
    }
  else if (work->size != N)
    {
      GSL_ERROR ("length of workspace must match second dimension of matrix A",
                 GSL_EBADLEN);
    }

  if (N == 1)
    {
      gsl_vector_float_view column = gsl_matrix_float_column (A, 0);
      float norm = gsl_blas_snrm2 (&column.vector);

      gsl_vector_float_set (S, 0, norm); 
      gsl_matrix_float_set (V, 0, 0, 1.0);
      
      if (norm != 0.0)
        {
          gsl_blas_sscal (1.0/norm, &column.vector);
        }

      return GSL_SUCCESS;
    }

  /* Convert A into an upper triangular matrix R */

  for (i = 0; i < N; i++)
    {
      gsl_vector_float_view c = gsl_matrix_float_column (A, i);
      gsl_vector_float_view v = gsl_vector_float_subvector (&c.vector, i, M - i);
      float tau_i = gsl_linalg_householder_transform_float (&v.vector);

      /* Apply the transformation to the remaining columns */

      if (i + 1 < N)
        {
          gsl_matrix_float_view m =
            gsl_matrix_float_submatrix (A, i, i + 1, M - i, N - (i + 1));
          gsl_linalg_householder_hm_float (tau_i, &v.vector, &m.matrix);
        }

      gsl_vector_float_set (S, i, tau_i);
    }

  /* Copy the upper triangular part of A into X */

  for (i = 0; i < N; i++)
    {
      for (j = 0; j < i; j++)
        {
          gsl_matrix_float_set (X, i, j, 0.0);
        }

      {
        float Aii = gsl_matrix_float_get (A, i, i);
        gsl_matrix_float_set (X, i, i, Aii);
      }

      for (j = i + 1; j < N; j++)
        {
          float Aij = gsl_matrix_float_get (A, i, j);
          gsl_matrix_float_set (X, i, j, Aij);
        }
    }

  /* Convert A into an orthogonal matrix L */

  for (j = N; j > 0 && j--;)
    {
      /* Householder column transformation to accumulate L */
      float tj = gsl_vector_float_get (S, j);
      gsl_matrix_float_view m = gsl_matrix_float_submatrix (A, j, j, M - j, N - j);
      gsl_linalg_householder_hm1_float (tj, &m.matrix);
    }

  /* unpack R into X V S */

  gsl_linalg_SV_decomp_float (X, V, S, work);

  /* Multiply L by X, to obtain U = L X, stored in U */

  {
    gsl_vector_float_view sum = gsl_vector_float_subvector (work, 0, N);

    for (i = 0; i < M; i++)
      {
        gsl_vector_float_view L_i = gsl_matrix_float_row (A, i);
        gsl_vector_float_set_zero (&sum.vector);

        for (j = 0; j < N; j++)
          {
            float Lij = gsl_vector_float_get (&L_i.vector, j);
            gsl_vector_float_view X_j = gsl_matrix_float_row (X, j);
            gsl_blas_saxpy (Lij, &X_j.vector, &sum.vector);
          }

        gsl_vector_float_memcpy (&L_i.vector, &sum.vector);
      }
  }

  return GSL_SUCCESS;
}



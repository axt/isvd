/* linalg/householder.c
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
#include <gsl/gsl_math.h>
#include <gsl/gsl_vector_float.h>
#include <gsl/gsl_matrix_float.h>
#include <gsl/gsl_blas.h>

#include "../isvd_float.h"

float
gsl_linalg_householder_transform_float (gsl_vector_float * v)
{
  /* replace v[0:n-1] with a householder vector (v[0:n-1]) and
     coefficient tau that annihilate v[1:n-1] */

  const size_t n = v->size ;

  if (n == 1)
    {
      return 0.0; /* tau = 0 */
    }
  else
    { 
      float alpha, beta, tau ;
      
      gsl_vector_float_view x = gsl_vector_float_subvector (v, 1, n - 1) ; 
      
      float xnorm = gsl_blas_snrm2 (&x.vector);

      if (xnorm == 0) 
        {
          return 0.0; /* tau = 0 */
        }
      
      alpha = gsl_vector_float_get (v, 0) ;
      beta = - (alpha >= 0.0 ? +1.0 : -1.0) * hypot(alpha, xnorm) ;
      tau = (beta - alpha) / beta ;

      gsl_blas_sscal (1.0 / (alpha - beta), &x.vector);
      gsl_vector_float_set (v, 0, beta) ;
      
      return tau;
    }
}

int
gsl_linalg_householder_hm_float (float tau, const gsl_vector_float * v, gsl_matrix_float * A)
{
  /* applies a householder transformation v,tau to matrix m */


  if (tau == 0.0)
    {
      return GSL_SUCCESS;
    }

#ifdef USE_BLAS
  {
    gsl_vector_float_const_view v1 = gsl_vector_float_const_subvector (v, 1, v->size - 1);
    gsl_matrix_float_view A1 = gsl_matrix_float_submatrix (A, 1, 0, A->size1 - 1, A->size2);
    size_t j;

    for (j = 0; j < A->size2; j++)
      {
        float wj = 0.0;
        gsl_vector_float_view A1j = gsl_matrix_float_column(&A1.matrix, j);
        gsl_blas_sdot (&A1j.vector, &v1.vector, &wj);
        wj += gsl_matrix_float_get(A,0,j);

        {
          float A0j = gsl_matrix_float_get (A, 0, j);
          gsl_matrix_float_set (A, 0, j, A0j - tau *  wj);
        }

        gsl_blas_saxpy (-tau * wj, &v1.vector, &A1j.vector);
      }
  }
#else
  {
    size_t i, j;
    
    for (j = 0; j < A->size2; j++)
      {
        /* Compute wj = Akj vk */
        
        float wj = gsl_matrix_float_get(A,0,j);  
        
        for (i = 1; i < A->size1; i++)  /* note, computed for v(0) = 1 above */
          {
            wj += gsl_matrix_float_get(A,i,j) * gsl_vector_float_get(v,i);
          }
        
        /* Aij = Aij - tau vi wj */
        
        /* i = 0 */
        {
          float A0j = gsl_matrix_float_get (A, 0, j);
          gsl_matrix_float_set (A, 0, j, A0j - tau *  wj);
        }
        
        /* i = 1 .. M-1 */
        
        for (i = 1; i < A->size1; i++)
          {
            float Aij = gsl_matrix_float_get (A, i, j);
            float vi = gsl_vector_float_get (v, i);
            gsl_matrix_float_set (A, i, j, Aij - tau * vi * wj);
          }
      }
  }
#endif
    
  return GSL_SUCCESS;
}

int
gsl_linalg_householder_mh_float (float tau, const gsl_vector_float * v, gsl_matrix_float * A)
{
  /* applies a householder transformation v,tau to matrix m from the
     right hand side in order to zero out rows */

  if (tau == 0)
    return GSL_SUCCESS;

  /* A = A - tau w v' */

#ifdef USE_BLAS
  {
    gsl_vector_float_const_view v1 = gsl_vector_float_const_subvector (v, 1, v->size - 1);
    gsl_matrix_float_view A1 = gsl_matrix_float_submatrix (A, 0, 1, A->size1, A->size2-1);
    size_t i;

    for (i = 0; i < A->size1; i++)
      {
        float wi = 0.0;
        gsl_vector_float_view A1i = gsl_matrix_float_row(&A1.matrix, i);
        gsl_blas_sdot (&A1i.vector, &v1.vector, &wi);
        wi += gsl_matrix_float_get(A,i,0);  
        
        {
          float Ai0 = gsl_matrix_float_get (A, i, 0);
          gsl_matrix_float_set (A, i, 0, Ai0 - tau *  wi);
        }
        
        gsl_blas_saxpy(-tau * wi, &v1.vector, &A1i.vector);
      }
  }
#else
  {
    size_t i, j;
    
    for (i = 0; i < A->size1; i++)
      {
        float wi = gsl_matrix_float_get(A,i,0);  
        
        for (j = 1; j < A->size2; j++)  /* note, computed for v(0) = 1 above */
          {
            wi += gsl_matrix_float_get(A,i,j) * gsl_vector_float_get(v,j);
          }
        
        /* j = 0 */
        
        {
          float Ai0 = gsl_matrix_float_get (A, i, 0);
          gsl_matrix_float_set (A, i, 0, Ai0 - tau *  wi);
        }
        
        /* j = 1 .. N-1 */
        
        for (j = 1; j < A->size2; j++) 
          {
            float vj = gsl_vector_float_get (v, j);
            float Aij = gsl_matrix_float_get (A, i, j);
            gsl_matrix_float_set (A, i, j, Aij - tau * wi * vj);
          }
      }
  }
#endif
    
  return GSL_SUCCESS;
}

int
gsl_linalg_householder_hv_float (float tau, const gsl_vector_float * v, gsl_vector_float * w)
{
  /* applies a householder transformation v to vector w */
  const size_t N = v->size;
 
  if (tau == 0)
    return GSL_SUCCESS ;

  {
    /* compute d = v'w */

    float d0 = gsl_vector_float_get(w,0);
    float d1, d;

    gsl_vector_float_const_view v1 = gsl_vector_float_const_subvector(v, 1, N-1);
    gsl_vector_float_view w1 = gsl_vector_float_subvector(w, 1, N-1);

    gsl_blas_sdot (&v1.vector, &w1.vector, &d1);
    
    d = d0 + d1;

    /* compute w = w - tau (v) (v'w) */
  
    {
      float w0 = gsl_vector_float_get (w,0);
      gsl_vector_float_set (w, 0, w0 - tau * d);
    }
    
    gsl_blas_saxpy (-tau * d, &v1.vector, &w1.vector);
  }
  
  return GSL_SUCCESS;
}


int
gsl_linalg_householder_hm1_float (float tau, gsl_matrix_float * A)
{
  /* applies a householder transformation v,tau to a matrix being
     build up from the identity matrix, using the first column of A as
     a householder vector */

  if (tau == 0)
    {
      size_t i,j;

      gsl_matrix_float_set (A, 0, 0, 1.0);
      
      for (j = 1; j < A->size2; j++)
        {
          gsl_matrix_float_set (A, 0, j, 0.0);
        }

      for (i = 1; i < A->size1; i++)
        {
          gsl_matrix_float_set (A, i, 0, 0.0);
        }

      return GSL_SUCCESS;
    }

  /* w = A' v */

#ifdef USE_BLAS
  {
    gsl_matrix_float_view A1 = gsl_matrix_float_submatrix (A, 1, 0, A->size1 - 1, A->size2);
    gsl_vector_float_view v1 = gsl_matrix_float_column (&A1.matrix, 0);
    size_t j;

    for (j = 1; j < A->size2; j++)
      {
        float wj = 0.0;   /* A0j * v0 */
        
        gsl_vector_float_view A1j = gsl_matrix_float_column(&A1.matrix, j);
        gsl_blas_sdot (&A1j.vector, &v1.vector, &wj);

        /* A = A - tau v w' */
        
        gsl_matrix_float_set (A, 0, j, - tau *  wj);
        
        gsl_blas_saxpy(-tau*wj, &v1.vector, &A1j.vector);
      }

    gsl_blas_sscal(-tau, &v1.vector);
    
    gsl_matrix_float_set (A, 0, 0, 1.0 - tau);
  }
#else
  {
    size_t i, j;
    
    for (j = 1; j < A->size2; j++)
      {
        float wj = 0.0;   /* A0j * v0 */
        
        for (i = 1; i < A->size1; i++)
          {
            float vi = gsl_matrix_float_get(A, i, 0);
            wj += gsl_matrix_float_get(A,i,j) * vi;
          }
        
        /* A = A - tau v w' */
        
        gsl_matrix_float_set (A, 0, j, - tau *  wj);
        
        for (i = 1; i < A->size1; i++)
          {
            float vi = gsl_matrix_float_get (A, i, 0);
            float Aij = gsl_matrix_float_get (A, i, j);
            gsl_matrix_float_set (A, i, j, Aij - tau * vi * wj);
          }
      }
    
    for (i = 1; i < A->size1; i++)
      {
        float vi = gsl_matrix_float_get(A, i, 0);
        gsl_matrix_float_set(A, i, 0, -tau * vi);
      }
    
    gsl_matrix_float_set (A, 0, 0, 1.0 - tau);
  }
#endif

  return GSL_SUCCESS;
}


static void
chop_small_elements_float (gsl_vector_float * d, gsl_vector_float * f)
{
  const size_t N = d->size;
  float d_i = gsl_vector_float_get (d, 0);

  size_t i;

  for (i = 0; i < N - 1; i++)
    {
      float f_i = gsl_vector_float_get (f, i);
      float d_ip1 = gsl_vector_float_get (d, i + 1);

      if (fabs (f_i) < GSL_DBL_EPSILON * (fabs (d_i) + fabs (d_ip1)))
        {
          gsl_vector_float_set (f, i, 0.0);
        }
      d_i = d_ip1;
    }

}

static float
trailing_eigenvalue_float (const gsl_vector_float * d, const gsl_vector_float * f)
{
  const size_t n = d->size;

  float da = gsl_vector_float_get (d, n - 2);
  float db = gsl_vector_float_get (d, n - 1);
  float fa = (n > 2) ? gsl_vector_float_get (f, n - 3) : 0.0;
  float fb = gsl_vector_float_get (f, n - 2);

  float ta = da * da + fa * fa;
  float tb = db * db + fb * fb;
  float tab = da * fb;

  float dt = (ta - tb) / 2.0;

  float mu;

  if (dt >= 0)
    {
      mu = tb - (tab * tab) / (dt + hypot (dt, tab));
    }
  else 
    {
      mu = tb + (tab * tab) / ((-dt) + hypot (dt, tab));
    }

  return mu;
}

static void
create_schur_float (float d0, float f0, float d1, float * c, float * s)
{
  float apq = 2.0 * d0 * f0;
  
  if (apq != 0.0)
    {
      float t;
      float tau = (f0*f0 + (d1 + d0)*(d1 - d0)) / apq;
      
      if (tau >= 0.0)
        {
          t = 1.0/(tau + hypot(1.0, tau));
        }
      else
        {
          t = -1.0/(-tau + hypot(1.0, tau));
        }

      *c = 1.0 / hypot(1.0, t);
      *s = t * (*c);
    }
  else
    {
      *c = 1.0;
      *s = 0.0;
    }
}

static void
svd2_float (gsl_vector_float * d, gsl_vector_float * f, gsl_matrix_float * U, gsl_matrix_float * V)
{
  size_t i;
  float c, s, a11, a12, a21, a22;

  const size_t M = U->size1;
  const size_t N = V->size1;

  float d0 = gsl_vector_float_get (d, 0);
  float f0 = gsl_vector_float_get (f, 0);
  
  float d1 = gsl_vector_float_get (d, 1);

  if (d0 == 0.0)
    {
      /* Eliminate off-diagonal element in [0,f0;0,d1] to make [d,0;0,0] */

      create_givens_float (f0, d1, &c, &s);

      /* compute B <= G^T B X,  where X = [0,1;1,0] */

      gsl_vector_float_set (d, 0, c * f0 - s * d1);
      gsl_vector_float_set (f, 0, s * f0 + c * d1);
      gsl_vector_float_set (d, 1, 0.0);
      
      /* Compute U <= U G */

      for (i = 0; i < M; i++)
        {
          float Uip = gsl_matrix_float_get (U, i, 0);
          float Uiq = gsl_matrix_float_get (U, i, 1);
          gsl_matrix_float_set (U, i, 0, c * Uip - s * Uiq);
          gsl_matrix_float_set (U, i, 1, s * Uip + c * Uiq);
        }

      /* Compute V <= V X */

      gsl_matrix_float_swap_columns (V, 0, 1);

      return;
    }
  else if (d1 == 0.0)
    {
      /* Eliminate off-diagonal element in [d0,f0;0,0] */

      create_givens_float (d0, f0, &c, &s);

      /* compute B <= B G */

      gsl_vector_float_set (d, 0, d0 * c - f0 * s);
      gsl_vector_float_set (f, 0, 0.0);

      /* Compute V <= V G */

      for (i = 0; i < N; i++)
        {
          float Vip = gsl_matrix_float_get (V, i, 0);
          float Viq = gsl_matrix_float_get (V, i, 1);
          gsl_matrix_float_set (V, i, 0, c * Vip - s * Viq);
          gsl_matrix_float_set (V, i, 1, s * Vip + c * Viq);
        }

      return;
    }
  else
    {
      /* Make columns orthogonal, A = [d0, f0; 0, d1] * G */
      
      create_schur_float (d0, f0, d1, &c, &s);
      
      /* compute B <= B G */
      
      a11 = c * d0 - s * f0;
      a21 = - s * d1;
      
      a12 = s * d0 + c * f0;
      a22 = c * d1;
      
      /* Compute V <= V G */
      
      for (i = 0; i < N; i++)
        {
          float Vip = gsl_matrix_float_get (V, i, 0);
          float Viq = gsl_matrix_float_get (V, i, 1);
          gsl_matrix_float_set (V, i, 0, c * Vip - s * Viq);
          gsl_matrix_float_set (V, i, 1, s * Vip + c * Viq);
        }
      
      /* Eliminate off-diagonal elements, bring column with largest
         norm to first column */
      
      if (hypot(a11, a21) < hypot(a12,a22))
        {
          float t1, t2;

          /* B <= B X */

          t1 = a11; a11 = a12; a12 = t1;
          t2 = a21; a21 = a22; a22 = t2;

          /* V <= V X */

          gsl_matrix_float_swap_columns(V, 0, 1);
        } 

      create_givens_float (a11, a21, &c, &s);
      
      /* compute B <= G^T B */
      
      gsl_vector_float_set (d, 0, c * a11 - s * a21);
      gsl_vector_float_set (f, 0, c * a12 - s * a22);
      gsl_vector_float_set (d, 1, s * a12 + c * a22);
      
      /* Compute U <= U G */
      
      for (i = 0; i < M; i++)
        {
          float Uip = gsl_matrix_float_get (U, i, 0);
          float Uiq = gsl_matrix_float_get (U, i, 1);
          gsl_matrix_float_set (U, i, 0, c * Uip - s * Uiq);
          gsl_matrix_float_set (U, i, 1, s * Uip + c * Uiq);
        }

      return;
    }
}


static void
chase_out_intermediate_zero_float (gsl_vector_float * d, gsl_vector_float * f, gsl_matrix_float * U, size_t k0)
{
#if !USE_BLAS
  const size_t M = U->size1;
#endif
  const size_t n = d->size;
  float c, s;
  float x, y;
  size_t k;

  x = gsl_vector_float_get (f, k0);
  y = gsl_vector_float_get (d, k0+1);

  for (k = k0; k < n - 1; k++)
    {
      create_givens_float (y, -x, &c, &s);
      
      /* Compute U <= U G */

#ifdef USE_BLAS
      {
        gsl_vector_float_view Uk0 = gsl_matrix_float_column(U,k0);
        gsl_vector_float_view Ukp1 = gsl_matrix_float_column(U,k+1);
        gsl_blas_srot(&Uk0.vector, &Ukp1.vector, c, -s);
      }
#else
      {
        size_t i;

        for (i = 0; i < M; i++)
          {
            float Uip = gsl_matrix_float_get (U, i, k0);
            float Uiq = gsl_matrix_float_get (U, i, k + 1);
            gsl_matrix_float_set (U, i, k0, c * Uip - s * Uiq);
            gsl_matrix_float_set (U, i, k + 1, s * Uip + c * Uiq);
          }
      }
#endif
      
      /* compute B <= G^T B */
      
      gsl_vector_float_set (d, k + 1, s * x + c * y);

      if (k == k0)
        gsl_vector_float_set (f, k, c * x - s * y );

      if (k < n - 2) 
        {
          float z = gsl_vector_float_get (f, k + 1);
          gsl_vector_float_set (f, k + 1, c * z); 

          x = -s * z ;
          y = gsl_vector_float_get (d, k + 2); 
        }
    }
}

static void
chase_out_trailing_zero_float (gsl_vector_float * d, gsl_vector_float * f, gsl_matrix_float * V)
{
#if !USE_BLAS
  const size_t N = V->size1;
#endif
  const size_t n = d->size;
  float c, s;
  float x, y;
  size_t k;

  x = gsl_vector_float_get (d, n - 2);
  y = gsl_vector_float_get (f, n - 2);

  for (k = n - 1; k > 0 && k--;)
    {
      create_givens_float (x, y, &c, &s);

      /* Compute V <= V G where G = [c, s ; -s, c] */

#ifdef USE_BLAS
      {
        gsl_vector_float_view Vp = gsl_matrix_float_column(V,k);
        gsl_vector_float_view Vq = gsl_matrix_float_column(V,n-1);
        gsl_blas_srot(&Vp.vector, &Vq.vector, c, -s);
      }
#else
      {
        size_t i;
   
        for (i = 0; i < N; i++)
          {
            float Vip = gsl_matrix_float_get (V, i, k);
            float Viq = gsl_matrix_float_get (V, i, n - 1);
            gsl_matrix_float_set (V, i, k, c * Vip - s * Viq);
            gsl_matrix_float_set (V, i, n - 1, s * Vip + c * Viq);
          }
      }
#endif

      /* compute B <= B G */
      
      gsl_vector_float_set (d, k, c * x - s * y);

      if (k == n - 2)
        gsl_vector_float_set (f, k, s * x + c * y );

      if (k > 0) 
        {
          float z = gsl_vector_float_get (f, k - 1);
          gsl_vector_float_set (f, k - 1, c * z); 

          x = gsl_vector_float_get (d, k - 1); 
          y = s * z ;
        }
    }
}

static void
qrstep_float (gsl_vector_float * d, gsl_vector_float * f, gsl_matrix_float * U, gsl_matrix_float * V)
{
#if !USE_BLAS
  const size_t M = U->size1;
  const size_t N = V->size1;
#endif
  const size_t n = d->size;
  float y, z;
  float ak, bk, zk, ap, bp, aq, bq;
  size_t i, k;

  if (n == 1)
    return;  /* shouldn't happen */

  /* Compute 2x2 svd directly */

  if (n == 2)
    {
      svd2_float (d, f, U, V);
      return;
    }

  /* Chase out any zeroes on the diagonal */

  for (i = 0; i < n - 1; i++)
    {
      float d_i = gsl_vector_float_get (d, i);
      
      if (d_i == 0.0)
        {
          chase_out_intermediate_zero_float (d, f, U, i);
          return;
        }
    }

  /* Chase out any zero at the end of the diagonal */

  {
    float d_nm1 = gsl_vector_float_get (d, n - 1);

    if (d_nm1 == 0.0) 
      {
        chase_out_trailing_zero_float (d, f, V);
        return;
      }
  }


  /* Apply QR reduction steps to the diagonal and offdiagonal */

  {
    float d0 = gsl_vector_float_get (d, 0);
    float f0 = gsl_vector_float_get (f, 0);
    
    float d1 = gsl_vector_float_get (d, 1);
    float f1 = gsl_vector_float_get (f, 1);
    
    {
      float mu = trailing_eigenvalue_float (d, f);
    
      y = d0 * d0 - mu;
      z = d0 * f0;
    }
    
    /* Set up the recurrence for Givens rotations on a bidiagonal matrix */
    
    ak = 0;
    bk = 0;
    
    ap = d0;
    bp = f0;
    
    aq = d1;
    bq = f1;
  }

  for (k = 0; k < n - 1; k++)
    {
      float c, s;
      create_givens_float (y, z, &c, &s);

      /* Compute V <= V G */

#ifdef USE_BLAS
      {
        gsl_vector_float_view Vk = gsl_matrix_float_column(V,k);
        gsl_vector_float_view Vkp1 = gsl_matrix_float_column(V,k+1);
        gsl_blas_srot(&Vk.vector, &Vkp1.vector, c, -s);
      }
#else
      for (i = 0; i < N; i++)
        {
          float Vip = gsl_matrix_float_get (V, i, k);
          float Viq = gsl_matrix_float_get (V, i, k + 1);
          gsl_matrix_float_set (V, i, k, c * Vip - s * Viq);
          gsl_matrix_float_set (V, i, k + 1, s * Vip + c * Viq);
        }
#endif

      /* compute B <= B G */

      {
        float bk1 = c * bk - s * z;

        float ap1 = c * ap - s * bp;
        float bp1 = s * ap + c * bp;
        float zp1 = -s * aq;

        float aq1 = c * aq;

        if (k > 0)
          {
            gsl_vector_float_set (f, k - 1, bk1);
          }

        ak = ap1;
        bk = bp1;
        zk = zp1;

        ap = aq1;

        if (k < n - 2)
          {
            bp = gsl_vector_float_get (f, k + 1);
          }
        else
          {
            bp = 0.0;
          }

        y = ak;
        z = zk;
      }

      create_givens_float (y, z, &c, &s);

      /* Compute U <= U G */

#ifdef USE_BLAS
      {
        gsl_vector_float_view Uk = gsl_matrix_float_column(U,k);
        gsl_vector_float_view Ukp1 = gsl_matrix_float_column(U,k+1);
        gsl_blas_srot(&Uk.vector, &Ukp1.vector, c, -s);
      }
#else
      for (i = 0; i < M; i++)
        {
          float Uip = gsl_matrix_float_get (U, i, k);
          float Uiq = gsl_matrix_float_get (U, i, k + 1);
          gsl_matrix_float_set (U, i, k, c * Uip - s * Uiq);
          gsl_matrix_float_set (U, i, k + 1, s * Uip + c * Uiq);
        }
#endif

      /* compute B <= G^T B */

      {
        float ak1 = c * ak - s * zk;
        float bk1 = c * bk - s * ap;
        float zk1 = -s * bp;

        float ap1 = s * bk + c * ap;
        float bp1 = c * bp;

        gsl_vector_float_set (d, k, ak1);

        ak = ak1;
        bk = bk1;
        zk = zk1;

        ap = ap1;
        bp = bp1;

        if (k < n - 2)
          {
            aq = gsl_vector_float_get (d, k + 2);
          }
        else
          {
            aq = 0.0;
          }

        y = bk;
        z = zk;
      }
    }

  gsl_vector_float_set (f, n - 2, bk);
  gsl_vector_float_set (d, n - 1, ap);
}



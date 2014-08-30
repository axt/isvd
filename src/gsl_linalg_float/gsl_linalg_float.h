/* gsl_linalg_float.h
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

#ifndef __GSL_LINALG_FLOAT__
#define __GSL_LINALG_FLOAT__


int
gsl_linalg_SV_decomp_float (gsl_matrix_float * A, gsl_matrix_float * V, gsl_vector_float * S,
                      gsl_vector_float * work);

int
gsl_linalg_SV_decomp_mod_float (gsl_matrix_float * A,
                          gsl_matrix_float * X,
                          gsl_matrix_float * V, gsl_vector_float * S, gsl_vector_float * work);


float
gsl_linalg_householder_transform_float (gsl_vector_float * v);

int
gsl_linalg_householder_mh_float (float tau, const gsl_vector_float * v, gsl_matrix_float * A);

int
gsl_linalg_householder_hv_float (float tau, const gsl_vector_float * v, gsl_vector_float * w);

int
gsl_linalg_householder_hm_float (float tau, const gsl_vector_float * v, gsl_matrix_float * A);

int
gsl_linalg_householder_hm1_float (float tau, gsl_matrix_float * A);

 
int
gsl_linalg_bidiag_unpack2_float (gsl_matrix_float * A,
                           gsl_vector_float * tau_U,
                           gsl_vector_float * tau_V,
                           gsl_matrix_float * V);

int
gsl_linalg_bidiag_decomp_float (gsl_matrix_float * A, gsl_vector_float * tau_U, gsl_vector_float * tau_V);


#endif

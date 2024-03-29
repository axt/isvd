/* linalg/givens.c
 * 
 * Copyright (C) 1996, 1997, 1998, 1999, 2000 Gerard Jungman, Brian Gough
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

/* Generate a Givens rotation (cos,sin) which takes v=(x,y) to (|v|,0) 

   From Golub and Van Loan, "Matrix Computations", Section 5.1.8 */

inline static void
create_givens_float (const float a, const float b, float *c, float *s)
{
  if (b == 0)
    {
      *c = 1;
      *s = 0;
    }
  else if (fabs (b) > fabs (a))
    {
      float t = -a / b;
      float s1 = 1.0 / sqrt (1 + t * t);
      *s = s1;
      *c = s1 * t;
    }
  else
    {
      float t = -b / a;
      float c1 = 1.0 / sqrt (1 + t * t);
      *c = c1;
      *s = c1 * t;
    }
}

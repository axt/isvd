#SUBDIRS = gsl_linalg_float 

lib_LTLIBRARIES = libisvd.la libisvdfloat.la

libisvd_la_SOURCES = isvd.c
libisvd_la_LIBADD = -lgsl -lgslcblas -lm

libisvdfloat_la_SOURCES = isvd_float.c \
			gsl_linalg_float/bidiag.c \
			gsl_linalg_float/svd.c \
			gsl_linalg_float/householder.c

libisvdfloat_la_LIBADD = -lgsl -lgslcblas -lm 

include_HEADERS = isvd.h isvd_float.h
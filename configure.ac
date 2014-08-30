AC_PREREQ(2.60)

AC_INIT([isvd], [0.1], [axt@load.hu])
AC_CONFIG_SRCDIR([src/isvd.c])
#AC_CONFIG_HEADER([config.h])

AM_INIT_AUTOMAKE
# Checks for programs. 
AC_PROG_LIBTOOL
AC_PROG_CC

# Checks for libraries. 
AC_CHECK_LIB(m,main,,       [AC_MSG_ERROR(isvd requires math library)])
AC_CHECK_LIB(gslcblas,main,,[AC_MSG_ERROR(isvd requires cblas library)])
AC_CHECK_LIB(gsl,main,,     [AC_MSG_ERROR(isvd requires gsl)])


# Checks for header files. 
AC_HEADER_STDC
AC_CHECK_HEADERS([stdlib.h])

# Checks for typedefs, structures, and compiler characteristics. 
AC_C_CONST

AC_SUBST([AM_CFLAGS], [-Wall -ansi])

# Checks for library functions. 
AC_FUNC_MALLOC
AC_CHECK_FUNCS([pow])

AC_CONFIG_FILES([
    Makefile		\
    src/Makefile	\
    test/Makefile	\
    doc/Makefile	\
])
AC_OUTPUT
#                 test/Makefile





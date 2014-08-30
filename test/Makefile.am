INCLUDES= -I$(top_builddir)/src
check_PROGRAMS = update downdate steps update_float downdate_float steps_float

update_SOURCES = update.c
update_LDADD = ../src/libisvd.la util.o

downdate_SOURCES = downdate.c
downdate_LDADD = ../src/libisvd.la util.o

steps_SOURCES = steps.c
steps_LDADD = ../src/libisvd.la util.o

update_float_SOURCES = update_float.c
update_float_LDADD = ../src/libisvdfloat.la util_float.o

downdate_float_SOURCES = downdate_float.c
downdate_float_LDADD = ../src/libisvdfloat.la util_float.o

steps_float_SOURCES = steps_float.c
steps_float_LDADD = ../src/libisvdfloat.la util_float.o

TESTS = $(check_PROGRAMS)

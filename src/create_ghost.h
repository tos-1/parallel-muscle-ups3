#ifndef GHOST_H_
#define GHOST_H_

void int_ghost( int NG, int maxsm, int *sift, int *ghost_sift, MPI_Comm comm );
void float_ghost( int NG, int maxsm, float *psi, float *ghost_psi, MPI_Comm comm );
void double_ghost( int NG, int maxsm, double *psi, double *ghost_psi, MPI_Comm comm );
void return_int_ghost( int NG, int maxsm, int *sift, int *ghost_sift, MPI_Comm comm );
void return_float_ghost( int NG, int maxsm, float *psi, float *ghost_psi, MPI_Comm comm );
void return_double_ghost( int NG, int maxsm, double *psi, double *ghost_psi, MPI_Comm comm );

#endif

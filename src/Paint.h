int Paint(int NG, int ns, int maxsm, int *sift, float *psi, MPI_Comm comm );
int pnum(int i, int j, int k, int NGy, int NG, int maxsm);
void Disk(int NG, int NGx, int NGy, int maxsm, int xyz_c[], int r, float *sph);
void Sphere(int NG, int NGx, int NGy, int maxsm, int index, int r , float *sph);

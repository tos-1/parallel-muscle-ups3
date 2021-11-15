#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#define TOP    0
#define BOTTOM 1
#define LEFT   2
#define RIGHT  3

#define pbc(A,B) (((A) >= (B)) ? (A-B):(((A) < 0) ? (A+B):(A)))

int pnum(int i, int j, int k, int NGy, int NG, int maxsm){
  return (NGy+2*maxsm)*NG*i + NG*j + k;
}

void Disk( int NG, int NGx, int NGy, int maxsm, int xyz_c[], int r, float *sph){
  int x = 0, y = r;
  int d = 3 - 2 * r;
  int i, index;
  float collapse = -3.0;

  i=-abs(x);
  while(i<=abs(x)){
    if (xyz_c[0]+i>=0 && xyz_c[0]+i<NGx+2*maxsm && xyz_c[1]+y>=0 && xyz_c[1]+y<NGy+2*maxsm) {
      index = pnum(xyz_c[0]+i, xyz_c[1]+y, xyz_c[2], NGy, NG, maxsm);
      *(sph+index)=collapse;
    }
    i++;
  }
  i=-abs(x);
  while(i<=abs(x)){
    if (xyz_c[0]+i>=0 && xyz_c[0]+i<NGx+2*maxsm && xyz_c[1]-y>=0 && xyz_c[1]-y<NGy+2*maxsm) {
      index = pnum(xyz_c[0]+i,xyz_c[1]-y, xyz_c[2], NGy, NG, maxsm);
      *(sph+index)=collapse;
    }
    i++;
  }
  i=-abs(y);
  while(i<=abs(y)){
    if (xyz_c[0]+i>=0 && xyz_c[0]+i<NGx+2*maxsm && xyz_c[1]+x>=0 && xyz_c[1]+x<NGy+2*maxsm) {
      index = pnum(xyz_c[0]+i,xyz_c[1]+x,xyz_c[2], NGy, NG, maxsm);
      *(sph+index)=collapse;
    }
    i++;
  }
  while (y >= x){
    x++;
  
    if (d > 0) {
      y--;
      d = d + 4 * (x - y) + 10;
    }
    else {
      d = d + 4 * x + 6;
    }
    i=-abs(x);
    while(i<=abs(x)){
      if (xyz_c[0]+i>=0 && xyz_c[0]+i<NGx+2*maxsm && xyz_c[1]+y>=0 && xyz_c[1]+y<NGy+2*maxsm) {
        index = pnum(xyz_c[0]+i,xyz_c[1]+y,xyz_c[2], NGy, NG, maxsm);
        *(sph+index)=collapse;
      }
      i++;
    }
    i=-abs(x);
    while(i<=abs(x)){
      if (xyz_c[0]+i>=0 && xyz_c[0]+i<NGx+2*maxsm && xyz_c[1]-y>=0 && xyz_c[1]-y<NGy+2*maxsm) {
        index = pnum(xyz_c[0]+i,xyz_c[1]-y,xyz_c[2], NGy, NG, maxsm);
        *(sph+index)=collapse;
      }
      i++;
    }
    i=-abs(y);
    while(i<=abs(y)){
      if (xyz_c[0]+i>=0 && xyz_c[0]+i<NGx+2*maxsm && xyz_c[1]+x>=0 && xyz_c[1]+x<NGy+2*maxsm) {
        index = pnum(xyz_c[0]+i,xyz_c[1]+x,xyz_c[2], NGy, NG, maxsm);
        *(sph+index)=collapse;
      }
      i++;
    }
    i=-abs(y);
    while(i<=abs(y)){
      if (xyz_c[0]+i>=0 && xyz_c[0]+i<NGx+2*maxsm && xyz_c[1]-x>=0 && xyz_c[1]-x<NGy+2*maxsm) {
        index = pnum(xyz_c[0]+i,xyz_c[1]-x,xyz_c[2], NGy, NG, maxsm);
        *(sph+index)=collapse;
      }
      i++;
    }
  }
}

/* Function for semi-circle generation, using Bresenham's algorithm */ 
void Sphere( int NG, int NGx, int NGy, int maxsm, int index, int r , float *sph ){
  int xyz_c[3],xyz_c2[3];
  xyz_c[2] = index%NG;
  xyz_c[1] = ((index-xyz_c[2])/NG)%(NGy+2*maxsm);
  xyz_c[0] = ( index-xyz_c[1]*NG-xyz_c[2] )/(NGy+2*maxsm)/NG;

  xyz_c2[0] = xyz_c[0];
  xyz_c2[1] = xyz_c[1];

  /* fixed yc */
  int x = 0, z = r;
  int d = 3 - 2 * r;

  /* don't want to miss the starting points, pbc along z only */
  xyz_c2[2] = pbc(xyz_c[2]+z,NG);
  Disk(NG, NGx, NGy, maxsm, xyz_c2, x, sph);
  xyz_c2[2] = pbc(xyz_c[2]-z,NG);
  Disk(NG, NGx, NGy, maxsm, xyz_c2, x, sph);
  xyz_c2[2] = pbc(xyz_c[2]+x,NG);
  Disk(NG, NGx, NGy, maxsm, xyz_c2, z, sph);

  while (z >= x){
    x++; 
    
    if (d > 0){
      z--;
      d = d + 4 * (x - z) + 10;
    }
    else{
      d = d + 4 * x + 6;
    }
    xyz_c2[2] = pbc(xyz_c[2]+z,NG);
    Disk(NG, NGx, NGy, maxsm, xyz_c2, x, sph);
    xyz_c2[2] = pbc(xyz_c[2]-z,NG);
    Disk(NG, NGx, NGy, maxsm, xyz_c2, x, sph);
    xyz_c2[2] = pbc(xyz_c[2]+x,NG);
    Disk(NG, NGx, NGy, maxsm, xyz_c2, z, sph);
    xyz_c2[2] = pbc(xyz_c[2]-x,NG);
    Disk(NG, NGx, NGy, maxsm, xyz_c2, z, sph);
  }
}


int Paint( int NG, int ns, int maxsm, int *sift, float *psi, MPI_Comm comm ){
  int size, rank, len;
  char pname[MPI_MAX_PROCESSOR_NAME];
  int i, j, k, l, R;

  MPI_Comm_size(comm, &size);
  MPI_Comm_rank(comm, &rank);
  MPI_Get_processor_name(pname, &len);
  pname[len] = 0;
  //printf("Hello, World! I am process %d of %d on %s.\n", rank, size, pname);

  int coords[2];
  MPI_Cart_coords(comm, rank, 2, coords);

  //printf("MPI process %d: I am located at (%d, %d)\n", rank, coords[0], coords[1]);

  int dims[2] = {0,0};
  MPI_Dims_create(size, 2, dims);
  //printf("dims= %d, %d\n",dims[0],dims[1]);
  int NGx, NGy;
  NGx = NG/dims[0];
  NGy = NG/dims[1];
  //printf("NG, NGx, NGy= %d, %d, %d\n",NG,NGx,NGy);

  int *ghost_sift;
  float *ghost_psi;
  ghost_sift = malloc(NG*(maxsm*2+NGx)*(maxsm*2+NGy)*sizeof(int *));
  ghost_psi  = malloc(NG*(maxsm*2+NGx)*(maxsm*2+NGy)*sizeof(float *));
  if (ghost_sift==NULL || ghost_psi==NULL){
    printf("Failed to allocate memory for ghost arrays.");
    exit(0);
  }

  for (int i=0;i<NG*(NGx+maxsm*2)*(NGy+2*maxsm);i++){
    *(ghost_sift+i) = -1;
    *(ghost_psi+i) = 0.0;
  }

  /* fill in array with non ghost values */
  int index, ghost_index;
  for (int i = 0; i < NGx; i++){
    for (int j = 0; j < NGy; j++){
      for (int k = 0; k < NG; k++){
	ghost_index = (NGy+2*maxsm)*NG*(i+maxsm) + NG*(j+maxsm) + k;
	index = NGy*NG*i + NG*j + k;
        *(ghost_psi+ghost_index) = *(psi+index);
        *(ghost_sift+ghost_index) = *(sift+index);
      }
    }
  }

  /* nearest neighbours */
  int nbrs[4];
  MPI_Cart_shift(comm, 1, 1, &nbrs[TOP],  &nbrs[BOTTOM]);
  MPI_Cart_shift(comm, 0, 1, &nbrs[LEFT], &nbrs[RIGHT] );
  //printf("on rank %d the neighbours are top=%d,bottom=%d,left=%d,right=%d\n",rank,nbrs[TOP],nbrs[BOTTOM],nbrs[LEFT],nbrs[RIGHT]);

  /* create sender data types (only for ghost cells initialization) */
  MPI_Datatype int_send_top, int_send_bottom, int_send_left, int_send_right;
  MPI_Datatype float_send_top, float_send_bottom, float_send_left, float_send_right;

  // send top
  int *lengths, *displacements;
  lengths = malloc(maxsm*NG*NGx*sizeof(int *));
  displacements = malloc(maxsm*NG*NGx*sizeof(int *));
  if (lengths==NULL || displacements==NULL){
    printf("Failed to allocate memory for lengths/displacements.");
    exit(0);
  }

  l = 0;
  for (i = 0; i < NGx; i++){
    for (j = 0; j < maxsm; j++){
      for (k = 0; k < NG; k++){
        displacements[l] = NGy*NG*i + NG*j + k;
        lengths[l] = 1;
        l++;
      }
    }
  }
  if (l != maxsm*NG*NGx ){
      printf("top\n");
      printf("l=%d and maxsm*NG=%d\n",l,maxsm*NG*NGx);
      exit(0);
  }
  MPI_Type_indexed(maxsm*NG*NGx, lengths, displacements, MPI_INT, &int_send_top);
  MPI_Type_indexed(maxsm*NG*NGx, lengths, displacements, MPI_FLOAT, &float_send_top);
  MPI_Type_commit(&int_send_top);
  MPI_Type_commit(&float_send_top);

  // send bottom
  l = 0;
  for (i = 0; i < NGx; i++){
    for (j = NGy-maxsm; j < NGy; j++){
      for (k = 0; k < NG; k++){
        displacements[l] = NGy*NG*i + NG*j + k;
        lengths[l] = 1;
        l++;
      }
    }
  }
  if (l != maxsm*NG*NGx){
      printf("bottom\n");
      printf("l=%d and maxsm*NG=%d\n",l,maxsm*NG*NGx);
      exit(0);
  }
  MPI_Type_indexed(maxsm*NG*NGx, lengths, displacements, MPI_INT, &int_send_bottom);
  MPI_Type_indexed(maxsm*NG*NGx, lengths, displacements, MPI_FLOAT, &float_send_bottom);
  MPI_Type_commit(&int_send_bottom);
  MPI_Type_commit(&float_send_bottom);

  // send left
  int *lengths2, *displacements2;
  lengths2 = malloc(maxsm*NG*NGy*sizeof(int *));
  displacements2 = malloc(maxsm*NG*NGy*sizeof(int *));
  if (lengths2==NULL || displacements2==NULL){
    printf("Failed to allocate memory for lengths2/displacements2.");
    exit(0);
  }
  l = 0;
  for (i = 0; i < maxsm; i++){
    for (j = 0; j < NGy; j++){
      for (k = 0; k < NG; k++){
        displacements2[l] = NGy*NG*i + NG*j + k;
        lengths2[l] = 1;
        l++;
      }
    }
  }
  if (l != maxsm*NG*NGy){
      printf("left\n");
      printf("l=%d and maxsm*NGy=%d\n",l,maxsm*NG*NGy);
      exit(0);
  }
  MPI_Type_indexed(maxsm*NG*NGy, lengths2, displacements2, MPI_INT, &int_send_left);
  MPI_Type_indexed(maxsm*NG*NGy, lengths2, displacements2, MPI_FLOAT, &float_send_left);
  MPI_Type_commit(&int_send_left);
  MPI_Type_commit(&float_send_left);

  // send right
  l = 0;
  for (i = NGx-maxsm; i < NGx; i++){
    for (j = 0; j < NGy; j++){
      for (k = 0; k < NG; k++){
        displacements2[l] = NGy*NG*i + NG*j + k;
        lengths2[l] = 1;
        l++;
      }
    }
  }
  if (l != maxsm*NG*NGy ){
      printf("right\n");
      printf("l=%d and maxsm*NG=%d\n",l,maxsm*NG*NGy);
      exit(0);
  }
  MPI_Type_indexed(maxsm*NG*NGy, lengths2, displacements2, MPI_INT, &int_send_right);
  MPI_Type_indexed(maxsm*NG*NGy, lengths2, displacements2, MPI_FLOAT, &float_send_right);
  MPI_Type_commit(&int_send_right);
  MPI_Type_commit(&float_send_right);

  /* create ghost data types */
  MPI_Datatype int_top, int_bottom, int_left, int_right;
  MPI_Datatype float_top, float_bottom, float_left, float_right;

  // top
  l = 0;
  for (i = maxsm; i < NGx+maxsm; i++){
    for (j = 0; j < maxsm; j++){
      for (k = 0; k < NG; k++){
        displacements[l] = (NGy+2*maxsm)*NG*i + NG*j + k;
        lengths[l] = 1;
        l++;
      }
    }
  }
  if (l != maxsm*NG*NGx){
      printf("l=%d and maxsm*NG=%d\n",l,maxsm*NG*NGx);
      exit(0);
  }
  MPI_Type_indexed(maxsm*NG*NGx, lengths, displacements, MPI_INT, &int_top);
  MPI_Type_indexed(maxsm*NG*NGx, lengths, displacements, MPI_FLOAT, &float_top);
  MPI_Type_commit(&int_top);
  MPI_Type_commit(&float_top);

  // bottom
  l = 0;
  for (i = maxsm; i < NGx+maxsm; i++){
    for (j = NGy+maxsm; j < NGy+2*maxsm; j++){
      for (k = 0; k < NG; k++){
        displacements[l] = (NGy+2*maxsm)*NG*i + NG*j + k;
        lengths[l] = 1;
        l++;
      }
    }
  }
  if (l != maxsm*NG*NGx){
      printf("l=%d and maxsm*NG=%d\n",l,maxsm*NG*NGx);
      exit(0);
  }
  MPI_Type_indexed(maxsm*NG*NGx, lengths, displacements, MPI_INT, &int_bottom);
  MPI_Type_indexed(maxsm*NG*NGx, lengths, displacements, MPI_FLOAT, &float_bottom);
  MPI_Type_commit(&int_bottom);
  MPI_Type_commit(&float_bottom);


  // left
  l = 0;
  for (i = 0; i < maxsm; i++){
    for (j = maxsm; j < NGy+maxsm; j++){
      for (k = 0; k < NG; k++){
        displacements2[l] = (NGy+2*maxsm)*NG*i + NG*j + k;
        lengths2[l] = 1;
        l++;
      }
    }
  }
  if (l != maxsm*NG*NGy ){
      printf("l=%d and maxsm*NG=%d\n",l,maxsm*NG*NGy);
      exit(0);
  }
  MPI_Type_indexed(maxsm*NG*NGy, lengths2, displacements2, MPI_INT, &int_left);
  MPI_Type_indexed(maxsm*NG*NGy, lengths2, displacements2, MPI_FLOAT, &float_left);
  MPI_Type_commit(&int_left);
  MPI_Type_commit(&float_left);

  // right
  l = 0;
  for (i = NGx+maxsm; i < NGx+2*maxsm; i++){
    for (j = maxsm; j < NGy+maxsm; j++){
      for (k = 0; k < NG; k++){
        displacements2[l] = (NGy+2*maxsm)*NG*i + NG*j + k;
        lengths2[l] = 1;
        l++;
      }
    }
  }
  if (l != maxsm*NG*NGy ){
      printf("l=%d and maxsm*NG=%d\n",l,maxsm*NG*NGy);
      exit(0);
  }
  MPI_Type_indexed(maxsm*NG*NGy, lengths2, displacements2, MPI_INT, &int_right);
  MPI_Type_indexed(maxsm*NG*NGy, lengths2, displacements2, MPI_FLOAT, &float_right);
  MPI_Type_commit(&int_right);
  MPI_Type_commit(&float_right);

  MPI_Sendrecv(sift, 1, int_send_top, nbrs[TOP], rank,
                 ghost_sift, 1, int_bottom, nbrs[BOTTOM], nbrs[BOTTOM], comm, MPI_STATUS_IGNORE);
  MPI_Sendrecv(psi, 1, float_send_top, nbrs[TOP], rank,
                 ghost_psi, 1, float_bottom, nbrs[BOTTOM], nbrs[BOTTOM], comm, MPI_STATUS_IGNORE);

  MPI_Sendrecv(sift, 1, int_send_bottom, nbrs[BOTTOM], rank,
                 ghost_sift, 1, int_top, nbrs[TOP], nbrs[TOP], comm, MPI_STATUS_IGNORE);
  MPI_Sendrecv(psi, 1, float_send_bottom, nbrs[BOTTOM], rank,
                 ghost_psi, 1, float_top, nbrs[TOP], nbrs[TOP], comm, MPI_STATUS_IGNORE);

  MPI_Sendrecv(sift, 1, int_send_left, nbrs[LEFT], rank,
                 ghost_sift, 1, int_right, nbrs[RIGHT], nbrs[RIGHT], comm, MPI_STATUS_IGNORE);
  MPI_Sendrecv(psi, 1, float_send_left, nbrs[LEFT], rank,
                 ghost_psi, 1, float_right, nbrs[RIGHT], nbrs[RIGHT], comm, MPI_STATUS_IGNORE);

  MPI_Sendrecv(sift, 1, int_send_right, nbrs[RIGHT], rank,
                 ghost_sift, 1, int_left, nbrs[LEFT], nbrs[LEFT], comm, MPI_STATUS_IGNORE);
  MPI_Sendrecv(psi, 1, float_send_right, nbrs[RIGHT], rank,
                 ghost_psi, 1, float_left, nbrs[LEFT], nbrs[LEFT], comm, MPI_STATUS_IGNORE);

  for (int i = 0; i < NGx+2*maxsm; i++){
    for (int j = 0; j < NGy+2*maxsm; j++){
      for (int k = 0; k < NG; k++){
        index = (NGy+2*maxsm)*NG*i + NG*j + k;
        R = *(ghost_sift+index);
        if (R>=ns) Sphere( NG, NGx, NGy, maxsm, index, R, ghost_psi );
      }
    }
  }

  /* replace values in array with non ghost values */
  for (int i = 0; i < NGx; i++){
    for (int j = 0; j < NGy; j++){
      for (int k = 0; k < NG; k++){
	ghost_index = (NGy+2*maxsm)*NG*(i+maxsm) + NG*(j+maxsm) + k;
	index = NGy*NG*i + NG*j + k;
        *(psi+index) = *(ghost_psi+ghost_index);
      }
    }
  }
  free(ghost_psi);
  free(ghost_sift);
  free(lengths);
  free(displacements);
  free(lengths2);
  free(displacements2);
  if (rank==0)  printf("finished painting halos\n");
  return 0;
}

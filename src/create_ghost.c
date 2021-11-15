#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#define TOP    0
#define BOTTOM 1
#define LEFT   2
#define RIGHT  3

void int_ghost( int NG, int maxsm, int *sift, int *ghost_sift, MPI_Comm comm ){
  int size, rank, len;
  char pname[MPI_MAX_PROCESSOR_NAME];
  int i, j, k, l;

  MPI_Comm_size(comm, &size);
  MPI_Comm_rank(comm, &rank);
  MPI_Get_processor_name(pname, &len);
  pname[len] = 0;

  int coords[2];
  MPI_Cart_coords(comm, rank, 2, coords);

  int dims[2] = {0,0};
  MPI_Dims_create(size, 2, dims);
  int NGx, NGy;
  NGx = NG/dims[0];
  NGy = NG/dims[1];

  /* fill in array with non ghost values */
  int index, ghost_index;
  for (int i = 0; i < NGx; i++){
    for (int j = 0; j < NGy; j++){
      for (int k = 0; k < NG; k++){
	ghost_index = (NGy+2*maxsm)*NG*(i+maxsm) + NG*(j+maxsm) + k;
	index = NGy*NG*i + NG*j + k;
        *(ghost_sift+ghost_index) = *(sift+index);
      }
    }
  }

  /* nearest neighbours */
  int nbrs[4];
  MPI_Cart_shift(comm, 1, 1, &nbrs[TOP],  &nbrs[BOTTOM]);
  MPI_Cart_shift(comm, 0, 1, &nbrs[LEFT], &nbrs[RIGHT] );

  /* create sender data types (only for ghost cells initialization) */
  MPI_Datatype int_send_top, int_send_bottom, int_send_left, int_send_right;

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
  MPI_Type_commit(&int_send_top);

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
  MPI_Type_commit(&int_send_bottom);

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
  MPI_Type_commit(&int_send_left);

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
  MPI_Type_commit(&int_send_right);

  /* create ghost data types */
  MPI_Datatype int_top, int_bottom, int_left, int_right;

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
  MPI_Type_commit(&int_top);

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
  MPI_Type_commit(&int_bottom);

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
  MPI_Type_commit(&int_left);

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
  MPI_Type_commit(&int_right);

  MPI_Sendrecv(sift, 1, int_send_top, nbrs[TOP], rank,
                 ghost_sift, 1, int_bottom, nbrs[BOTTOM], nbrs[BOTTOM], comm, MPI_STATUS_IGNORE);

  MPI_Sendrecv(sift, 1, int_send_bottom, nbrs[BOTTOM], rank,
                 ghost_sift, 1, int_top, nbrs[TOP], nbrs[TOP], comm, MPI_STATUS_IGNORE);

  MPI_Sendrecv(sift, 1, int_send_left, nbrs[LEFT], rank,
                 ghost_sift, 1, int_right, nbrs[RIGHT], nbrs[RIGHT], comm, MPI_STATUS_IGNORE);

  MPI_Sendrecv(sift, 1, int_send_right, nbrs[RIGHT], rank,
                 ghost_sift, 1, int_left, nbrs[LEFT], nbrs[LEFT], comm, MPI_STATUS_IGNORE);

  free(lengths);
  free(displacements);
  free(lengths2);
  free(displacements2);

}

void float_ghost( int NG, int maxsm, float *psi, float *ghost_psi, MPI_Comm comm ){
  int size, rank, len;
  char pname[MPI_MAX_PROCESSOR_NAME];
  int i, j, k, l;

  MPI_Comm_size(comm, &size);
  MPI_Comm_rank(comm, &rank);
  MPI_Get_processor_name(pname, &len);
  pname[len] = 0;

  int coords[2];
  MPI_Cart_coords(comm, rank, 2, coords);

  int dims[2] = {0,0};
  MPI_Dims_create(size, 2, dims);
  int NGx, NGy;
  NGx = NG/dims[0];
  NGy = NG/dims[1];

  /* fill in array with non ghost values */
  int index, ghost_index;
  for (int i = 0; i < NGx; i++){
    for (int j = 0; j < NGy; j++){
      for (int k = 0; k < NG; k++){
	ghost_index = (NGy+2*maxsm)*NG*(i+maxsm) + NG*(j+maxsm) + k;
	index = NGy*NG*i + NG*j + k;
        *(ghost_psi+ghost_index) = *(psi+index);
      }
    }
  }

  /* nearest neighbours */
  int nbrs[4];
  MPI_Cart_shift(comm, 1, 1, &nbrs[TOP],  &nbrs[BOTTOM]);
  MPI_Cart_shift(comm, 0, 1, &nbrs[LEFT], &nbrs[RIGHT] );

  /* create sender data types (only for ghost cells initialization) */
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
  MPI_Type_indexed(maxsm*NG*NGx, lengths, displacements, MPI_FLOAT, &float_send_top);
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
  MPI_Type_indexed(maxsm*NG*NGx, lengths, displacements, MPI_FLOAT, &float_send_bottom);
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
  MPI_Type_indexed(maxsm*NG*NGy, lengths2, displacements2, MPI_FLOAT, &float_send_left);
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
  MPI_Type_indexed(maxsm*NG*NGy, lengths2, displacements2, MPI_FLOAT, &float_send_right);
  MPI_Type_commit(&float_send_right);

  /* create ghost data types */
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
  MPI_Type_indexed(maxsm*NG*NGx, lengths, displacements, MPI_FLOAT, &float_top);
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
  MPI_Type_indexed(maxsm*NG*NGx, lengths, displacements, MPI_FLOAT, &float_bottom);
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
  MPI_Type_indexed(maxsm*NG*NGy, lengths2, displacements2, MPI_FLOAT, &float_left);
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
  MPI_Type_indexed(maxsm*NG*NGy, lengths2, displacements2, MPI_FLOAT, &float_right);
  MPI_Type_commit(&float_right);

  MPI_Sendrecv(psi, 1, float_send_top, nbrs[TOP], rank,
                 ghost_psi, 1, float_bottom, nbrs[BOTTOM], nbrs[BOTTOM], comm, MPI_STATUS_IGNORE);

  MPI_Sendrecv(psi, 1, float_send_bottom, nbrs[BOTTOM], rank,
                 ghost_psi, 1, float_top, nbrs[TOP], nbrs[TOP], comm, MPI_STATUS_IGNORE);

  MPI_Sendrecv(psi, 1, float_send_left, nbrs[LEFT], rank,
                 ghost_psi, 1, float_right, nbrs[RIGHT], nbrs[RIGHT], comm, MPI_STATUS_IGNORE);

  MPI_Sendrecv(psi, 1, float_send_right, nbrs[RIGHT], rank,
                 ghost_psi, 1, float_left, nbrs[LEFT], nbrs[LEFT], comm, MPI_STATUS_IGNORE);
  free(lengths);
  free(displacements);
  free(lengths2);
  free(displacements2);
}


void double_ghost( int NG, int maxsm, double *psi, double *ghost_psi, MPI_Comm comm ){
  int size, rank, len;
  char pname[MPI_MAX_PROCESSOR_NAME];
  int i, j, k, l;

  MPI_Comm_size(comm, &size);
  MPI_Comm_rank(comm, &rank);
  MPI_Get_processor_name(pname, &len);
  pname[len] = 0;

  int coords[2];
  MPI_Cart_coords(comm, rank, 2, coords);

  int dims[2] = {0,0};
  MPI_Dims_create(size, 2, dims);
  int NGx, NGy;
  NGx = NG/dims[0];
  NGy = NG/dims[1];

  /* fill in array with non ghost values */
  int index, ghost_index;
  for (int i = 0; i < NGx; i++){
    for (int j = 0; j < NGy; j++){
      for (int k = 0; k < NG; k++){
	ghost_index = (NGy+2*maxsm)*NG*(i+maxsm) + NG*(j+maxsm) + k;
	index = NGy*NG*i + NG*j + k;
        *(ghost_psi+ghost_index) = *(psi+index);
      }
    }
  }

  /* nearest neighbours */
  int nbrs[4];
  MPI_Cart_shift(comm, 1, 1, &nbrs[TOP],  &nbrs[BOTTOM]);
  MPI_Cart_shift(comm, 0, 1, &nbrs[LEFT], &nbrs[RIGHT] );

  /* create sender data types (only for ghost cells initialization) */
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
  MPI_Type_indexed(maxsm*NG*NGx, lengths, displacements, MPI_DOUBLE, &float_send_top);
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
  MPI_Type_indexed(maxsm*NG*NGx, lengths, displacements, MPI_DOUBLE, &float_send_bottom);
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
  MPI_Type_indexed(maxsm*NG*NGy, lengths2, displacements2, MPI_DOUBLE, &float_send_left);
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
  MPI_Type_indexed(maxsm*NG*NGy, lengths2, displacements2, MPI_DOUBLE, &float_send_right);
  MPI_Type_commit(&float_send_right);

  /* create ghost data types */
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
  MPI_Type_indexed(maxsm*NG*NGx, lengths, displacements, MPI_DOUBLE, &float_top);
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
  MPI_Type_indexed(maxsm*NG*NGx, lengths, displacements, MPI_DOUBLE, &float_bottom);
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
  MPI_Type_indexed(maxsm*NG*NGy, lengths2, displacements2, MPI_DOUBLE, &float_left);
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
  MPI_Type_indexed(maxsm*NG*NGy, lengths2, displacements2, MPI_DOUBLE, &float_right);
  MPI_Type_commit(&float_right);

  MPI_Sendrecv(psi, 1, float_send_top, nbrs[TOP], rank,
                 ghost_psi, 1, float_bottom, nbrs[BOTTOM], nbrs[BOTTOM], comm, MPI_STATUS_IGNORE);

  MPI_Sendrecv(psi, 1, float_send_bottom, nbrs[BOTTOM], rank,
                 ghost_psi, 1, float_top, nbrs[TOP], nbrs[TOP], comm, MPI_STATUS_IGNORE);

  MPI_Sendrecv(psi, 1, float_send_left, nbrs[LEFT], rank,
                 ghost_psi, 1, float_right, nbrs[RIGHT], nbrs[RIGHT], comm, MPI_STATUS_IGNORE);

  MPI_Sendrecv(psi, 1, float_send_right, nbrs[RIGHT], rank,
                 ghost_psi, 1, float_left, nbrs[LEFT], nbrs[LEFT], comm, MPI_STATUS_IGNORE);
  free(lengths);
  free(displacements);
  free(lengths2);
  free(displacements2);
}


void return_int_ghost( int NG, int maxsm, int *sift, int *ghost_sift, MPI_Comm comm ){
  int size, rank, len;
  char pname[MPI_MAX_PROCESSOR_NAME];
  int i, j, k, l;

  MPI_Comm_size(comm, &size);
  MPI_Comm_rank(comm, &rank);
  MPI_Get_processor_name(pname, &len);
  pname[len] = 0;

  int coords[2];
  MPI_Cart_coords(comm, rank, 2, coords);

  int dims[2] = {0,0};
  MPI_Dims_create(size, 2, dims);
  int NGx, NGy;
  NGx = NG/dims[0];
  NGy = NG/dims[1];

  /* fill in array with non ghost values */
  int index, ghost_index;
  for (int i = 0; i < NGx; i++){
    for (int j = 0; j < NGy; j++){
      for (int k = 0; k < NG; k++){
        ghost_index = (NGy+2*maxsm)*NG*(i+maxsm) + NG*(j+maxsm) + k;
        index = NGy*NG*i + NG*j + k;
        *(sift+index) = *(ghost_sift+ghost_index);
      }
    }
  }

  /* nearest neighbours */
  int nbrs[4];
  MPI_Cart_shift(comm, 1, 1, &nbrs[TOP],  &nbrs[BOTTOM]);
  MPI_Cart_shift(comm, 0, 1, &nbrs[LEFT], &nbrs[RIGHT] );

  /* create sender data types (only for ghost cells initialization) */
  MPI_Datatype int_send_top, int_send_bottom, int_send_left, int_send_right;

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
  MPI_Type_commit(&int_send_top);

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
  MPI_Type_commit(&int_send_bottom);

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
  MPI_Type_commit(&int_send_left);

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
  MPI_Type_commit(&int_send_right);

  /* create ghost data types */
  MPI_Datatype int_top, int_bottom, int_left, int_right;

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
  MPI_Type_commit(&int_top);

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
  MPI_Type_commit(&int_bottom);

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
  MPI_Type_commit(&int_left);

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
  MPI_Type_commit(&int_right);

  MPI_Sendrecv(ghost_sift, 1, int_top, nbrs[TOP], rank,
                 sift, 1, int_send_bottom, nbrs[BOTTOM], nbrs[BOTTOM], comm, MPI_STATUS_IGNORE);

  MPI_Sendrecv(ghost_sift, 1, int_bottom, nbrs[BOTTOM], rank,
                 sift, 1, int_send_top, nbrs[TOP], nbrs[TOP], comm, MPI_STATUS_IGNORE);

  MPI_Sendrecv(ghost_sift, 1, int_left, nbrs[LEFT], rank,
                 sift, 1, int_send_right, nbrs[RIGHT], nbrs[RIGHT], comm, MPI_STATUS_IGNORE);

  MPI_Sendrecv(ghost_sift, 1, int_right, nbrs[RIGHT], rank,
                 sift, 1, int_send_left, nbrs[LEFT], nbrs[LEFT], comm, MPI_STATUS_IGNORE);
  free(lengths);
  free(displacements);
  free(lengths2);
  free(displacements2);
}


void return_float_ghost( int NG, int maxsm, float *psi, float *ghost_psi, MPI_Comm comm ){
  int size, rank, len;
  char pname[MPI_MAX_PROCESSOR_NAME];
  int i, j, k, l;

  MPI_Comm_size(comm, &size);
  MPI_Comm_rank(comm, &rank);
  MPI_Get_processor_name(pname, &len);
  pname[len] = 0;

  int coords[2];
  MPI_Cart_coords(comm, rank, 2, coords);

  int dims[2] = {0,0};
  MPI_Dims_create(size, 2, dims);
  int NGx, NGy;
  NGx = NG/dims[0];
  NGy = NG/dims[1];

  /* fill in array with non ghost values */
  int index, ghost_index;
  for (int i = 0; i < NGx; i++){
    for (int j = 0; j < NGy; j++){
      for (int k = 0; k < NG; k++){
        ghost_index = (NGy+2*maxsm)*NG*(i+maxsm) + NG*(j+maxsm) + k;
        index = NGy*NG*i + NG*j + k;
        *(psi+index) = *(ghost_psi+ghost_index);
      }
    }
  }

  /* nearest neighbours */
  int nbrs[4];
  MPI_Cart_shift(comm, 1, 1, &nbrs[TOP],  &nbrs[BOTTOM]);
  MPI_Cart_shift(comm, 0, 1, &nbrs[LEFT], &nbrs[RIGHT] );

  /* create sender data types (only for ghost cells initialization) */
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
  MPI_Type_indexed(maxsm*NG*NGx, lengths, displacements, MPI_FLOAT, &float_send_top);
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
  MPI_Type_indexed(maxsm*NG*NGx, lengths, displacements, MPI_FLOAT, &float_send_bottom);
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
  MPI_Type_indexed(maxsm*NG*NGy, lengths2, displacements2, MPI_FLOAT, &float_send_left);
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
  MPI_Type_indexed(maxsm*NG*NGy, lengths2, displacements2, MPI_FLOAT, &float_send_right);
  MPI_Type_commit(&float_send_right);

  /* create ghost data types */
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
  MPI_Type_indexed(maxsm*NG*NGx, lengths, displacements, MPI_FLOAT, &float_top);
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
  MPI_Type_indexed(maxsm*NG*NGx, lengths, displacements, MPI_FLOAT, &float_bottom);
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
  MPI_Type_indexed(maxsm*NG*NGy, lengths2, displacements2, MPI_FLOAT, &float_left);
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
  MPI_Type_indexed(maxsm*NG*NGy, lengths2, displacements2, MPI_FLOAT, &float_right);
  MPI_Type_commit(&float_right);

  MPI_Sendrecv(ghost_psi, 1, float_top, nbrs[TOP], rank,
                 psi, 1, float_send_bottom, nbrs[BOTTOM], nbrs[BOTTOM], comm, MPI_STATUS_IGNORE);

  MPI_Sendrecv(ghost_psi, 1, float_bottom, nbrs[BOTTOM], rank,
                 psi, 1, float_send_top, nbrs[TOP], nbrs[TOP], comm, MPI_STATUS_IGNORE);

  MPI_Sendrecv(ghost_psi, 1, float_left, nbrs[LEFT], rank,
                 psi, 1, float_send_right, nbrs[RIGHT], nbrs[RIGHT], comm, MPI_STATUS_IGNORE);

  MPI_Sendrecv(ghost_psi, 1, float_right, nbrs[RIGHT], rank,
                 psi, 1, float_send_left, nbrs[LEFT], nbrs[LEFT], comm, MPI_STATUS_IGNORE);
  free(lengths);
  free(displacements);
  free(lengths2);
  free(displacements2);
}


void return_double_ghost( int NG, int maxsm, double *psi, double *ghost_psi, MPI_Comm comm ){
  int size, rank, len;
  char pname[MPI_MAX_PROCESSOR_NAME];
  int i, j, k, l;

  MPI_Comm_size(comm, &size);
  MPI_Comm_rank(comm, &rank);
  MPI_Get_processor_name(pname, &len);
  pname[len] = 0;

  int coords[2];
  MPI_Cart_coords(comm, rank, 2, coords);

  int dims[2] = {0,0};
  MPI_Dims_create(size, 2, dims);
  int NGx, NGy;
  NGx = NG/dims[0];
  NGy = NG/dims[1];

  /* fill in array with non ghost values */
  int index, ghost_index;
  for (int i = 0; i < NGx; i++){
    for (int j = 0; j < NGy; j++){
      for (int k = 0; k < NG; k++){
        ghost_index = (NGy+2*maxsm)*NG*(i+maxsm) + NG*(j+maxsm) + k;
        index = NGy*NG*i + NG*j + k;
        *(psi+index) = *(ghost_psi+ghost_index);
      }
    }
  }

  /* nearest neighbours */
  int nbrs[4];
  MPI_Cart_shift(comm, 1, 1, &nbrs[TOP],  &nbrs[BOTTOM]);
  MPI_Cart_shift(comm, 0, 1, &nbrs[LEFT], &nbrs[RIGHT] );

  /* create sender data types (only for ghost cells initialization) */
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
  MPI_Type_indexed(maxsm*NG*NGx, lengths, displacements, MPI_DOUBLE, &float_send_top);
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
  MPI_Type_indexed(maxsm*NG*NGx, lengths, displacements, MPI_DOUBLE, &float_send_bottom);
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
  MPI_Type_indexed(maxsm*NG*NGy, lengths2, displacements2, MPI_DOUBLE, &float_send_left);
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
  MPI_Type_indexed(maxsm*NG*NGy, lengths2, displacements2, MPI_DOUBLE, &float_send_right);
  MPI_Type_commit(&float_send_right);

  /* create ghost data types */
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
  MPI_Type_indexed(maxsm*NG*NGx, lengths, displacements, MPI_DOUBLE, &float_top);
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
  MPI_Type_indexed(maxsm*NG*NGx, lengths, displacements, MPI_DOUBLE, &float_bottom);
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
  MPI_Type_indexed(maxsm*NG*NGy, lengths2, displacements2, MPI_DOUBLE, &float_left);
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
  MPI_Type_indexed(maxsm*NG*NGy, lengths2, displacements2, MPI_DOUBLE, &float_right);
  MPI_Type_commit(&float_right);

  MPI_Sendrecv(ghost_psi, 1, float_top, nbrs[TOP], rank,
                 psi, 1, float_send_bottom, nbrs[BOTTOM], nbrs[BOTTOM], comm, MPI_STATUS_IGNORE);

  MPI_Sendrecv(ghost_psi, 1, float_bottom, nbrs[BOTTOM], rank,
                 psi, 1, float_send_top, nbrs[TOP], nbrs[TOP], comm, MPI_STATUS_IGNORE);

  MPI_Sendrecv(ghost_psi, 1, float_left, nbrs[LEFT], rank,
                 psi, 1, float_send_right, nbrs[RIGHT], nbrs[RIGHT], comm, MPI_STATUS_IGNORE);

  MPI_Sendrecv(ghost_psi, 1, float_right, nbrs[RIGHT], rank,
                 psi, 1, float_send_left, nbrs[LEFT], nbrs[LEFT], comm, MPI_STATUS_IGNORE);
  free(lengths);
  free(displacements);
  free(lengths2);
  free(displacements2);
}

#define MPICH_SKIP_MPICXX 1
#define OMPI_SKIP_MPICXX  1
#include "Python.h"
#include <numpy/arrayobject.h>
#include <mpi4py/mpi4py.h>
#include "Paint.h"

// bind function to Python
static PyObject* Paint_Paint(PyObject* self, PyObject* args)
{
    int ng, ns, maxsm;
    // increase reference count
    PyObject *sift_obj,*psi_obj,*py_comm;

    if ( !PyArg_ParseTuple(args, "iiiOOO", &ng, &ns, &maxsm, &sift_obj, &psi_obj, &py_comm) )
      return NULL;

    /* Interpret the input objects as numpy arrays. */
    PyObject *sift_array = PyArray_FROM_OTF( sift_obj, NPY_INT32 , NPY_IN_ARRAY);
    PyObject *psi_array = PyArray_FROM_OTF( psi_obj, NPY_FLOAT32 , NPY_IN_ARRAY);
 	
    /* If that didn't work, throw an exception, then decrement the reference count.
    Py_XDECREF accounts for the possibility of NULL objects */
    if ( sift_array == NULL || psi_array == NULL ){
      Py_XDECREF( sift_array );
      Py_XDECREF( psi_array );
      return NULL;
    }

    /* Get pointers to the data as C-types. */
    int *sift = (int*)PyArray_DATA(sift_array);
    float *psi = (float*)PyArray_DATA(psi_array);

     MPI_Comm *comm_p;
     comm_p = PyMPIComm_Get(py_comm);
   
     if (comm_p == NULL)
       return NULL;

    int zero = Paint( ng, ns, maxsm, sift, psi, *comm_p);

    /* Reference count clean up */
    Py_DECREF( sift_array );
    Py_DECREF( psi_array );

    PyObject *ret = Py_BuildValue("i", zero);
    return ret;
}

static PyMethodDef myMethods[] = {
    {"Paint", Paint_Paint, METH_VARARGS, "group halo particles in a catalogue"},
    {NULL, NULL, 0, NULL}  // always
};


#if PY_MAJOR_VERSION < 3
/* --- Python 2 --- */

PyMODINIT_FUNC init_Paint(void)
{
    /* Initialize mpi4py C-API */
    import_mpi4py();

    (void) Py_InitModule("_Paint", myMethods);

    /* Load `numpy` functionality. */
    import_array();
}

#else
/* --- Python 3 --- */
static struct PyModuleDef Paint_module = {
  PyModuleDef_HEAD_INIT,
  "Paint", 	/* m_name */
  NULL,         /* m_doc */
  -1,           /* m_size */
  myMethods,    /* m_methods */
};

PyMODINIT_FUNC
PyInit__Paint(void)
{

  PyObject *m = NULL;

  /* Initialize mpi4py C-API */
  import_mpi4py();

  /* Load `numpy` functionality. */
  import_array();

  /* Module initialization  */
  m = PyModule_Create(&Paint_module);
  if (m == NULL) goto bad;

  return m;

 bad:
  return NULL;
}

#endif

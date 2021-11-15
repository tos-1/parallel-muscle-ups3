import numpy as N
import cosmo
import pyfftw
import gadgetutils
import os
import warnings
import multiprocessing
from _Paint import Paint
from mpi4py import MPI
from mpi4py_fft import PFFT, newDistArray
from mpi4py_fft.fftw import rfftn, irfftn
from mpi4py_fft.pencil import Subcomm, Pencil
import gc


class muscleups(object):
    '''
    Inputs::
      cosmo: whether to use Eisenstein & Hu linear power spectrum ('ehu') or CLASS ('cls')
      h: normalized hubble rate
      omega_b: physical density of baryons
      Omega_cdm: cosmological cdm density
      ns: spectral index
      sigma8: power spectrum normalization
      z_pk: redshift of initial conditions
      redshift: redshift at which the output is computed
      ng: number of particles per side
      box: box size in Mpc/h
      sigmaalpt: scale of the interpolating kernel in alpt, in Mpc/h
      scheme: scheme among which to choose the evolution. The options are
        -zeld
        -2lpt
        -sc
        -muscle
      smallscheme: selecting this activates alpt. It works only with sc and muscle, while 2lpt on large scales is automatically set
      threads: number of threads used by pyfftw
      extra: initial string for the output folder and files
      seed: seed of the random number generator of initial conditions
      exact_pk: boolean to fix the fourier amplitudes of the initial density
      makeic: write the parameter file and the binaries for Gadget2. If z_pk!=redshift an error is raised. It works only with 2lpt
      pos: boolean to return the array of pos, otherwise positions are written on a binary in Gadget2 format
    '''

    def __init__(
            self,
            cosmology='ehu',
            h=0.7,
            omega_b=0.0225,
            Omega_cdm=0.25,
            ns=0.96,
            sigma8=0.8,
            z_pk=50.,
            redshift=0.,
            ng=64,
            boxsize=64,
            sigmaalpt=4.,
            scheme='zeld',
            smallscheme=None,
            makeic=False,
            return_pos=True,
            threads=1,
            extra_info='',
            seed=1,
            exact_pk=True):

        comm = MPI.COMM_WORLD
        self.comm = comm
        self.rank = comm.Get_rank()
        self.size = comm.Get_size()
        self.dims = MPI.Compute_dims(self.size, 2)

        # 2D cartesian topology
        self.cart = comm.Create_cart(
            self.dims, reorder=True, periods=[
                True, True])
        self.coord = self.cart.Get_coords(self.rank)
        self.ngx = ng / self.dims[0]
        self.ngy = ng / self.dims[1]

        # fft pencil instance
        arr = N.array([ng, ng, ng], dtype=int)
        self.transforms = {(0, 1, 2): (rfftn, irfftn)}
        # subcommunicators, z is not distributed
        self.subcomms = Subcomm(comm, [0, 0, 1])

        self.fft = PFFT(self.subcomms, arr, axes=(0, 1, 2),
                        transforms=self.transforms, dtype=N.float32)

        self.ng = int(ng)
        self.thirdim = self.ng // 2 + 1
        self.boxsize = float(boxsize)
        self.cellsize = boxsize / float(ng)
        self.h = float(h)
        self.redshift = float(redshift)
        self.z_pk = float(z_pk)
        self.sigmaalpt = float(sigmaalpt)
        self.ns = float(ns)
        self.scheme = scheme
        self.smallscheme = smallscheme
        self.return_pos = return_pos
        self.extra_info = extra_info
        self.seed = seed
        self.exact_pk = exact_pk
        self.mpx = 1  # remove in the future

        self.makeic = makeic
        if self.makeic:
            if not z_pk == redshift:
                raise ValueError(
                    "for initial conditions you need z_pk=redshift")

        # for fftw
        self.threads = threads
        cpus = multiprocessing.cpu_count()
        if not cpus >= threads:
            raise ValueError(
                "requested a number of threads > than available cpus")

        # store the kgrid
        self.kx, self.ky, self.kz, self.k = self.getkgrid()
        x, y, z = self.getmeshgrid(cellsize=False)
        ids = x * self.ng**2 + y * self.ng + z
        self.ids = ids.flatten().astype(N.int32)

        # cosmology
        if cosmology == 'cls':
            try:
                from classy import Class
                self.C = cosmo.PSClass(h, omega_b, Omega_cdm, ns, sigma8)
            except ImportError:
                print('class is not installed, using ehu')
                self.C = cosmo.EisHu(h, omega_b, Omega_cdm, ns, sigma8)

        elif cosmology == 'ehu':
            self.C = cosmo.EisHu(h, omega_b, Omega_cdm, ns, sigma8)

        else:
            raise ValueError("select the cosmology correctly")

        self.D_i = self.C.d1(z_pk)
        self.D_f = self.C.d1(redshift)
        self.growth = self.D_f / self.D_i
        self.rho = 2.77536627e+11 * self.C.Omega_0

        # growth factors, from Bouchet95
        self.f1 = self.C.Om0z(redshift)**(5. / 9.)
        self.f2 = 2. * self.C.Om0z(redshift)**(6. / 11.)

    def generate(self):
        ''' Main function '''

        # generate primordial density field
        dk = newDistArray(self.fft, forward_output=True)
        dk = self.dk()

        # returns the displacement fields
        disp_field, vel = self.disp_field(dk)

        # get eulerian positions on the grid
        pos = self.get_pos(disp_field)

        # create the folders where binaries are stored
        if self.rank == 0:
            path, fileroot = gadgetutils.writedir(
                self.sigmaalpt,
                self.extra_info,
                scheme=self.scheme,
                smallscheme=self.smallscheme,
                redshift=self.redshift,
                boxsize=self.boxsize,
                ngrid=self.ng,
                hubble=self.C.h,
                Omega0=self.C.Omega_0,
                makeic=self.makeic)

            path_to_halocatalogue = path + fileroot
        else:
            path = None
            fileroot = None
            path_to_halocatalogue = None

        path_to_halocatalogue = self.cart.bcast(path_to_halocatalogue, root=0)
        path = self.cart.bcast(path, root=0)
        fileroot = self.cart.bcast(fileroot, root=0)

        if self.rank == 0:
            if ((self.makeic) and (self.scheme == '2lpt') and (path is not None)):
                # write the param file for Gadget2
                gadgetutils.writeparam(
                    path_sims=path,
                    fileroot=fileroot,
                    scheme=self.scheme,
                    redshift=self.redshift,
                    boxsize=self.boxsize,
                    ngrid=self.ng,
                    hubble=self.C.h,
                    ombh2=self.C.omega_b,
                    Omega0=self.C.Omega_0)
                print('written gadget param file')

        if not self.makeic:
            vel = N.zeros_like(pos)

        gadgetutils.writegadget(
            pos,
            vel,
            self.redshift,
            self.boxsize,
            self.C.Omega_0,
            1. - self.C.Omega_0,
            self.C.h,
            path,
            fileroot,
            id=None)
        print('written binaries in', path + fileroot + '.dat')

        if self.return_pos:
            return pos
        else:
            return pos

    def get_pos(self, disp):
        '''
        From displacement field, get the Eulerian position with respect to an initial uniform grid
        '''
        xp = newDistArray(self.fft, forward_output=False)
        yp = newDistArray(self.fft, forward_output=False)
        zp = newDistArray(self.fft, forward_output=False)
        xp[:], yp[:], zp[:] = disp

        # setup particles on a uniform grid
        a, b, c = self.getmeshgrid()
        _a = a + xp
        _b = b + yp
        _c = c + zp

        # periodic boundary conditions PBC
        _a = _a % self.boxsize
        _b = _b % self.boxsize
        _c = _c % self.boxsize

        return _a, _b, _c

    def invdiv(self, psi_k):
        ''' returns the displacement field given the divergence field '''

        phixc = newDistArray(self.fft, forward_output=True)
        phixr = newDistArray(self.fft, forward_output=False)
        phiyc = newDistArray(self.fft, forward_output=True)
        phiyr = newDistArray(self.fft, forward_output=False)
        phizc = newDistArray(self.fft, forward_output=True)
        phizr = newDistArray(self.fft, forward_output=False)

        G = newDistArray(self.fft, forward_output=True)
        G[:] = -1 / self.k**2.
        G = N.where(self.k == 0., 0., G)

        phixc[:] = 1j * self.kx * G * psi_k
        phiyc[:] = 1j * self.ky * G * psi_k
        phizc[:] = 1j * self.kz * G * psi_k

        # diplacement field
        phixr = self.fft.backward(phixc, phixr, normalize=True)
        phiyr = self.fft.backward(phiyc, phiyr, normalize=True)
        phizr = self.fft.backward(phizc, phizr, normalize=True)

        return phixr, phiyr, phizr

    def disp_field(self, dk):
        ''' It returns the displacement field according to
            the lpt scheme you chose '''

        psi_k = newDistArray(self.fft, forward_output=True)
        psi = newDistArray(self.fft, forward_output=False)
        #disp_field = newDistArray(self.fft, rank=1, forward_output=False)
        vel = newDistArray(self.fft, forward_output=False)
        if self.smallscheme is None:

            if self.scheme == 'zeld':
                if self.rank == 0:
                    print("using Zel'dovich approximation")
                psi_k = -self.growth * dk
                disp_field = self.invdiv(psi_k)

                if self.makeic == True:
                    vel_factor = self.C.Om0z(
                        self.redshift)**(5. / 9.) * self.C.E(self.redshift) * 100. / (1. + self.redshift)
                    vel = tuple([vel_factor * x for x in disp_field[0::]])

            elif self.scheme == 'sc':
                if self.rank == 0:
                    print("using spherical collapse")
                psi = self.sc(dk)
                psi_k = self.fft.backward(psi)
                disp_field = self.invdiv(psi_k)

            elif self.scheme == 'muscle':
                if self.rank == 0:
                    print("using muscle")
                psi = self.muscle(dk)
                psi_k = self.fft.backward(psi)
                disp_field = self.invdiv(psi_k)

            elif self.scheme == 'muscleups':
                if self.rank == 0:
                    print("using muscleups")
                psi = self.muscleups(dk)
                psi_k = self.fft.forward(psi, normalize=False)
                disp_field = self.invdiv(psi_k)

            elif self.scheme == '2lpt':
                if self.rank == 0:
                    print("using 2lpt")
                psi_2lpt_k = newDistArray(self.fft, forward_output=True)
                psi_2lpt = newDistArray(self.fft, forward_output=False)
                psi_2lpt[:] = self.twolpt(dk)
                psi_2lpt_k[:] = self.fft.forward(psi_2lpt, normalize=False)
                psi_k[:] = -self.growth * dk
                disp_field[:] = self.invdiv(psi_k + psi_2lpt_k)
                # if ( (self.makeic==False) and (self.rsd==False) ):
                #    disp_field = self.invdiv(psi_k + psi_2lpt_k)

                # else:
                #    disp_field1 = self.invdiv(psi_k)
                #    disp_field2 = self.invdiv(psi_2lpt_k)

                #    if self.makeic==True:
                #        vel1 = self.C.Om0z(self.redshift)**(5./9.)*self.C.E(self.redshift)*100./(1.+self.redshift)
                #        vel2 = 2.*self.C.Om0z(self.redshift)**(6./11.)*self.C.E(self.redshift)*100./(1.+self.redshift)
                #        vel1 = tuple([vel1*x for x in disp_field1[0::]])
                #        vel2 = tuple([vel2*x for x in disp_field2[0::]])
                #        vel = [sum(x) for x in zip( vel1,vel2 )]

                #    if self.rsd==True:
                #        disp_field = disp_field1,disp_field2

                #    else:
                #        disp_field = [sum(x) for x in zip( disp_field1,disp_field2 )]
            else:
                raise ValueError(
                    "you did not correctly specify the gravity solver")

        else:  # ALPT case
            psi2_k = newDistArray(self.fft, forward_output=True)
            psi_k, psi2_k = self.alpt(dk)
            disp_field = self.invdiv(psi_k + psi2_k)

        return disp_field, vel

    def dk(self):
        ''' Makes a primordial gaussian density field '''

        #dk = newDistArray(self.fft, forward_output=True)
        #d  = newDistArray(self.fft, forward_output=False)
        ##d = N.load('psi_'+str(self.rank)+'.npy')
        #d = N.load('ICs256.npy')

        r = N.random.RandomState(self.rank)
        dk = newDistArray(self.fft, forward_output=True)
        d = newDistArray(self.fft, forward_output=False)
        sh = N.prod(N.shape(dk))
        phase = r.uniform(0, 1, sh)
        _k = self.k.flatten()
        shc = self.k.shape

        if not self.exact_pk:
            amp = N.empty(sh, dtype=N.complex64)
            amp.real = r.normal(size=sh).astype(N.float32)
            amp.imag = r.normal(size=sh).astype(N.float32)
            amp /= N.sqrt(2.)
        else:
            amp = 1

        dk = amp * N.exp(2j * N.pi * phase).astype(N.complex64)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            pk = self.C.pk_lin(_k, self.z_pk).astype(N.complex64)

        dk *= N.sqrt(pk) / self.boxsize**1.5 * self.ng**3.
        dk = N.where(_k == 0., 0., dk)
        dk = N.reshape(dk, shc)

        # make it hermitian
        d = self.fft.backward(dk, d, normalize=True)

        # save it to plot it
        # ----------------------------
        #dens = d.flatten()

        #x, y, z = self.getmeshgrid(cellsize=False)
        #ids = x*self.ng**2 + y*self.ng + z
        #ids = ids.flatten().astype(N.int32)

        #allids = None
        #alld   = None
        # if self.rank==0:
        #    allids = N.empty((self.size*ids.shape[0]), dtype=N.int32)
        #    alld   = N.empty((self.size*ids.shape[0]), dtype=N.float32)
        #self.cart.Gather(ids, allids, root=0)
        #self.cart.Gather(dens, alld, root=0)

        # if self.rank==0:
        #    indices = N.argsort(allids)
        #    alld  = alld[indices]
        #    #N.save('paperimages/parallel/dens_parallel', alld) # save this if you need to plot
        # ----------------------------

        dk = self.fft.forward(d, normalize=False)

        # to test serial with same density as parallel
        # ----------------------------
        #d = N.load('paperimages/parallel/dens_parallel.npy').reshape(self.shr)
        # dk = self.fft.forward(d,normalize=False) # N.fft.rfftn(d)
        # ----------------------------
        return dk

    def getkgrid(self):
        '''https://bitbucket.org/mpi4py/mpi4py-fft/src/9b09967ccef876cdd4bae60a8585d536d47637b5/examples/spectral_dns_solver.py?at=master#spectral_dns_solver.py-44:45,54,67
        It returns a meshgrid of kx,ky ,kz and of modulus k, on the pencil '''
        s = self.fft.local_slice()
        Nn = self.fft.global_shape()
        k = [N.fft.fftfreq(n, 1. / n).astype(int) for n in Nn[:-1]]
        k.append(N.fft.rfftfreq(Nn[-1], 1. / Nn[-1]).astype(int))
        K = [ki[si] for ki, si in zip(k, s)]
        Ks = N.meshgrid(*K, indexing='ij', sparse=True)
        Lp = 2 * N.pi / self.boxsize
        for i in range(3):
            Ks[i] = (Ks[i] * Lp).astype(N.float32)

        kk = N.asarray([N.broadcast_to(k, self.fft.shape(True)) for k in Ks])
        K = N.sum(kk * kk, 0, dtype=N.float32)
        K = N.sqrt(K)

        return kk[0], kk[1], kk[2], K

    def getmeshgrid(self, cellsize=True):
        ''' Returns local mesh on the pencil '''
        X = N.ogrid[self.fft.local_slice(False)]
        if cellsize == True:
            for i in range(3):
                X[i] = (X[i] * self.cellsize)
        X = N.asarray([N.broadcast_to(x, self.fft.shape(False)) for x in X])
        if cellsize == False:
            X = X.astype(int)
        return X

    def sc(self, dk):
        ''' spherical collapse '''
        psi_za_k = newDistArray(self.fft, forward_output=True)
        psi_za = newDistArray(self.fft, forward_output=False)
        psi_za_k = -dk * self.growth
        psi_za = self.fft.backward(psi_za_k)

        # collapse condition
        cc = newDistArray(self.fft, forward_output=False)
        cc = 1. + psi_za * 2. / 3.

        psi_za[cc > 0.] = 3. * (N.sqrt(1. + psi_za[cc > 0.] * 2. / 3.) - 1.)
        psi_za[cc <= 0.] = -3.

        # impose zero mean for the non collapsed regions
        psi_sum = N.sum(psi_za, axis=None)
        Nnc = len(N.where(cc.flatten() > 0.)[0])
        psi_sum = comm.gather(psi_sum, root=0)
        Nnc = comm.gather(Nnc, root=0)
        psi_sum = N.sum(psi_sum) / N.sum(Nnc)
        psi_sum = N.tile(psi_sum, self.size)
        psi_sum = comm.scatter(psi_sum, root=0)
        psi_za[cc > 0.] -= psi_sum

        return psi_za

    def muscle(self, dk):
        ''' MUltiscale Spherical colLapse Evolution '''

        psi_k = newDistArray(self.fft, forward_output=True)
        psi = newDistArray(self.fft, forward_output=False)
        psi_k = -dk * self.growth
        psi = self.fft.backward(psi_k)

        # collapse condition
        cc = newDistArray(self.fft, forward_output=False)
        cc = 1 + psi * 2. / 3.

        cc_R = newDistArray(self.fft, forward_output=False)
        psi_k_R = newDistArray(self.fft, forward_output=True)
        psi_R = newDistArray(self.fft, forward_output=False)
        Wk = newDistArray(self.fft, forward_output=True)
        w = newDistArray(self.fft, forward_output=False)

        twofolds = int(N.log(self.ng) / N.log(2.))
        starter = 0
        cc_R_min = 0.
        for i in N.arange(starter, twofolds):
            sigma = 2**i
            sigma_k = 2.0 * N.pi / sigma
            Wk = N.exp(-(self.k / sigma_k)**2 / 2.)

            psi_k_R = Wk * psi_k
            psi_R = self.fft.backward(psi_k_R)

            cc_R = 1 + (2. / 3.) * psi_R
            cc_R_min = N.min(cc_R)

            # if we're so low-res that nothing's collapsing
            cc_R_min = comm.gather(cc_R_min, root=0)
            cc_R_min = N.min(cc_R_min)
            cc_R_min = N.tile(cc_R_min, self.size)
            cc_R_min = comm.scatter(cc_R_min, root=0)
            if cc_R_min > 0.:
                break

            # does it collapse at any scale?
            w = (N.where(cc_R <= 0.) or N.where(cc <= 0.))
            cc[w] = N.minimum(cc_R[w], cc[w])

        # where no collapse
        psi[cc > 0.] = 3. * (N.sqrt(1 + (2. / 3.) * psi[cc > 0.]) - 1.)
        psi[cc <= 0.] = -3.

        # compute mean psi and subtract it
        psi_sum = N.sum(psi, axis=None)
        Nnc = len(N.where(cc.flatten() > 0.)[0])
        psi_sum = comm.gather(psi_sum, root=0)
        Nnc = comm.gather(Nnc, root=0)
        psi_sum = N.sum(psi_sum) / N.sum(Nnc)
        psi_sum = N.tile(psi_sum, self.size)
        psi_sum = comm.scatter(psi_sum, root=0)
        psi[cc > 0.] -= psi_sum

        return psi

    def muscleups(self, dk):
        ''' MUltiscale Spherical colLapse Evolution Using Press Schechter '''

        psi_k = -dk * self.growth
        psi = newDistArray(self.fft, forward_output=False)
        psi = self.fft.backward(psi_k, psi, normalize=True)

        maxsm = int(2 * 36. / self.cellsize)

        # collapse condition to consider
        cc = 1 + psi * 2. / 3.

        # store smoothing scale in terms of res
        sift = -N.ones(psi.shape, dtype=N.int32)
        sift[cc <= 0.] = 0

        psi_k_R = newDistArray(self.fft, forward_output=True)
        psi_R = newDistArray(self.fft, forward_output=False)

        starter = 1
        cc_R_min = 0.
        for i in N.arange(starter, maxsm):
            sigma = i * self.cellsize
            ks = self.k * sigma
            Wk = N.exp(-(ks)**2. / 2.)
            Wk.flat[0] = 1
            psi_k_R = Wk * psi_k
            psi_R = self.fft.backward(psi_k_R, psi_R, normalize=True)
            cc_R = 1 + (2. / 3.) * psi_R
            cc_R_min = N.min(cc_R)

            # if we're so low-res that nothing's collapsing
            cc_R_min = self.cart.gather(cc_R_min, root=0)
            cc_R_min = N.min(cc_R_min)
            cc_R_min = N.tile(cc_R_min, self.size)
            cc_R_min = self.cart.scatter(cc_R_min, root=0)

            if cc_R_min > 0.:  # if we're so low-res that nothing's collapsing, no voids in clouds
                break

            sift[cc_R <= 0.] = i
            cc = N.where(cc_R <= 0., cc_R, cc)

        del cc_R
        gc.collect()

        # where no collapse
        wnc = N.where(cc > 0.)
        wc = N.where(cc <= 0.)
        psi[wnc] = 3. * (N.sqrt(1 + (2. / 3.) * psi[wnc]) - 1.)
        psi[wc] = -3.0

        # extend halo seeds
        psi = N.asarray(psi, dtype=N.float32).flatten()
        sift = N.asarray(sift, N.int32).flatten()
        maxsm = N.max(sift)
        maxsm = self.cart.gather(maxsm, root=0)
        maxsm = N.max(maxsm)
        maxsm = N.tile(maxsm, self.size)
        maxsm = self.cart.scatter(maxsm, root=0)
        if self.rank == 0:
            print('maxsm=', maxsm)
        Paint(self.ng, self.mpx, maxsm, sift, psi, self.cart)
        psi = N.reshape(psi, cc.shape)
        del sift
        gc.collect()

        ks = self.k * self.sigmaalpt
        Wk = N.exp(-(ks)**2. / 2.)
        psi_k = self.fft.forward(self.twolpt(dk * Wk), normalize=False)
        psi_k = -dk * Wk * self.growth + psi_k
        psik_sc = self.fft.forward(psi, normalize=False)
        psi_k = psik_sc * (1 - Wk) + psi_k
        psi = self.fft.backward(psi_k, psi, normalize=True)

        # compute mean psi and subtract it
        psi_sum = N.sum(psi, axis=None)
        Nnc = len(N.where(cc.flatten() > 0.)[0])
        psi_sum = self.cart.gather(psi_sum, root=0)
        Nnc = self.cart.gather(Nnc, root=0)
        if self.rank == 0:
            psi_sum = N.sum(psi_sum)
            Nnc = N.sum(Nnc)
            psi_sum = psi_sum / Nnc
        psi_sum = N.tile(psi_sum, self.size)
        psi_sum = self.cart.scatter(psi_sum, root=0)
        psi = N.where(cc > 0., psi - psi_sum, psi)

        # saving by gathering
        # --------------------------------------
        #psi = psi.flatten()
        #psi_all = None
        #allids  = None
        # if self.rank==0:
        #    psi_all = N.empty((self.size*self.ids.shape[0]), dtype=N.float32)
        #    allids  = N.empty((self.size*self.ids.shape[0]), dtype=N.int32)
        #self.cart.Gather(psi, psi_all, root=0)
        #self.cart.Gather(self.ids, allids , root=0)

        # if self.rank==0:
        #    indices = N.argsort(allids)
        #    psi_all = psi_all[indices]
        #    #N.save('paperimages/parallel/psi',psi_all)
        #    N.save('paperimages/parallel/psi_parallel',psi_all)
        #psi = psi.reshape(cc.shape)
        # --------------------------------------

        return psi

    def twolpt(self, dk):
        ''' it returns the divergence of displacement potential at second order '''

        phixxc = newDistArray(self.fft, forward_output=True)
        phixxr = newDistArray(self.fft, forward_output=False)
        phiyyc = newDistArray(self.fft, forward_output=True)
        phiyyr = newDistArray(self.fft, forward_output=False)
        phizzc = newDistArray(self.fft, forward_output=True)
        phizzr = newDistArray(self.fft, forward_output=False)
        phixyc = newDistArray(self.fft, forward_output=True)
        phixyr = newDistArray(self.fft, forward_output=False)
        phixzc = newDistArray(self.fft, forward_output=True)
        phixzr = newDistArray(self.fft, forward_output=False)
        phiyzc = newDistArray(self.fft, forward_output=True)
        phiyzr = newDistArray(self.fft, forward_output=False)

        G = newDistArray(self.fft, forward_output=True)
        G = -1 / self.k**2.
        G = N.where(self.k == 0., 0., G)

        phixxc[:] = -self.kx * self.kx * G * dk
        phixyc[:] = -self.kx * self.ky * G * dk
        phixzc[:] = -self.kx * self.kz * G * dk
        phiyyc[:] = -self.ky * self.ky * G * dk
        phiyzc[:] = -self.ky * self.kz * G * dk
        phizzc[:] = -self.kz * self.kz * G * dk

        # diplacement field
        phixxr = self.fft.backward(phixxc, phixxr, normalize=True)
        phixyr = self.fft.backward(phixyc, phixyr, normalize=True)
        phixzr = self.fft.backward(phixzc, phixzr, normalize=True)
        phiyyr = self.fft.backward(phiyyc, phiyyr, normalize=True)
        phiyzr = self.fft.backward(phiyzc, phiyzr, normalize=True)
        phizzr = self.fft.backward(phizzc, phizzr, normalize=True)

        # phi2
        phixxr = phixxr * phiyyr + phixxr * phizzr + phiyyr * phizzr \
            - phixyr * phixyr - phixzr * phixzr - phiyzr * phiyzr

        # account time evolution
        phixxr *= - self.growth**2. * 3. / 7.

        return phixxr

    def alpt(self, dk):
        """
        Interpolates between large- and small-scale displacement divergences.
        """
        psi = newDistArray(self.fft, forward_output=False)
        psi_k_small = newDistArray(self.fft, forward_output=True)
        psi_k_alpt = newDistArray(self.fft, forward_output=True)
        gaussian = newDistArray(self.fft, forward_output=True)

        # small scale overdensity determined by sc
        if self.smallscheme == 'sc':
            print('implementing alpt with sc')
            psi = self.sc(dk)
            psi_k_small = self.fft.forward(psi)

        elif self.smallscheme == 'muscle':
            print('implementing alpt with muscle')
            psi = self.muscle(dk)
            psi_k_small = self.fft.forward(psi)

        else:
            print('you did not choose correctly the small scale scheme')
            assert 0

        # large scale overdensity determined by 2lpt
        psi_k_alpt = [-self.growth * dk, self.fft.forward(self.twolpt(dk))]

        if self.rank == 0:
            print('sigma of alpt: ', self.sigmaalpt)

        gaussian = N.exp(-(self.sigmaalpt * self.k)**2 / 2.)

        # I need this split just because of RSD
        psi_k_alpt[0] = psi_k_small * \
            (1. - gaussian) + psi_k_alpt[0] * gaussian
        psi_k_alpt[1] = psi_k_alpt[1] * gaussian

        # get the fourier transformed displacement potentials
        return psi_k_alpt[0], psi_k_alpt[1]

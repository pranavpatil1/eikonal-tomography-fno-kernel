# Operators to perform 3D Eikonal tomography
import pykonal
import numpy as np
from numba import jit
from joblib import Parallel, delayed
import psutil 

import occamypy as o
import torch

from lib import *
from lib.datahelper import EikonalDataset
from lib.fno import*
from lib.datahelper import UnitGaussianNormalizer
from lib.gaussian import gaussian_function_vectorized, gaussian_function


# Maximum number of cores that can be employed
if psutil.cpu_count(logical = True) != psutil.cpu_count(logical = False):
    Ncores = int(psutil.cpu_count(logical = True)*0.5) 
else:
    Ncores = psutil.cpu_count(logical = False)


# tqdm bar with parallel processing
import contextlib
import joblib
from tqdm import tqdm

@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""

    def tqdm_print_progress(self):
        if self.n_completed_tasks > tqdm_object.n:
            n_completed = self.n_completed_tasks - tqdm_object.n
            tqdm_object.update(n=n_completed)

    original_print_progress = joblib.parallel.Parallel.print_progress
    joblib.parallel.Parallel.print_progress = tqdm_print_progress

    try:
        yield tqdm_object
    finally:
        joblib.parallel.Parallel.print_progress = original_print_progress
        tqdm_object.close()

# Solver library
import pyOperator as pyOp
import pyVector
from numba import jit, prange

# import dask.distributed as daskD
from dask_util import DaskClient

# Dask for multi-node inversions
def create_client(**kwargs):
    """
       Function to create Dask client if requested
    """
    hostnames = kwargs.get("hostnames","noHost")
    pbs_args = kwargs.get("pbs_args","noPBS")
    lsf_args = kwargs.get("lsf_args","noLSF")
    slurm_args = kwargs.get("slurm_args","noSLURM")
    cluster_args = None
    if pbs_args != "noPBS":
        cluster_args = pbs_args
        cluster_name = "pbs_params"
    elif lsf_args != "noLSF":
        cluster_args = lsf_args
        cluster_name = "lsf_params"
    elif slurm_args != "noSLURM":
        cluster_args = slurm_args
        cluster_name = "slurm_params"
    if hostnames != "noHost" and cluster_args is not None:
        raise ValueError("Only one interface can be used for a client! User provided both SSH and PBS/LSF/SLURM parameters!")
    #Starting Dask client if requested
    client = None
    nWrks = None
    args = None
    if hostnames != "noHost":
        global Ncores 
        # Creating a worker for each core (needed for Joblib to work in parallel)
        hostnames_list = hostnames.split(",")
        hostname_workers = []
        nworkers = kwargs.get("nworkers",Ncores)
        nworkers = min(nworkers, Ncores)
        # Setting number of threads to 1 for joblib
        Ncores = 1
        for host in hostnames_list:
            hostname_workers += [host]*nworkers
        # args = {"hostnames":hostnames.split(",")}
        args = {"hostnames":hostname_workers}
        scheduler_file = kwargs.get("scheduler_file","noFile")
        if scheduler_file != "noFile":
            args.update({"scheduler_file_prefix":scheduler_file})
        print("Starting Dask client using the following hosts: %s with %s workers each" % (hostnames, nworkers), flush=True)
    elif cluster_args:
        n_wrks = kwargs.get("n_wrks",1)
        n_jobs = kwargs.get("n_jobs")
        args = {"n_jobs":n_jobs}
        args.update({"n_wrks":n_wrks})
        cluster_dict={elem.split(";")[0] : elem.split(";")[1] for elem in cluster_args.split(",")}
        if "cores" in cluster_dict.keys():
            cluster_dict.update({"cores":int(cluster_dict["cores"])})
        if "mem" in cluster_dict.keys():
            cluster_dict.update({"mem":int(cluster_dict["mem"])})
        if "ncpus" in cluster_dict.keys():
            cluster_dict.update({"ncpus":int(cluster_dict["ncpus"])})
        if "nanny" in cluster_dict.keys():
            nanny_flag = True
            if cluster_dict["nanny"] in "0falseFalse":
                nanny_flag = False
            cluster_dict.update({"nanny":nanny_flag})
        if "dashboard_address" in cluster_dict.keys():
            if cluster_dict["dashboard_address"] in "Nonenone":
                cluster_dict.update({"dashboard_address":None})
        if "env_extra" in cluster_dict.keys():
            cluster_dict.update({"env_extra":cluster_dict["env_extra"].split(":")})
        if "job_extra" in cluster_dict.keys():
            job_extra = cluster_dict["job_extra"].split("|")
            job_extra = [item.replace(" OR ", "|") for item in job_extra]
            cluster_dict.update({"job_extra":job_extra})
        cluster_dict={cluster_name:cluster_dict}
        args.update(cluster_dict)
        print("Starting jobqueue Dask client using %s workers on %s jobs"%(n_wrks,n_jobs), flush=True)

    if args:
        client = DaskClient(**args)
        print("Client has started!", flush=True)
        nWrks = client.getNworkers()
    return client, nWrks

# Operator to swap axis for reciprocity
class Swap_axis_op(pyOp.Operator):

    def __init__(self, Vec, par=2):
        """Constructor for swap axis operator"""
        domain = Vec # (nrec or nrec*2, nshot)
        self.par = par
        self.nshots = Vec.shape[1]
        self.rec = Vec.shape[0] if self.par >= 2 else int(Vec.shape[0]*0.5)
        shapeOutput = (self.nshots,self.rec) if self.par >= 2 else (self.nshots*2,self.rec)
        dataRange = pyVector.vectorIC(np.zeros(shapeOutput, dtype=Vec.getNdArray().dtype))
        # Setting Domain and Range of the operator
        self.setDomainRange(domain, dataRange)

    def forward(self, add, model, data):
        """Foward operator"""
        self.checkDomainRange(model, data)
        if not add: 
            data.zero()
        if self.par >= 2:
            data.getNdArray()[:] += model.getNdArray().T[:]
        else:
            data.getNdArray()[:self.nshots,:] += model.getNdArray()[:self.rec,:].T
            data.getNdArray()[self.nshots:,:] += model.getNdArray()[self.rec:,:].T
        return

    def adjoint(self, add, model, data):
        """Adjoint operator"""
        self.checkDomainRange(model, data)
        if not add: 
            model.zero()
        if self.par >= 2:
            model.getNdArray()[:] += data.getNdArray().T[:]
        else:
            model.getNdArray()[:self.rec,:].T[:] += data.getNdArray()[:self.nshots,:]
            model.getNdArray()[self.rec:,:].T[:] += data.getNdArray()[self.nshots:,:]
        return

# Reordering operator for Dask-based inversion
class Dask_TT_Order(pyOp.Operator):

    def __init__(self, dataVec, Nevents):
        """Constructor for re-ordering traveltime from remote vector when Dask is requested"""
        # Setting Domain and Range of the operator
        self.setDomainRange(dataVec, dataVec)
        # Number of events for each worker
        self.Nevents = Nevents
        # Total number of events (P- and S-wave traveltimes)
        self.nSou = int(dataVec.shape[0]*0.5)

    def forward(self, add, model, data):
        """Forward from Dask-TT vector to Local-TT vector"""
        self.checkDomainRange(model, data)
        if not add: 
            data.zero()
        dataNd = data.getNdArray()
        modelNd = model.getNdArray()  
        first_local = 0
        first_remote = 0
        for idx, Nev in enumerate(self.Nevents):
            # P-wave traveltimes
            dataNd[first_local:first_local+Nev,:] += modelNd[first_remote:first_remote+Nev,:]
            # S-wave traveltimes
            dataNd[first_local+self.nSou:first_local+self.nSou+Nev,:] += modelNd[first_remote+Nev:first_remote+2*Nev,:]
            first_local += Nev
            first_remote += 2*Nev
        return

    def adjoint(self, add, model, data):
        """Adjoint from Local-TT vector to Dask-TT vector"""
        self.checkDomainRange(model, data)
        if not add: 
            model.zero()
        dataNd = data.getNdArray()
        modelNd = model.getNdArray()
        first_local = 0
        first_remote = 0
        for idx, Nev in enumerate(self.Nevents):
            # P-wave traveltimes
            modelNd[first_remote:first_remote+Nev,:] += dataNd[first_local:first_local+Nev,:]
            # S-wave traveltimes
            modelNd[first_remote+Nev:first_remote+2*Nev,:] += dataNd[first_local+self.nSou:first_local+self.nSou+Nev,:]
            first_local += Nev
            first_remote += 2*Nev
        return

@jit(nopython=True, parallel=True, cache=True)
def compute_DD_fwd(DD_vec,tau_vec,W_DD, comb):
    ncombs = comb.shape[0]
    for icomb in prange(ncombs):
        DD_vec[icomb,:] += W_DD[icomb,:] * (tau_vec[comb[icomb,0], :] - tau_vec[comb[icomb,1], :])
    return

@jit(nopython=True, parallel=True, cache=True)
def compute_intdist_fwd(delta_x, delta_y, delta_z, x, y, z, comb):
    ncombs = comb.shape[0]
    for icomb in prange(ncombs):
        delta_x[icomb] = (x[comb[icomb,0]] - x[comb[icomb,1]])
        delta_y[icomb] = (y[comb[icomb,0]] - y[comb[icomb,1]])
        delta_z[icomb] = (z[comb[icomb,0]] - z[comb[icomb,1]])
    return

@jit(nopython=True, parallel=True, cache=True)
def compute_DD_adj_p(DD_vec, tau_vec, W_DD, comb):
    nshots = tau_vec.shape[0]
    for ishot in prange(nshots):
        # Positive coefficients
        pos_idx = np.where(comb[:,0]==ishot)[0]
        for icomb in pos_idx:
            tau_vec[ishot, :] += W_DD[icomb,:] * DD_vec[icomb,:]
        # Negative coefficients
        neg_idx = np.where(comb[:,1]==ishot)[0]
        for icomb in neg_idx:
            tau_vec[ishot, :] -= W_DD[icomb,:] * DD_vec[icomb,:]
    return

@jit(nopython=True, parallel=True, cache=True)
def init_W_DD(W_DD, WeiNd, comb, ncombs, nshots, par):
    for icomb in prange(ncombs):
        # DD Weighting P-waves
        W_DD[icomb,:] = WeiNd[comb[icomb,0], :]*WeiNd[comb[icomb,1], :]
        if par < 2:
            # DD Weighting S-waves
            W_DD[ncombs+icomb,:] = WeiNd[nshots+comb[icomb,0], :]*WeiNd[nshots+comb[icomb,1], :]
    return W_DD

@jit(nopython=True, parallel=True, cache=True)
def init_W_DD_dist(W_DD, W_Dist, ncombs, par):
    for icomb in prange(ncombs):
        W_DD[icomb,:] *= W_Dist[icomb]
        if par < 2:
            W_DD[ncombs+icomb,:] *= W_Dist[icomb]
    return W_DD


def compute_combination(ns, ev_x=None, ev_y=None, ev_z=None, maxrad=None):
    comb1 = []
    comb2 = []
    comb = []
    if maxrad is None:
        for ishot in range(ns-1):
            shot_range = np.arange(ishot+1,ns)
            ishot_cur = np.full(shot_range.shape,ishot)
            comb1.append(ishot_cur)
            comb2.append(shot_range)
    else:
        if ev_x is None or ev_y is None or ev_z is None:
            raise ValueError("Provide event coordinates!")
        if ev_x.shape[0] < ns or ev_y.shape[0] < ns or ev_z.shape[0] < ns:
            raise ValueError("Number of shots (%s) inconsistent with number of events (%s)"%(ns,ev_x.shape[0]))
        for ishot in range(ns-1):
            shot_range = np.arange(ishot+1,ns)
            ishot_cur = np.full(shot_range.shape,ishot)
            # Computing inter-event distance and threshold based on the value
            distance = np.sqrt((ev_x[shot_range]-ev_x[ishot_cur])**2.0 + (ev_y[shot_range]-ev_y[ishot_cur])**2 + (ev_z[shot_range]-ev_z[ishot_cur])**2)
            mask = distance <= maxrad
            comb1.append(ishot_cur[mask])
            comb2.append(shot_range[mask])
    comb = np.array([np.concatenate(comb1),np.concatenate(comb2)]).T
    return comb

# Double-difference traveltime operator from (Zhang and Thurber (2006) and Waldhauser and Ellsworth (2000))
from itertools import combinations
class DD_op(pyOp.Operator):

    def __init__(self, Wvec, par=2, Ev_loc=None, a=1.0, b=1.0, c=2.0, maxrad=False):
        """
           Double-difference operator
        """
        self.nshots = Wvec.shape[0] if par >= 2 else int(Wvec.shape[0]*0.5)
        self.nrecs = Wvec.shape[1]
        if maxrad:
            # Maximum radius provided (limit number of interevent considered)
            if Ev_loc is None:
                raise ValueError("Maxrad enabled. Ev_loc must be provided! ")
            self.comb = compute_combination(self.nshots, Ev_loc[:, 0], Ev_loc[:, 1], Ev_loc[:, 2], c)
            if len(self.comb) == 0:
                raise ValueError("No event pair is within % [km] distance"%c)
        else:
            # Considering every single pair
            # self.comb = np.array(list(combinations(np.arange(self.nshots), 2)))
            self.comb = compute_combination(self.nshots)
        self.ncombs = self.comb.shape[0]
        # Defining weighting vector
        WeiNd = Wvec.getNdArray()
        shape_DD_vec = (self.ncombs,self.nrecs) if par >= 2 else (2*self.ncombs,self.nrecs)
        self.W_DD = np.zeros(shape_DD_vec, dtype=WeiNd.dtype)
        self.W_DD = init_W_DD(self.W_DD, WeiNd, self.comb, self.ncombs, self.nshots, par)
        self.setDomainRange(Wvec,pyVector.vectorIC(self.W_DD))
        # Parametrization for tomography
        self.par = par
        # Distance weighting see equation 16 in A Double-Difference Earthquake Location Algorithm: Method and Application to the Northern Hayward Fault, California 
        # by Felix Waldhauser and William L. Ellsworth (2000)
        if Ev_loc is not None:
            # Compute inter-event distances
            delta_x = np.zeros_like(self.W_DD[:self.ncombs,0])
            delta_y = np.zeros_like(self.W_DD[:self.ncombs,0])
            delta_z = np.zeros_like(self.W_DD[:self.ncombs,0])
            # Compute delta x
            compute_intdist_fwd(delta_x, delta_y, delta_z, Ev_loc[:, 0], Ev_loc[:, 1], Ev_loc[:, 2], self.comb)
            s_ev = np.sqrt(delta_x*delta_x + delta_y*delta_y, + delta_z*delta_z)
            W_Dist = np.maximum(0.0, 1.0 - (s_ev/c)**a)**b
            self.W_DD = init_W_DD_dist(self.W_DD, W_Dist, self.ncombs, par)

    def forward(self, add, model, data):
        """Forward double-difference operator (absolute -> differential)"""
        self.checkDomainRange(model, data)
        if not add: 
            data.zero()
        dataNd = data.getNdArray()
        modelNd = model.getNdArray()  
        compute_DD_fwd(dataNd[:self.ncombs,:], modelNd[:self.nshots,:], self.W_DD[:self.ncombs,:], self.comb)
        if self.par < 2:
            compute_DD_fwd(dataNd[self.ncombs:,:], modelNd[self.nshots:,:], self.W_DD[self.ncombs:,:], self.comb)
        return

    def adjoint(self, add, model, data):
        """Adjoint double-difference operator (differential -> absolute)"""
        self.checkDomainRange(model, data)
        if not add: 
            model.zero()
        dataNd = data.getNdArray()
        modelNd = model.getNdArray()
        compute_DD_adj_p(dataNd[:self.ncombs,:], modelNd[:self.nshots,:], self.W_DD[:self.ncombs,:], self.comb)
        if self.par < 2:
            compute_DD_adj_p(dataNd[self.ncombs:,:], modelNd[self.nshots:,:], self.W_DD[self.ncombs:,:], self.comb)
        return

    def scale_weights(self, scale):
        """Function to scale inversion weights (useful for tomoDD process)"""
        self.W_DD *= scale
        return

@jit(nopython=True, cache=True)
def extract_tt_3D_FWD(tt_3D, ch_y, ch_x, ch_z, oy, ox, oz, dy, dx, dz):
    """Function to extract the traveltime at specific channel locations"""
    tt_ch = np.zeros(ch_x.shape)
    # ny = tt_3D.shape[0]
    # nx = tt_3D.shape[1]
    # nz = tt_3D.shape[2]
    for ich in range(tt_ch.shape[0]):
        wy = (ch_y[ich] - oy) / dy
        wx = (ch_x[ich] - ox) / dx
        wz = (ch_z[ich] - oz) / dz
        iy = int(wy)
        ix = int(wx)
        iz = int(wz)
        # Interpolation weights
        wy -= iy
        wx -= ix
        wz -= iz
        tt_ch[ich] += tt_3D[iy,ix,iz] * (1.0 - wy)*(1.0 - wx)*(1.0 - wz) + tt_3D[iy,ix,iz+1] * (1.0 - wy)*(1.0 - wx)*(wz) + tt_3D[iy,ix+1,iz] * (1.0 - wy)*(wx)*(1.0 - wz)  + tt_3D[iy+1,ix,iz] * (wy)*(1.0 - wx)*(1.0 - wz)  + tt_3D[iy,ix+1,iz+1] * (1.0 - wy)*(wx)*(wz)  + tt_3D[iy+1,ix,iz+1] * (wy)*(1.0 - wx)*(wz)  + tt_3D[iy+1,ix+1,iz] * (wy)*(wx)*(1.0 - wz)  + tt_3D[iy+1,ix+1,iz+1] * (wy)*(wx)*(wz) 
    return tt_ch

def compute_travel_time(vel, ishot, oy, ox, oz, dy, dx, dz, SouPos, RecPos, Acc_Inj, TTsrc=None, returnTT=True):
    """Function to compute traveltime in parallel"""
    velocity = pykonal.fields.ScalarField3D(coord_sys="cartesian")
    velocity.min_coords = oy, ox, oz
    velocity.node_intervals = dy, dx, dz
    velocity.npts = vel.shape[0], vel.shape[1], vel.shape[2]
    if SouPos.ndim == 2:
        # Single point source
        if Acc_Inj:
            # Set Eikonal solver
            solver_ek = pykonal.solver.PointSourceSolver(coord_sys="cartesian")
            solver_ek.vv.min_coords = velocity.min_coords
            solver_ek.vv.node_intervals = velocity.node_intervals
            solver_ek.vv.npts = velocity.npts
            solver_ek.vv.values[:] = vel
            # Setting source position (ys,xs,zs)
            solver_ek.src_loc = [SouPos[ishot,1],SouPos[ishot,0],SouPos[ishot,2]] 
        else:
            # Set Eikonal solver
            solver_ek = pykonal.EikonalSolver(coord_sys="cartesian")
            solver_ek.vv.min_coords = velocity.min_coords
            solver_ek.vv.node_intervals = velocity.node_intervals
            solver_ek.vv.npts = velocity.npts
            solver_ek.vv.values[:] = vel
            # Initial conditions
            solver_ek.tt.values[:] = np.inf
            solver_ek.known[:] = False
            solver_ek.unknown[:] = True
            eq_iz = int((SouPos[ishot,2]-oz)/dz + 0.5)
            eq_iy = int((SouPos[ishot,1]-oy)/dy + 0.5)
            eq_ix = int((SouPos[ishot,0]-ox)/dx + 0.5)
            src_idx = (eq_iy, eq_ix, eq_iz)
            solver_ek.tt.values[src_idx] = 0.0
            solver_ek.unknown[src_idx] = False
            solver_ek.trial.push(*src_idx)
    else:
        # Multiple source points
        npnt_src = SouPos.shape[2]
        for iPnt in range(npnt_src):
            eq_iz = int((SouPos[ishot,2,iPnt]-oz)/dz + 0.5)
            eq_iy = int((SouPos[ishot,1,iPnt]-oy)/dy + 0.5)
            eq_ix = int((SouPos[ishot,0,iPnt]-ox)/dx + 0.5)
            src_idx = (eq_iy, eq_ix, eq_iz)
            solver_ek.tt.values[src_idx] = 0.0 if TTsrc is None else TTsrc[iPnt]
            solver_ek.unknown[src_idx] = False
            solver_ek.trial.push(*src_idx)
    # Solving Eikonal equation
    solver_ek.solve()
    traveltimes = extract_tt_3D_FWD(solver_ek.tt.values, RecPos[:,1], RecPos[:,0], RecPos[:,2], oy, ox, oz, dy, dx, dz)
    if returnTT:
        return traveltimes, solver_ek.tt.values
    else:
        return traveltimes

class EikonalTT_3D(pyOp.Operator):

    def __init__(self, vel, tt_data, oy, ox, oz, dy ,dx, dz, SouPos, RecPos, TTsrc=None, verbose=False, **kwargs):
        """3D Eikonal-equation traveltime prediction operator"""
        # Setting Domain and Range of the operator
        self.setDomainRange(vel, tt_data)
        # Setting acquisition geometry
        self.nSou = SouPos.shape[0]
        # Get velocity array
        velNd = vel.getNdArray()
        # Accurate injection of initial conditions
        self.Acc_Inj = kwargs.get("Acc_Inj", True)
        # Getting number of threads to run the modeling code
        self.nthrs = min(self.nSou, Ncores, kwargs.get("nthreads", Ncores))
        self.nRec = RecPos.shape[0]
        self.SouPos = SouPos.copy()
        self.RecPos = RecPos.copy()
        if TTsrc is not None:
            if len(TTsrc) != self.nSou:
                raise ValueError("Number of initial traveltime (len(TTsrc)=%s) inconsistent with number of sources (%s)"%(len(TTsrc),self.nSou))
        else:
            TTsrc = [None]*self.nSou
        self.TTsrc = TTsrc # Traveltime vector for distributed sources
        dataShape = tt_data.shape
        self.oy = oy
        self.ox = ox
        self.oz = oz
        self.dy = dy
        self.dx = dx
        self.dz = dz
        self.ncomp = vel.shape[0] # Number of velocities to use
        self.ny = vel.shape[1]
        self.nx = vel.shape[2]
        self.nz = vel.shape[3]
        self.xAxis = np.linspace(ox, ox+(self.nx-1)*dx, self.nx)
        self.yAxis = np.linspace(oy, oy+(self.ny-1)*dy, self.ny)
        self.zAxis = np.linspace(oz, oz+(self.nz-1)*dz, self.nz)
        # Use smallest possible domain (Ginsu knives)
        self.ginsu = kwargs.get("ginsu", False)
        buffer = kwargs.get("buffer", 2.0) # By default 2.0 km
        self.bufferX = int(buffer/dx)
        self.bufferY = int(buffer/dy)
        self.bufferZ = int(buffer/dz)
        if dataShape[0] != self.nSou*self.ncomp:
            raise ValueError("Number of sources inconsistent with traveltime vector (data_shape[0])")
        if dataShape[1] != self.nRec:
            raise ValueError("Number of receivers inconsistent with traveltime vector (data_shape[1])")
        # List of traveltime maps to avoid double computation
        self.tt_maps = []
        self.allocateTT = kwargs.get("allocateTT", False)
        if self.allocateTT:
            for _ in range(self.nSou*self.ncomp):
                self.tt_maps.append(np.zeros_like(velNd[0,:,:,:]))
        # verbosity of the program
        self.verbose = verbose
    
    def forward(self, add, model, data):
        """Forward non-linear traveltime prediction"""
        self.checkDomainRange(model, data)
        if not add: 
            data.zero()
        dataNd = data.getNdArray()
        velNd = model.getNdArray()
        # Parallel modeling 
        # result = Parallel(n_jobs=self.nthrs, verbose=self.verbose, backend="multiprocessing")(delayed(self.compute_travel_time)(velNd, ishot) for ishot in range(self.nSou))
        for icomp in range(self.ncomp):
            if self.ginsu:
                minX = np.zeros(self.nSou, dtype=int)
                maxX = np.zeros(self.nSou, dtype=int)
                minY = np.zeros(self.nSou, dtype=int)
                maxY = np.zeros(self.nSou, dtype=int)
                minZ = np.zeros(self.nSou, dtype=int)
                maxZ = np.zeros(self.nSou, dtype=int)
                for ishot in range(self.nSou):
                    minX[ishot] =  max(0, min(np.argmin(abs(self.xAxis-self.SouPos[ishot,0])), np.argmin(abs(self.xAxis-self.RecPos[:,0].min())))-self.bufferX)
                    maxX[ishot] =  min(self.nx, max(np.argmin(abs(self.xAxis-self.SouPos[ishot,0])), np.argmin(abs(self.xAxis-self.RecPos[:,0].max())))+self.bufferX)
                    minY[ishot] =  max(0, min(np.argmin(abs(self.yAxis-self.SouPos[ishot,1])), np.argmin(abs(self.yAxis-self.RecPos[:,1].min())))-self.bufferY)
                    maxY[ishot] =  min(self.ny, max(np.argmin(abs(self.yAxis-self.SouPos[ishot,1])), np.argmin(abs(self.yAxis-self.RecPos[:,1].max())))+self.bufferY)
                    minZ[ishot] =  max(0, min(np.argmin(abs(self.zAxis-self.SouPos[ishot,2])), np.argmin(abs(self.zAxis-self.RecPos[:,2].min())))-self.bufferZ)
                    maxZ[ishot] =  min(self.nz, max(np.argmin(abs(self.zAxis-self.SouPos[ishot,2])), np.argmin(abs(self.zAxis-self.RecPos[:,2].max())))+self.bufferZ)
                result = Parallel(n_jobs=self.nthrs, verbose=self.verbose)(delayed(compute_travel_time)(velNd[icomp,minY[ishot]:maxY[ishot],minX[ishot]:maxX[ishot],minZ[ishot]:maxZ[ishot]], ishot, self.yAxis[minY[ishot]], self.xAxis[minX[ishot]], self.zAxis[minZ[ishot]], self.dy, self.dx, self.dz, self.SouPos, self.RecPos, self.Acc_Inj, self.TTsrc[ishot], self.allocateTT) for ishot in range(self.nSou))
            else:
                result = Parallel(n_jobs=self.nthrs, verbose=self.verbose)(delayed(compute_travel_time)(velNd[icomp,:,:,:], ishot, self.oy, self.ox, self.oz, self.dy, self.dx, self.dz, self.SouPos, self.RecPos, self.Acc_Inj, self.TTsrc[ishot], self.allocateTT) for ishot in range(self.nSou))
            for ishot in range(self.nSou):
                if self.allocateTT:
                    dataNd[ishot+icomp*self.nSou, :] += result[ishot][0]
                    if self.ginsu:
                        self.tt_maps[ishot+icomp*self.nSou][minY[ishot]:maxY[ishot],minX[ishot]:maxX[ishot],minZ[ishot]:maxZ[ishot]] = result[ishot][1]
                    else:
                        self.tt_maps[ishot+icomp*self.nSou][:] = result[ishot][1]
                else:
                    dataNd[ishot+icomp*self.nSou, :] += result[ishot]
        return

class EikonalTT_3D_LocInv(pyOp.Operator):

    def __init__(self, vel, tt_data, oy, ox, oz, dy ,dx, dz, SouPos, RecPos, **kwargs):
        """3D Eikonal-equation traveltime prediction from receiver for source-location inversion"""
        # Setting Domain and Range of the operator
        self.setDomainRange(SouPos, tt_data)
        self.nSou = SouPos.shape[0]
        self.nRec = RecPos.shape[0]
        # Accurate injection of initial conditions
        self.Acc_Inj = kwargs.get("Acc_Inj", True)
        # Getting number of threads to run the modeling code
        self.RecPos = RecPos.copy()
        self.nthrs = min(self.nRec, Ncores)
        dataShape = tt_data.shape
        if dataShape[0] != self.nSou:
            raise ValueError("Number of sources inconsistent with traveltime vector (shape[0])")
        if dataShape[1] != self.nRec:
            raise ValueError("Number of receivers inconsistent with traveltime vector (shape[1])")
        self.oy = oy
        self.ox = ox
        self.oz = oz
        self.dy = dy
        self.dx = dx
        self.dz = dz
        self.ny = vel.shape[0]
        self.nx = vel.shape[1]
        self.nz = vel.shape[2]
        # List of traveltime maps to avoid double computation
        self.tt_maps = []
        for _ in range(self.nRec):
            self.tt_maps.append(np.zeros_like(vel.getNdArray()))
        # verbosity of the program
        self.verbose = kwargs.get("verbose", True)
        # Parallel modeling of initial traveltime maps
        if self.verbose:
            with tqdm_joblib(tqdm(desc="Computing reciprocal traveltimes", total=self.nRec)) as progress_bar:
                result = Parallel(n_jobs=self.nthrs)(delayed(compute_travel_time)(vel.getNdArray(), irec, self.oy, self.ox, self.oz, self.dy, self.dx, self.dz, self.RecPos, SouPos.getNdArray(), self.Acc_Inj, None, True) for irec in range(self.nRec))
        else:
           result = Parallel(n_jobs=self.nthrs)(delayed(compute_travel_time)(vel.getNdArray(), irec, self.oy, self.ox, self.oz, self.dy, self.dx, self.dz, self.RecPos, SouPos.getNdArray(), self.Acc_Inj, None, True) for irec in range(self.nRec)) 
        for irec in range(self.nRec):
                self.tt_maps[irec][:] = result[irec][1]
    
    def forward(self, add, model, data):
        """Forward non-linear traveltime prediction for source-location inversion"""
        self.checkDomainRange(model, data)
        if not add: 
            data.zero()
        dataNd = data.getNdArray()
        modelNd = model.getNdArray()
        for irec in range(self.nRec):
            dataNd[:,irec] += extract_tt_3D_FWD(self.tt_maps[irec], modelNd[:,1], modelNd[:,0], modelNd[:,2], self.oy, self.ox, self.oz, self.dy, self.dx, self.dz)
        return

#######################################################
# Linearized operators
#######################################################

# @jit(nopython=False, cache=True)
def compute_der_TT(TT, oy, ox, oz, dy, dx ,dz, SouLoc):
    # Derivative along y
    dT_grad_temp = np.gradient(TT, (dy), axis=0)
    dT_dys = extract_tt_3D_FWD(dT_grad_temp, SouLoc[:,1], SouLoc[:,0], SouLoc[:,2], oy, ox, oz, dy, dx, dz)
    # Derivative along x
    dT_grad_temp = np.gradient(TT, (dx), axis=1)
    dT_dxs = extract_tt_3D_FWD(dT_grad_temp, SouLoc[:,1], SouLoc[:,0], SouLoc[:,2], oy, ox, oz, dy, dx, dz)
    # Derivative along z
    dT_grad_temp = np.gradient(TT, (dz), axis=2)
    dT_dzs = extract_tt_3D_FWD(dT_grad_temp, SouLoc[:,1], SouLoc[:,0], SouLoc[:,2], oy, ox, oz, dy, dx, dz)
    return dT_dys, dT_dxs, dT_dzs


def interp_der_TT(TT, oy, ox, oz, dy, dx ,dz, SouLoc, dT_grady, dT_gradx, dT_gradz):
    # Derivative along y
    dT_dys = extract_tt_3D_FWD(dT_grady, SouLoc[:,1], SouLoc[:,0], SouLoc[:,2], oy, ox, oz, dy, dx, dz)
    # Derivative along x
    dT_dxs = extract_tt_3D_FWD(dT_gradx, SouLoc[:,1], SouLoc[:,0], SouLoc[:,2], oy, ox, oz, dy, dx, dz)
    # Derivative along z
    dT_dzs = extract_tt_3D_FWD(dT_gradz, SouLoc[:,1], SouLoc[:,0], SouLoc[:,2], oy, ox, oz, dy, dx, dz)
    return dT_dys, dT_dxs, dT_dzs


def compute_der_fields(TT, dy, dx, dz):
    TT_grady = np.gradient(TT, (dy), axis=0)
    TT_gradx = np.gradient(TT, (dx), axis=1)
    TT_gradz = np.gradient(TT, (dz), axis=2)
    return TT_grady, TT_gradx, TT_gradz


# Eikonal source-location operator
class EikonalTT_3D_LocInv_lin(pyOp.Operator):

    def __init__(self, vel, tt_data, oy, ox, oz, dy ,dx, dz, SouPos, RecPos, **kwargs):
        """3D Eikonal-equation traveltime prediction from receiver for source-location inversion Jacobian"""
        if not isinstance(SouPos, pyVector.vector):
            raise("")
        # Setting Domain and Range of the operator
        self.setDomainRange(SouPos, tt_data)
        self.nSou = SouPos.shape[0]
        self.nRec = RecPos.shape[0]
        self.SouLoc0 = SouPos.clone()
        # Accurate injection of initial conditions
        self.Acc_Inj = kwargs.get("Acc_Inj", True)
        # Getting number of threads to run the modeling code
        self.RecPos = RecPos.copy()
        self.nthrs = min(self.nRec, Ncores)
        dataShape = tt_data.shape
        if dataShape[0] != self.nSou:
            raise ValueError("Number of sources inconsistent with traveltime vector (shape[0])")
        if dataShape[1] != self.nRec:
            raise ValueError("Number of receivers inconsistent with traveltime vector (shape[1])")
        self.oy = oy
        self.ox = ox
        self.oz = oz
        self.dy = dy
        self.dx = dx
        self.dz = dz
        self.ny = vel.shape[0]
        self.nx = vel.shape[1]
        self.nz = vel.shape[2]
        # List of traveltime maps to avoid extra computation
        tt_maps = kwargs.get("tt_maps", None)
        if tt_maps is None:
            self.tt_maps = []
            for _ in range(self.nRec):
                self.tt_maps.append(np.zeros_like(vel.getNdArray()))
        else:
            self.tt_maps = tt_maps
        # verbosity of the program
        self.verbose = kwargs.get("verbose", True)
        # Parallel modeling of initial traveltime maps (if necessary)
        if tt_maps is None:
            if self.verbose:
                with tqdm_joblib(tqdm(desc="Computing reciprocal traveltimes", total=self.nRec)) as progress_bar:
                    result = Parallel(n_jobs=self.nthrs)(delayed(compute_travel_time)(vel.getNdArray(), irec, self.oy, self.ox, self.oz, self.dy, self.dx, self.dz, self.RecPos, SouPos.getNdArray(), self.Acc_Inj, None, True) for irec in range(self.nRec))
            else:
                result = Parallel(n_jobs=self.nthrs)(delayed(compute_travel_time)(vel.getNdArray(), irec, self.oy, self.ox, self.oz, self.dy, self.dx, self.dz, self.RecPos, SouPos.getNdArray(), self.Acc_Inj, None, True) for irec in range(self.nRec)) 
            for irec in range(self.nRec):
                    self.tt_maps[irec][:] = result[irec][1]
        # List of derivative arrays
        self.store_tt_der = kwargs.get("store_tt_der", False)
        if self.store_tt_der:
            self.tt_grady = []
            self.tt_gradx = []
            self.tt_gradz = []
            result = Parallel(n_jobs=self.nthrs)(delayed(compute_der_fields)(self.tt_maps[irec], self.dy, self.dx, self.dz) for irec in range(self.nRec)) 
            for irec in range(self.nRec):
                self.tt_grady.append(result[irec][0])
                self.tt_gradx.append(result[irec][1])
                self.tt_gradz.append(result[irec][2])
    
    def forward(self, add, model, data):
        """Forward operator of Jacobian matrix for source-location inversion"""
        self.checkDomainRange(model, data)
        if not add: 
            data.zero()
        dataNd = data.getNdArray()
        modelNd = model.getNdArray()
        SouLoc0Nd = self.SouLoc0.getNdArray()
        # Computing partial derivatives with respect source location
        if self.store_tt_der:
            for irec in range(self.nRec):
                dT_dys, dT_dxs, dT_dzs = interp_der_TT(self.tt_maps[irec], self.oy, self.ox, self.oz, self.dy, self.dx, self.dz, SouLoc0Nd, self.tt_grady[irec], self.tt_gradx[irec], self.tt_gradz[irec])
                dataNd[:,irec] += dT_dys*modelNd[:,1] + dT_dxs*modelNd[:,0] + dT_dzs*modelNd[:,2]
        else:
            result = Parallel(n_jobs=self.nthrs)(delayed(compute_der_TT)(self.tt_maps[irec], self.oy, self.ox, self.oz, self.dy, self.dx, self.dz, SouLoc0Nd) for irec in range(self.nRec)) 
            for irec in range(self.nRec):
                dataNd[:,irec] += result[irec][0]*modelNd[:,1] + result[irec][1]*modelNd[:,0] + result[irec][2]*modelNd[:,2]
        return

    def adjoint(self, add, model, data):
        """Adjoint operator of Jacobian matrix for source-location inversion"""
        self.checkDomainRange(model, data)
        if not add: 
            model.zero()
        dataNd = data.getNdArray()
        modelNd = model.getNdArray()
        SouLoc0Nd = self.SouLoc0.getNdArray()
        # Computing partial derivatives with respect source location
        if self.store_tt_der:
            for irec in range(self.nRec):
                dT_dys, dT_dxs, dT_dzs = interp_der_TT(self.tt_maps[irec], self.oy, self.ox, self.oz, self.dy, self.dx, self.dz, SouLoc0Nd, self.tt_grady[irec], self.tt_gradx[irec], self.tt_gradz[irec])
                modelNd[:,1] += dT_dys*dataNd[:,irec]
                modelNd[:,0] += dT_dxs*dataNd[:,irec]
                modelNd[:,2] += dT_dzs*dataNd[:,irec]
        else:
            result = Parallel(n_jobs=self.nthrs)(delayed(compute_der_TT)(self.tt_maps[irec], self.oy, self.ox, self.oz, self.dy, self.dx, self.dz, SouLoc0Nd) for irec in range(self.nRec)) 
            for irec in range(self.nRec):
                modelNd[:,1] += result[irec][0]*dataNd[:,irec]
                modelNd[:,0] += result[irec][1]*dataNd[:,irec]
                modelNd[:,2] += result[irec][2]*dataNd[:,irec]
        return

    def set_SourceLoc(self,SouLoc0):
        """Set source location for derivative computation"""
        self.SouLoc0.copy(SouLoc0)
        return

# Eikonal tomography-related operator
def sorting3D(tt, idx_l, ordering="a"):
    idx1 = idx_l[:,0]
    idx2 = idx_l[:,1]
    idx3 = idx_l[:,2]
    idx = np.ravel_multi_index((idx1, idx2, idx3), tt.shape)
    if ordering == "a":
        sorted_indices = np.argsort(tt.ravel()[idx])
    elif ordering == "d":
        sorted_indices = np.argsort(-tt.ravel()[idx])
    else:
        raise ValueError("Unknonw ordering: %s! Provide a or d for ascending or descending" % ordering)
        
    # Sorted indices for entire array
    sorted_indices = idx[sorted_indices]
    
    # Sorting indices
    idx1, idx2, idx3 = np.unravel_index(sorted_indices, tt.shape)
    idx_sort = np.array([idx1,idx2,idx3], dtype=np.int64).T
    # idx_sort = [[iy,ix,iz] for iy, ix, iz in zip(idx1, idx2, idx3)]
    return idx_sort

@jit(nopython=True, cache=True)
def FMM_tt_lin_fwd3D(delta_v, delta_tt, vv, tt, tt_idx, dy, dx, dz):
    """Fast-marching method linearized forward"""
    ny = delta_v.shape[0]
    nx = delta_v.shape[1]
    nz = delta_v.shape[2]
    drxns = [-1, 1]
    dy_inv = 1.0 / dy
    dx_inv = 1.0 / dx
    dz_inv = 1.0 / dz
    ds_inv = np.array([dy_inv, dx_inv, dz_inv])
    
    # Shift variables
    order = np.zeros(2, dtype=np.int64)
    shift = np.zeros(3, dtype=np.int64)
    idrx = np.zeros(3, dtype=np.int64)
    fdt0 = np.zeros(3)
    
    # Scaling the velocity perturbation
    delta_v_scaled = - 2.0 * delta_v / (vv * vv * vv)
    
    # Looping over all indices to solve linear equations from increasing traveltime values
    for idx_t0 in tt_idx:
        tt0 = tt[idx_t0[0], idx_t0[1], idx_t0[2]]
        # If T = 0 or v = 0, then assuming zero to avoid singularity
        if tt0 == 0.0 or vv[idx_t0[0], idx_t0[1], idx_t0[2]] == 0.0:
            continue

        fdt0.fill(0.0)
        idrx.fill(0)
        for iax in range(3):
            # Loop over neighbourning points to find up-wind direction
            fdt = np.zeros(2)
            order.fill(0)
            shift.fill(0)
            for idx in range(2):
                shift[iax] = drxns[idx]
                nb = idx_t0[:] + shift[:]
                # If point is outside the domain skip it
                # if np.any(nb < 0) or np.any(nb >= ns):
                if nb[0] < 0 or nb[1] < 0 or nb[2] < 0 or nb[0] >= ny or nb[1] >= nx or nb[2] >= nz:
                    continue
                if vv[nb[0], nb[1], nb[2]] > 0.0:
                    order[idx] = 1
                    fdt[idx] = drxns[idx] * (tt[nb[0], nb[1], nb[2]] - tt0) * ds_inv[iax]
                else:
                    order[idx] = 0
            # Selecting upwind derivative 
            shift.fill(0)
            if fdt[0] > -fdt[1] and order[0] > 0:
                idrx[iax], shift[iax] = -1, -1
            elif fdt[0] <= -fdt[1] and order[1] > 0:
                idrx[iax], shift[iax] = 1, 1
            else:
                idrx[iax] = 0
            nb = idx_t0[:] + shift[:]
            # Computing t0 space derivative
            fdt0[iax] = idrx[iax] * (tt[nb[0], nb[1], nb[2]] - tt0) * ds_inv[iax] * ds_inv[iax]

        # Checking traveltime values of neighbourning points
        tty = tt[idx_t0[0] + idrx[0], idx_t0[1], idx_t0[2]]
        ttx = tt[idx_t0[0], idx_t0[1] + idrx[1], idx_t0[2]]
        ttz = tt[idx_t0[0], idx_t0[1], idx_t0[2] + idrx[2]]
                                                            
        # Using single stencil along z direction to update value
        if ttx > tt0 and tty > tt0:
            denom = - 2.0 * idrx[2] * fdt0[2]
            if abs(denom) > 0.0:
                delta_tt[idx_t0[0], idx_t0[1], idx_t0[2]] += (- idrx[2] * 2.0 * fdt0[2] * delta_tt[idx_t0[0], idx_t0[1], idx_t0[2] + idrx[2]] 
                                                              + delta_v_scaled[idx_t0[0], idx_t0[1], idx_t0[2]]) / denom
        # Using single stencil along x direction to update value
        elif tty > tt0 and ttz > tt0:
            denom = - 2.0 * idrx[1] * fdt0[1]
            if abs(denom) > 0.0:
                delta_tt[idx_t0[0], idx_t0[1], idx_t0[2]] += (- idrx[1] * 2.0 * fdt0[1] * delta_tt[idx_t0[0], idx_t0[1] + idrx[1], idx_t0[2]]
                                                              + delta_v_scaled[idx_t0[0], idx_t0[1], idx_t0[2]]) / denom
        # Using single stencil along y direction to update value
        elif ttx > tt0 and ttz > tt0:
            denom = - 2.0 * idrx[0] * fdt0[0]
            if abs(denom) > 0.0:
                delta_tt[idx_t0[0], idx_t0[1], idx_t0[2]] += (- idrx[0] * 2.0 * fdt0[0] * delta_tt[idx_t0[0] + idrx[0], idx_t0[1], idx_t0[2]]
                                                              + delta_v_scaled[idx_t0[0], idx_t0[1], idx_t0[2]]) / denom
        # Using single stencil along x-y direction to update value
        elif ttz > tt0:
            denom = - 2.0 * (idrx[0] * fdt0[0] + idrx[1] * fdt0[1])
            if abs(denom) > 0.0:
                delta_tt[idx_t0[0], idx_t0[1], idx_t0[2]] += (- idrx[0] * 2.0 * fdt0[0] * delta_tt[idx_t0[0] + idrx[0], idx_t0[1], idx_t0[2]] +
                                                              - idrx[1] * 2.0 * fdt0[1] * delta_tt[idx_t0[0], idx_t0[1] + idrx[1], idx_t0[2]] +
                                                              delta_v_scaled[idx_t0[0], idx_t0[1], idx_t0[2]]) / denom
        # Using single stencil along x-z direction to update value
        elif tty > tt0:
            denom = - 2.0 * (idrx[1] * fdt0[1] + idrx[2] * fdt0[2])
            if abs(denom) > 0.0:
                delta_tt[idx_t0[0], idx_t0[1], idx_t0[2]] += (- idrx[1] * 2.0 * fdt0[1] * delta_tt[idx_t0[0], idx_t0[1] + idrx[1], idx_t0[2]] +
                                                              - idrx[2] * 2.0 * fdt0[2] * delta_tt[idx_t0[0], idx_t0[1], idx_t0[2] + idrx[2]] +
                                                              delta_v_scaled[idx_t0[0], idx_t0[1], idx_t0[2]]) / denom
        # Using single stencil along y-z direction to update value
        elif ttx > tt0:
            denom = - 2.0 * (idrx[0] * fdt0[0] + idrx[2] * fdt0[2])
            if abs(denom) > 0.0:
                delta_tt[idx_t0[0], idx_t0[1], idx_t0[2]] += (- idrx[0] * 2.0 * fdt0[0] * delta_tt[idx_t0[0] + idrx[0], idx_t0[1], idx_t0[2]] +
                                                              - idrx[2] * 2.0 * fdt0[2] * delta_tt[idx_t0[0], idx_t0[1], idx_t0[2] + idrx[2]] +
                                                              delta_v_scaled[idx_t0[0], idx_t0[1], idx_t0[2]]) / denom
        else:
            denom = - 2.0 * (idrx[0] * fdt0[0] + idrx[1] * fdt0[1] + idrx[2] * fdt0[2])
            if abs(denom) > 0.0:
                delta_tt[idx_t0[0], idx_t0[1], idx_t0[2]] += (- idrx[0] * 2.0 * fdt0[0] * delta_tt[idx_t0[0] + idrx[0], idx_t0[1], idx_t0[2]] +
                                                              - idrx[1] * 2.0 * fdt0[1] * delta_tt[idx_t0[0], idx_t0[1] + idrx[1], idx_t0[2]] +
                                                              - idrx[2] * 2.0 * fdt0[2] * delta_tt[idx_t0[0], idx_t0[1], idx_t0[2] + idrx[2]] +
                                                              delta_v_scaled[idx_t0[0], idx_t0[1], idx_t0[2]]) / denom
    return


@jit(nopython=True, cache=True)
def select_upwind_der3D(tt, idx_t0, vv, ds_inv, iax):
    """Find upwind derivative along iax"""
    ny = vv.shape[0]
    nx = vv.shape[1]
    nz = vv.shape[2]
    nb = np.zeros(3, dtype=np.int64)
    shift = np.zeros(3, dtype=np.int64)
    drxns = [-1, 1]
    fdt = np.zeros(2)
    order = np.zeros(2, dtype=np.int64)
    
    # Computing derivative for the neighboring points along iax
    for idx in range(2):
        shift[iax] = drxns[idx]
        nb[:] = idx_t0[:] + shift[:]
        # If point is outside the domain skip it
        # if np.any(nb < 0) or np.any(nb >= ns):
        if nb[0] < 0 or nb[1] < 0 or nb[2] < 0 or nb[0] >= ny or nb[1] >= nx or nb[2] >= nz:
            continue
        if vv[nb[0], nb[1], nb[2]] > 0.0:
            order[idx] = 1
            fdt[idx] = drxns[idx] * (tt[nb[0], nb[1], nb[2]] - tt[idx_t0[0], idx_t0[1], idx_t0[2]]) * ds_inv[iax]
        else:
            order[idx] = 0
    # Selecting upwind derivative 
    if fdt[0] > -fdt[1] and order[0] > 0:
        fd, idrx = fdt[0], -1
    elif fdt[0] <= -fdt[1] and order[1] > 0:
        fd, idrx = fdt[1], 1
    else:
        fd, idrx = 0.0, 0
    return fd, idrx

# Adjoint operator
@jit(nopython=True, cache=True)
def FMM_tt_lin_adj3D(delta_v, delta_tt, vv, tt, tt_idx, dy, dx, dz):
    """Fast-marching method linearized forward"""
    ny = delta_v.shape[0]
    nx = delta_v.shape[1]
    nz = delta_v.shape[2]
    drxns = [-1, 1]
    dy_inv = 1.0 / dy
    dx_inv = 1.0 / dx
    dz_inv = 1.0 / dz
    ds_inv = np.array([dy_inv, dx_inv, dz_inv])
    
    # Internal variables
    shift = np.zeros(3, dtype=np.int64)
    nbrs = np.zeros((6,3), dtype=np.int64)
    fdt_nb = np.zeros(6)
    order_nb = np.zeros(6, dtype=np.int64)
    idrx_nb = np.zeros(6, dtype=np.int64)
    
    # Looping over all indices to solve linear equations from increasing traveltime values
    for kk in range(tt_idx.shape[0]):
        idx_t0 = tt_idx[kk]
        tt0 =  tt[idx_t0[0], idx_t0[1], idx_t0[2]] 
        # If T = 0 or v = 0, then assuming zero to avoid singularity
        if tt0 == 0.0 or vv[idx_t0[0], idx_t0[1], idx_t0[2]] == 0.0:
            continue
        
        # Creating indices of neighbouring points
        # Order left/right bottom/top
        inbr = 0
        for iax in range(3):
            shift.fill(0)
            for idx in range(2):
                shift[iax] = drxns[idx]
                nbrs[inbr][:] = idx_t0[:] + shift[:]
                inbr += 1
        
        # Looping over neighbouring points
        fdt_nb.fill(0)
        idrx_nb.fill(0)
        for ib, nb in enumerate(nbrs):
            # Point outside of modeling domain
            if nb[0] < 0 or nb[1] < 0 or nb[2] < 0 or nb[0] >= ny or nb[1] >= nx or nb[2] >= nz:
                order_nb[ib] = 0
                continue
            # Point with lower traveltime compared to current point
            if tt0 > tt[nb[0], nb[1], nb[2]]:
                order_nb[ib] = 0
                continue
            order_nb[ib] = 1
            # Getting derivative along given axis
            if ib in [0,1]:
                iax = 0
            elif ib in [2,3]:
                iax = 1
            elif ib in [4,5]:
                iax = 2
            fdt_nb[ib], idrx_nb[ib] = select_upwind_der3D(tt, nb, vv, ds_inv, iax)
            # Removing point if derivative at nb did not use idx_t0
            if ib in [0,1]:
                # Checking y direction
                if idx_t0[0] != nb[0] + idrx_nb[ib]:
                    fdt_nb[ib], idrx_nb[ib] = 0.0, 0
            elif ib in [2,3]:
                # Checking x direction
                if idx_t0[1] != nb[1] + idrx_nb[ib]:
                    fdt_nb[ib], idrx_nb[ib] = 0.0, 0
            else:
                # Checking z direction
                if idx_t0[2] != nb[2] + idrx_nb[ib]:
                    fdt_nb[ib], idrx_nb[ib] = 0.0, 0
        
        # Updating delta_v according to stencil
        fdt_nb *= -idrx_nb
        fdt0 = 0.0
        fdt_nb[0] *= dy_inv
        fdt_nb[1] *= dy_inv
        fdt_nb[2] *= dx_inv
        fdt_nb[3] *= dx_inv
        fdt_nb[4] *= dz_inv
        fdt_nb[5] *= dz_inv

        # Only z
        if order_nb[0] > 0 and order_nb[1] > 0 and order_nb[2] > 0 and order_nb[3] > 0:
            fdt0, idrx0 = select_upwind_der3D(tt, idx_t0, vv, ds_inv, 2)
            fdt0 *= np.sign(idrx0) * dz_inv
        # Only x
        elif order_nb[0] > 0 and order_nb[1] > 0 and order_nb[4] > 0 and order_nb[5] > 0:
            fdt0, idrx0 = select_upwind_der3D(tt, idx_t0, vv, ds_inv, 1)
            fdt0 *= np.sign(idrx0) * dx_inv
        # Only y
        elif order_nb[2] > 0 and order_nb[3] > 0 and order_nb[4] > 0 and order_nb[5] > 0:
            fdt0, idrx0 = select_upwind_der3D(tt, idx_t0, vv, ds_inv, 0)
            fdt0 *= np.sign(idrx0) * dy_inv
        # Only x-y
        elif order_nb[4] > 0 and order_nb[5] > 0:
            fdt0y, idrx0y = select_upwind_der3D(tt, idx_t0, vv, ds_inv, 0)
            fdt0x, idrx0x = select_upwind_der3D(tt, idx_t0, vv, ds_inv, 1)
            # Necessary to consider correct stencil central value
            if tt0 < tt[idx_t0[0] + idrx0y, idx_t0[1], idx_t0[2]]: 
                fdt0y, idrx0y = 0.0, 0
            if tt0 < tt[idx_t0[0], idx_t0[1] + idrx0x, idx_t0[2]]: 
                fdt0x, idrx0x = 0.0, 0
            fdt0 = idrx0y * fdt0y * dy_inv + idrx0x * fdt0x * dx_inv
        # Only x-z
        elif order_nb[0] > 0 and order_nb[1] > 0:
            fdt0x, idrx0x = select_upwind_der3D(tt, idx_t0, vv, ds_inv, 1)
            fdt0z, idrx0z = select_upwind_der3D(tt, idx_t0, vv, ds_inv, 2)
            # Necessary to consider correct stencil central value
            if tt0 < tt[idx_t0[0], idx_t0[1] + idrx0x, idx_t0[2]]: 
                fdt0x, idrx0x = 0.0, 0
            if tt0 < tt[idx_t0[0], idx_t0[1], idx_t0[2] + idrx0z]: 
                fdt0z, idrx0z = 0.0, 0
            fdt0 = idrx0x * fdt0x * dx_inv + idrx0z * fdt0z * dz_inv
        # Only y-z
        elif order_nb[2] > 0 and order_nb[3] > 0:
            fdt0y, idrx0y = select_upwind_der3D(tt, idx_t0, vv, ds_inv, 0)
            fdt0z, idrx0z = select_upwind_der3D(tt, idx_t0, vv, ds_inv, 2)
            # Necessary to consider correct stencil central value
            if tt0 < tt[idx_t0[0] + idrx0y, idx_t0[1], idx_t0[2]]: 
                fdt0y, idrx0y = 0.0, 0
            if tt0 < tt[idx_t0[0], idx_t0[1], idx_t0[2] + idrx0z]: 
                fdt0z, idrx0z = 0.0, 0
            fdt0 = idrx0y * fdt0y * dy_inv + idrx0z * fdt0z * dz_inv
        else:
            fdt0y, idrx0y = select_upwind_der3D(tt, idx_t0, vv, ds_inv, 0)
            fdt0x, idrx0x = select_upwind_der3D(tt, idx_t0, vv, ds_inv, 1)
            fdt0z, idrx0z = select_upwind_der3D(tt, idx_t0, vv, ds_inv, 2)
            # Necessary to consider correct stencil central value
            if tt0 < tt[idx_t0[0] + idrx0y, idx_t0[1], idx_t0[2]]: 
                fdt0y, idrx0y = 0.0, 0
            if tt0 < tt[idx_t0[0], idx_t0[1] + idrx0x, idx_t0[2]]: 
                fdt0x, idrx0x = 0.0, 0
            if tt0 < tt[idx_t0[0], idx_t0[1], idx_t0[2] + idrx0z]: 
                fdt0z, idrx0z = 0.0, 0
            fdt0 = idrx0y * fdt0y * dy_inv + idrx0x * fdt0x * dx_inv + idrx0z * fdt0z * dz_inv
        
        # Update delta_v value
        if abs(fdt0) > 0.0:
            delta_v[idx_t0[0], idx_t0[1], idx_t0[2]] -= (  fdt_nb[0] * delta_v[idx_t0[0]-order_nb[0], idx_t0[1], idx_t0[2]] 
                                                        + fdt_nb[1] * delta_v[idx_t0[0]+order_nb[1], idx_t0[1], idx_t0[2]] 
                                                        + fdt_nb[2] * delta_v[idx_t0[0], idx_t0[1]-order_nb[2], idx_t0[2]] 
                                                        + fdt_nb[3] * delta_v[idx_t0[0], idx_t0[1]+order_nb[3], idx_t0[2]] 
                                                        + fdt_nb[4] * delta_v[idx_t0[0], idx_t0[1], idx_t0[2]-order_nb[4]] 
                                                        + fdt_nb[5] * delta_v[idx_t0[0], idx_t0[1], idx_t0[2]+order_nb[5]] 
                                                        - 0.5 * delta_tt[idx_t0[0], idx_t0[1], idx_t0[2]]) / fdt0
    
    # Scaling the velocity perturbation
    delta_v[:] = 2.0 * delta_v / (vv * vv * vv)
            
    return

def compute_travel_timeLin(vel, ishot, oy, ox, oz, dy, dx, dz, SouPos, Acc_Inj, TTsrc=None):
    """Function to compute traveltime in parallel"""
    velocity = pykonal.fields.ScalarField3D(coord_sys="cartesian")
    velocity.min_coords = oy, ox, oz
    velocity.node_intervals = dy, dx, dz
    velocity.npts = vel.shape[0], vel.shape[1], vel.shape[2]
    if SouPos.ndim == 2:
        # Single point source
        if Acc_Inj:
            # Set Eikonal solver
            solver_ek = pykonal.solver.PointSourceSolver(coord_sys="cartesian")
            solver_ek.vv.min_coords = velocity.min_coords
            solver_ek.vv.node_intervals = velocity.node_intervals
            solver_ek.vv.npts = velocity.npts
            solver_ek.vv.values[:] = vel
            # Setting source position (ys,xs,zs)
            solver_ek.src_loc = [SouPos[ishot,1],SouPos[ishot,0],SouPos[ishot,2]] 
        else:
            # Set Eikonal solver
            solver_ek = pykonal.EikonalSolver(coord_sys="cartesian")
            solver_ek.vv.min_coords = velocity.min_coords
            solver_ek.vv.node_intervals = velocity.node_intervals
            solver_ek.vv.npts = velocity.npts
            solver_ek.vv.values[:] = vel
            # Initial conditions
            solver_ek.tt.values[:] = np.inf
            solver_ek.known[:] = False
            solver_ek.unknown[:] = True
            eq_iz = int((SouPos[ishot,2]-oz)/dz + 0.5)
            eq_iy = int((SouPos[ishot,1]-oy)/dy + 0.5)
            eq_ix = int((SouPos[ishot,0]-ox)/dx + 0.5)
            src_idx = (eq_iy, eq_ix, eq_iz)
            solver_ek.tt.values[src_idx] = 0.0
            solver_ek.unknown[src_idx] = False
            solver_ek.trial.push(*src_idx)
    else:
        # Multiple source points
        npnt_src = SouPos.shape[2]
        for iPnt in range(npnt_src):
            eq_iz = int((SouPos[ishot,2,iPnt]-oz)/dz + 0.5)
            eq_iy = int((SouPos[ishot,1,iPnt]-oy)/dy + 0.5)
            eq_ix = int((SouPos[ishot,0,iPnt]-ox)/dx + 0.5)
            src_idx = (eq_iy, eq_ix, eq_iz)
            solver_ek.tt.values[src_idx] = 0.0 if TTsrc is None else TTsrc[iPnt]
            solver_ek.unknown[src_idx] = False
            solver_ek.trial.push(*src_idx)
    # Solving Eikonal equation
    solver_ek.solve()
    return solver_ek.tt.values

def solve_linearized_fwd(vel0, delta_v, ishot, dy, dx, dz, oy, ox, oz, tt0, tt_idx, RecPos):
    """Function to solve linearized problem"""
    # Sorting traveltime in ascending order
    tt_idx = sorting3D(tt0, tt_idx)
    delta_tt = np.zeros_like(vel0)
    FMM_tt_lin_fwd3D(delta_v, delta_tt, vel0, tt0, tt_idx, dy, dx, dz)
    data_tt_lin = extract_tt_3D_FWD(delta_tt, RecPos[:,1], RecPos[:,0], RecPos[:,2], oy, ox, oz, dy, dx, dz)
    return data_tt_lin

def solve_linearized_adj(vel0, data, ishot, oy, ox, oz, dy, dx, dz, tt0, tt_idx, RecPos):
    delta_tt = np.zeros_like(vel0)
    delta_v = np.zeros_like(vel0)
    # Sorting traveltime in descending order
    tt_idx = sorting3D(tt0, tt_idx, ordering="d")
    # Injecting traveltime to correct grid positions
    for iRec in range(RecPos.shape[0]):
        wy = (RecPos[iRec, 1] - oy) / dy
        wx = (RecPos[iRec, 0] - ox) / dx
        wz = (RecPos[iRec, 2] - oz) / dz
        iy = int(wy)
        ix = int(wx)
        iz = int(wz)
        # Interpolation weights
        wy -= iy
        wx -= ix
        wz -= iz
        delta_tt[iy,ix,iz]       += data[ishot, iRec] * (1.0 - wy)*(1.0 - wx)*(1.0 - wz) 
        delta_tt[iy,ix,iz+1]     += data[ishot, iRec] * (1.0 - wy)*(1.0 - wx)*(wz) 
        delta_tt[iy,ix+1,iz+1]   += data[ishot, iRec] * (1.0 - wy)*(wx)*(wz) 
        delta_tt[iy,ix+1,iz]     += data[ishot, iRec] * (1.0 - wy)*(wx)*(1.0 - wz)  
        delta_tt[iy+1,ix,iz]     += data[ishot, iRec] * (wy)*(1.0 - wx)*(1.0 - wz) 
        delta_tt[iy+1,ix,iz+1]   += data[ishot, iRec] * (wy)*(1.0 - wx)*(wz) 
        delta_tt[iy+1,ix+1,iz+1] += data[ishot, iRec] * (wy)*(wx)*(wz) 
        delta_tt[iy+1,ix+1,iz]   += data[ishot, iRec] * (wy)*(wx)*(1.0 - wz)  
    FMM_tt_lin_adj3D(delta_v, delta_tt, vel0, tt0, tt_idx, dy, dx, dz)
    return delta_v

def solve_linearized_adj_model(vel0: np.ndarray, data: np.ndarray, ishot: int, dx: float, dz: float, tt0: np.ndarray, tt_idx: np.ndarray, SouPos, RecPos: np.ndarray, fno_model, y_normalizer):
    print("MODEL")
    XX, ZZ = np.meshgrid(np.linspace(0, 1, 100), np.linspace(0, 1, 100))

    fno_model = torch.load('model-mode-8-width-64-epoch-1000', map_location=torch.device('cpu'))
    filename = '../Data/DatVel30_Sou100_Rec100_Dim100x100_Downsampled.npz'
    with np.load(filename, allow_pickle=True) as fid:
        train_data = EikonalDataset(fid, 'train')
    x_train_vels = train_data.vels

    # x_normalizer = UnitGaussianNormalizer(torch.tensor(np.float32(vel0)))
    x_normalizer = UnitGaussianNormalizer(x_train_vels)

    tmp1 = x_normalizer.encode(np.float32(vel0))

    source_loc_gaussian=torch.tensor(gaussian_function(XX, ZZ, np.asarray(list(SouPos[0])[::-1]).flatten()/100))
    tmp2 = np.float32(source_loc_gaussian)

    # print("max: " + str(np.amax(np.float32(vel0))))
    # print("min: " + str(np.amin(np.float32(vel0))))

    out = np.zeros((100, 100))

    for iRec in range(RecPos.shape[0]):
        rec_loc_gaussian=torch.tensor(gaussian_function(XX, ZZ, np.asarray(list(RecPos[iRec])[::-1]).flatten()/100))
        # tmp1 = np.float32(vel0) - 1
        tmp3 = np.float32(rec_loc_gaussian)


        tmp = np.asarray([tmp1, tmp2, tmp3], dtype=float)
        x = torch.from_numpy(np.expand_dims(np.transpose(tmp), axis=0))

        x = x.type(torch.float32)

        x_np0 = x[0,:,:,0].cpu().numpy()
        x_np1 = x[0,:,:,1].cpu().numpy()
        x_np2 = x[0,:,:,2].cpu().numpy()

        print(x.shape)
        
        # print("max: " + str(np.amax(x_np0)))
        # print("min: " + str(np.amin(x_np0)))

        out_sample1 = y_normalizer.decode(fno_model(x).squeeze(-1))
        # out_sample1 = fno_model(x).squeeze(-1)
        out_sample1 = out_sample1.detach().numpy()[0, :, :] 

        out += out_sample1

        fig, axs = plt.subplots(1, 4, figsize=(16, 4))

        # Display the images
        axs[0].imshow(x_np0, cmap='jet_r')  # you can change the colormap if needed
        axs[0].set_title("x_sample")
        axs[0].axis('off')  # Hide the axis values

        axs[1].imshow(x_np1, cmap='jet_r')  # you can change the colormap if needed
        axs[1].set_title("source")
        axs[1].axis('off')  # Hide the axis values

        axs[2].imshow(x_np2, cmap='jet_r')  # you can change the colormap if needed
        axs[2].set_title("receiver")
        axs[2].axis('off')  # Hide the axis values

        im=axs[3].imshow(out_sample1, cmap='gray')  # you can change the colormap if needed
        fig.colorbar(im, ax=axs[3])
        axs[3].set_title("kernel")
        axs[3].axis('off')  # Hide the axis values

        plt.tight_layout()
        plt.show()

    return out

class EikonalTT_lin_3D(pyOp.Operator):

    def __init__(self, vel, tt_data, oy, ox, oz, dy, dx, dz, SouPos, RecPos, TTsrc=None, tt_maps=None, verbose=False, **kwargs):
        """3D Eikonal-equation traveltime prediction operator"""
        # Setting Domain and Range of the operator
        self.setDomainRange(vel, tt_data)
        # Setting acquisition geometry
        self.nSou = SouPos.shape[0]
        self.nRec = RecPos.shape[0]
        self.SouPos = SouPos.copy()
        self.RecPos = RecPos.copy()
        # Get velocity array
        velNd = vel.getNdArray()
        # Getting number of threads to run the modeling code
        self.nthrs = min(self.nSou, Ncores, kwargs.get("nthreads", Ncores))
        if TTsrc is not None:
            if len(TTsrc) != self.nSou:
                raise ValueError("Number of initial traveltime (len(TTsrc)=%s) inconsistent with number of sources (%s)"%(len(TTsrc),self.nSou))
        else:
            TTsrc = [None]*self.nSou
        # Accurate injection of initial conditions
        self.Acc_Inj = kwargs.get("Acc_Inj", False)
        self.TTsrc = TTsrc # Traveltime vector for distributed sources
        dataShape = tt_data.shape
        self.oy = oy
        self.ox = ox
        self.oz = oz
        self.dy = dy
        self.dx = dx
        self.dz = dz
        self.ncomp = vel.shape[0]
        self.ny = vel.shape[1]
        self.nx = vel.shape[2]
        self.nz = vel.shape[3]
        # Use smallest possible domain (Ginsu knives)
        self.ginsu = kwargs.get("ginsu", False)
        buffer = kwargs.get("buffer", 2.0) # By default 2.0 km
        self.bufferX = int(buffer/dx)
        self.bufferY = int(buffer/dy)
        self.bufferZ = int(buffer/dz)
        self.xAxis = np.linspace(ox, ox+(self.nx-1)*dx, self.nx)
        self.yAxis = np.linspace(oy, oy+(self.ny-1)*dy, self.ny)
        self.zAxis = np.linspace(oz, oz+(self.nz-1)*dz, self.nz)
        if dataShape[0] != self.nSou*self.ncomp:
            raise ValueError("Number of sources inconsistent with traveltime vector (data_shape[0])")
        if dataShape[1] != self.nRec:
            raise ValueError("Number of receivers inconsistent with traveltime vector (data_shape[1])")
        # Internal velocity model
        self.vel0 = vel.clone()
        # Verbosity level
        self.verbose = verbose
        # General unsorted traveltime indices
        idx_1d = np.arange(velNd[0,:,:,:].size)
        idy,idx,idz = np.unravel_index(idx_1d, velNd[0,:,:,:].shape)
        self.tt_idx = np.array([idy,idx,idz]).T
        # Traveltime maps
        if tt_maps is None:
            self.tt_maps = []
            for _ in range(self.nSou*self.ncomp):
                self.tt_maps.append(np.zeros_like(velNd[0,:,:,:]))
        else:
            self.tt_maps = tt_maps
        # Variable for updating the traveltimes
        self.TT_update = True

    def reset_tt_maps(self):
        """Function to zero-out tt_maps variable to recompute traveltime maps"""
        for ishot in range(self.nSou*self.ncomp):
            self.tt_maps[ishot].fill(0.0)
      
    def forward(self, add, model, data):
        """Forward linearized traveltime prediction"""
        self.checkDomainRange(model, data)
        if not add: 
            data.zero()
        dataNd = data.getNdArray()
        modelNd = model.getNdArray()
        vel0Nd = self.vel0.getNdArray()
        ###################################
        # Computing background traveltime #
        ###################################
        # if np.any([not np.any(self.tt_maps[ishot]) for ishot in range(self.nSou)]):
        if self.TT_update:
            for icomp in range(self.ncomp):
                if self.ginsu:
                    minX = np.zeros(self.nSou, dtype=int)
                    maxX = np.zeros(self.nSou, dtype=int)
                    minY = np.zeros(self.nSou, dtype=int)
                    maxY = np.zeros(self.nSou, dtype=int)
                    minZ = np.zeros(self.nSou, dtype=int)
                    maxZ = np.zeros(self.nSou, dtype=int)
                    for ishot in range(self.nSou):
                        minX[ishot] =  max(0, min(np.argmin(abs(self.xAxis-self.SouPos[ishot,0])), np.argmin(abs(self.xAxis-self.RecPos[:,0].min())))-self.bufferX)
                        maxX[ishot] =  min(self.nx, max(np.argmin(abs(self.xAxis-self.SouPos[ishot,0])), np.argmin(abs(self.xAxis-self.RecPos[:,0].max())))+self.bufferX)
                        minY[ishot] =  max(0, min(np.argmin(abs(self.yAxis-self.SouPos[ishot,1])), np.argmin(abs(self.yAxis-self.RecPos[:,1].min())))-self.bufferY)
                        maxY[ishot] =  min(self.ny, max(np.argmin(abs(self.yAxis-self.SouPos[ishot,1])), np.argmin(abs(self.yAxis-self.RecPos[:,1].max())))+self.bufferY)
                        minZ[ishot] =  max(0, min(np.argmin(abs(self.zAxis-self.SouPos[ishot,2])), np.argmin(abs(self.zAxis-self.RecPos[:,2].min())))-self.bufferZ)
                        maxZ[ishot] =  min(self.nz, max(np.argmin(abs(self.zAxis-self.SouPos[ishot,2])), np.argmin(abs(self.zAxis-self.RecPos[:,2].max())))+self.bufferZ)
                    result = Parallel(n_jobs=self.nthrs, verbose=self.verbose)(delayed(compute_travel_timeLin)(vel0Nd[icomp,minY[ishot]:maxY[ishot],minX[ishot]:maxX[ishot],minZ[ishot]:maxZ[ishot]], ishot, self.yAxis[minY[ishot]], self.xAxis[minX[ishot]], self.zAxis[minZ[ishot]], self.dy, self.dx, self.dz, self.SouPos, self.Acc_Inj, self.TTsrc[ishot]) for ishot in range(self.nSou))
                else:
                    result = Parallel(n_jobs=self.nthrs, verbose=self.verbose)(delayed(compute_travel_timeLin)(vel0Nd[icomp,:,:,:], ishot, self.oy, self.ox, self.oz, self.dy, self.dx, self.dz, self.SouPos, self.Acc_Inj, self.TTsrc[ishot]) for ishot in range(self.nSou))
                for ishot in range(self.nSou):
                    if self.ginsu:
                        self.tt_maps[ishot+icomp*self.nSou][minY[ishot]:maxY[ishot],minX[ishot]:maxX[ishot],minZ[ishot]:maxZ[ishot]] = result[ishot]
                    else:
                        self.tt_maps[ishot+icomp*self.nSou][:] = result[ishot]
            self.TT_update = False
        ###################################
        # Computing linearized traveltime #
        ###################################
        for icomp in range(self.ncomp):
            result = Parallel(n_jobs=self.nthrs, verbose=self.verbose)(delayed(solve_linearized_fwd)(vel0Nd[icomp,:,:,:], modelNd[icomp,:,:,:], ishot, self.dy, self.dx, self.dz, self.oy, self.ox, self.oz, self.tt_maps[ishot+icomp*self.nSou], self.tt_idx, self.RecPos) for ishot in range(self.nSou))
            for ishot in range(self.nSou):
                dataNd[ishot+icomp*self.nSou,:] += result[ishot]
        return
    
    def adjoint(self, add, model, data, use_original=True):
        """Adjoint linearized traveltime prediction"""

        if use_original:
            func = solve_linearized_adj
        else:  
            func = solve_linearized_adj_model

        fno_model = torch.load('model-mode-8-width-64-epoch-1000', map_location=torch.device('cpu'))
        filename = '../Data/DatVel30_Sou100_Rec100_Dim100x100_Downsampled.npz'
        with np.load(filename, allow_pickle=True) as fid:
            train_data = EikonalDataset(fid, 'train')
        x_train_vels = train_data.vels

        y_train = train_data.kernels
        y_normalizer = UnitGaussianNormalizer(y_train)
        fno_model.eval()

        self.checkDomainRange(model, data)
        if not add: 
            model.zero()
        dataNd = data.getNdArray()
        modelNd = model.getNdArray()
        vel0Nd = self.vel0.getNdArray()
        ###################################
        # Computing background traveltime #
        ###################################
        # if np.any([not np.any(self.tt_maps[ishot]) for ishot in range(self.nSou)]):
        if self.TT_update:
            for icomp in range(self.ncomp):
                if self.ginsu:
                    minX = np.zeros(self.nSou, dtype=int)
                    maxX = np.zeros(self.nSou, dtype=int)
                    minY = np.zeros(self.nSou, dtype=int)
                    maxY = np.zeros(self.nSou, dtype=int)
                    minZ = np.zeros(self.nSou, dtype=int)
                    maxZ = np.zeros(self.nSou, dtype=int)
                    for ishot in range(self.nSou):
                        minX[ishot] =  max(0, min(np.argmin(abs(self.xAxis-self.SouPos[ishot,0])), np.argmin(abs(self.xAxis-self.RecPos[:,0].min())))-self.bufferX)
                        maxX[ishot] =  min(self.nx, max(np.argmin(abs(self.xAxis-self.SouPos[ishot,0])), np.argmin(abs(self.xAxis-self.RecPos[:,0].max())))+self.bufferX)
                        minY[ishot] =  max(0, min(np.argmin(abs(self.yAxis-self.SouPos[ishot,1])), np.argmin(abs(self.yAxis-self.RecPos[:,1].min())))-self.bufferY)
                        maxY[ishot] =  min(self.ny, max(np.argmin(abs(self.yAxis-self.SouPos[ishot,1])), np.argmin(abs(self.yAxis-self.RecPos[:,1].max())))+self.bufferY)
                        minZ[ishot] =  max(0, min(np.argmin(abs(self.zAxis-self.SouPos[ishot,2])), np.argmin(abs(self.zAxis-self.RecPos[:,2].min())))-self.bufferZ)
                        maxZ[ishot] =  min(self.nz, max(np.argmin(abs(self.zAxis-self.SouPos[ishot,2])), np.argmin(abs(self.zAxis-self.RecPos[:,2].max())))+self.bufferZ)
                    result = Parallel(n_jobs=self.nthrs, verbose=self.verbose)(delayed(compute_travel_timeLin)(vel0Nd[icomp,minY[ishot]:maxY[ishot],minX[ishot]:maxX[ishot],minZ[ishot]:maxZ[ishot]], ishot, self.yAxis[minY[ishot]], self.xAxis[minX[ishot]], self.zAxis[minZ[ishot]], self.dy, self.dx, self.dz, self.SouPos, self.Acc_Inj, self.TTsrc[ishot]) for ishot in range(self.nSou))
                else:
                    result = Parallel(n_jobs=self.nthrs, verbose=self.verbose)(delayed(compute_travel_timeLin)(vel0Nd[icomp,:,:,:], ishot, self.oy, self.ox, self.oz, self.dy, self.dx, self.dz, self.SouPos, self.Acc_Inj, self.TTsrc[ishot]) for ishot in range(self.nSou))
                for ishot in range(self.nSou):
                    if self.ginsu:
                        self.tt_maps[ishot+icomp*self.nSou][minY[ishot]:maxY[ishot],minX[ishot]:maxX[ishot],minZ[ishot]:maxZ[ishot]] = result[ishot]
                    else:
                        self.tt_maps[ishot+icomp*self.nSou][:] = result[ishot]
            self.TT_update = False
        ###################################
        # Computing velocity perturbation #
        ###################################

       if np.any([not np.any(self.tt_maps[ishot]) for ishot in range(self.ns)]):
            for shot in range(self.ns):
                _, self.tt_maps[shot] = _compute_traveltime(self.vel[:], self.SouPos, self.RecPos, self.dx, self.dz, dummyData, shot)
        ###################################
        # Computing velocity perturbation #
        ###################################
        for shot in range(self.ns):
            model[:] += func(self.vel[:], data[:], shot, self.dx, self.dz, self.tt_maps[shot], self.tt_idx, self.SouPos, self.RecPos, fno_model, y_normalizer)

        return
    
    def set_vel(self, vel):
        """Function to set background velocity model"""
        if self.vel0.isDifferent(vel):
            self.TT_update = True
        self.vel0.copy(vel)
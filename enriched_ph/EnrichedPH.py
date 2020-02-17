import dionysus as d
import diode

import numpy as np
from math import sqrt, factorial
from scipy import sparse
from scipy.sparse.linalg import lsqr
from scipy import optimize
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import squareform

import sys
import time
import pickle

import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


mpl.rcParams['font.size'] = 24

def volume(Points):
    """
    Computes volume
    For triangle:
    Heron's formula
    For tetrahedron:
    Wikipedia: tetrahedron
    For higher dimension:
    Cayley-Menger Determinant
    """
    if len(Points) == 1:
        return( 1.0 )
    elif len(Points) == 2:
        return( np.linalg.norm(np.asarray(Points[0])-np.asarray(Points[1])) )
    elif len(Points) == 3:
        [A,B,C] = Points
        A = np.asarray(A); B = np.asarray(B); C = np.asarray(C);
        a = np.linalg.norm(B-C);
        b = np.linalg.norm(A-C);
        c = np.linalg.norm(A-B);
        p = (a+b+c)*0.5
        area = np.sqrt(p*(p-a)*(p-b)*(p-c))
        return( area )
    elif len(Points) == 4:
        [A,B,C,D] = Points
        A = np.asarray(A); B = np.asarray(B);
        C = np.asarray(C); D = np.asarray(D);
        mat = np.array([A-D, B-D, C-D])
        return( (1.0/6.0)*abs(np.linalg.det(mat)) )
    elif len(Points) > 4:
        simplex = np.array(Points)
        edges = np.subtract(simplex[:, None], simplex[None, :])
        ei_dot_ej = np.einsum('...k,...k->...', edges, edges)
        j = simplex.shape[0] - 1
        a = np.empty((j+2, j+2) + ei_dot_ej.shape[2:])
        a[1:, 1:] = ei_dot_ej
        a[0, 1:] = 1.0
        a[1:, 0] = 1.0
        a[0, 0] = 0.0
        a = np.moveaxis(a, (0, 1), (-2, -1))
        det = np.linalg.det(a)
        vol = np.sqrt((-1.0)**(j+1) / 2**j / factorial(j)**2 * det)
        return vol

class DefinedDistance:
    def __init__(self, dismat):
        self.dismat = dismat
    def __len__(self):
        return self.dismat.shape[0]
    def __call__(self, p1, p2):
        return self.dismat[p1,p2]

class PCF:
    """
    An object for persistent cohomology analysis of heterogeneous data.
    """
    def __init__(self, X, dmat=None):
        """
        :param X: point cloud data (n_pts, n_dim)
        :type X: class:`numpy.ndarray`
        :param dmat: distance matrix of point cloud data (n_pts, n_pts)
        :type dmat: class:`numpy.ndarray`
        """
        self.pts = X
        self.X = X
        if not dmat is None:
            self.dmat = dmat
        self.nd = X.shape[1]
        # (list of tuples) Simplices ordered by filtration
        self.f_simplices = []
        # (list of scalars) Filtration values of simplices with same order
        self.f_values = []
        # (list of scalars) Volume of simplices with same order
        self.f_volumes = []
        # The smoothed cocycles
        self.pcoh_cocycles = {}
        # (dic of numpy.ndarray) cobmats[1] is the cobounadry matrix of dim 1
        self.org_cocycles = {}
        # same structure as pcoh_cocycles but the original ones
        self.cobmats = {}
        # (dic of numpy.ndarray) weighted coboundary matrices
        self.wcobmats = {}
        # A dictionary for simplex indices. (0,1) is the simplex_dic[(0,1)] th 1-simplex
        self.simplex_dic = {}
        # nd_simplex_collection[1] is an ordered list of 1-simplices
        self.nd_simplex_collection = {}
        # nd_simplex_filt[1] is the corresponding filtration value of nd_simplex_collection[1]
        self.nd_simplex_filt = {}
        # nd_simplex_volume[1] is the corresponding volumes of nd_simplex_collection[1]
        self.nd_simplex_volume = {}

    def filtration(self, rule, nd = None, rips_max = None):
        """Run the filtration through dionysus.

        :param rule: the filtration rules. 1. 'alpha', alpha complex; 2. 'rips_euc', rips complex with Euclidean distance of point cloud; 3. 'rips_dmat', rips complex on the distance matrix)
        :type rule: str
        :param nd: max dimension of filtration
        :type nd: int, defaults to dimension of X
        :param rips_max: threshold value for rips filtration if rips is used
        :type rips_max: float
        """
        self.rips_max = rips_max
        if nd == None:
            self.phnd = self.nd
        else:
            self.phnd = nd
        if rule == 'alpha':
            self.f = d.Filtration( diode.fill_alpha_shapes(self.pts) )
            self.rule = 'alpha'
        elif rule == 'rips_euc':
            self.f = d.fill_rips(self.pts, self.phnd, rips_max)
            self.rule = 'rips_euc'
        elif rule == 'rips_dmat':
            self.f = d.fill_rips(squareform(self.dmat), self.phnd, rips_max)
            self.rule = 'rips_dmat'

        for s in self.f:
            if self.rule == 'alpha':
                self.f_values.append(sqrt(s.data))
            elif self.rule == 'rips_euc' or self.rule == 'rips_dmat':
                self.f_values.append(s.data)
            tmp_nodes = str(s).split(' ')[0][1:-1].split(',')
            nodes = []
            for tmp_node in tmp_nodes: nodes.append(int(tmp_node))
            self.f_simplices.append(tuple(np.sort(nodes).tolist()))

    def compute_volume(self, uniform=False):
        """Compute volumes for each simplex.

        :param uniform: whether use unit weight for all simplices, if not, geometric volume is used.
        :type uniform: boolean, default to False.
        """
        for i in range(len(self.f_simplices)):
            s = self.f_simplices[i]
            points = []
            for v in s:
                points.append(self.pts[v])
            self.f_volumes.append(volume(points))

    def pcoh(self, prime=547):
        """Run persistent cohomology through dionysus

        :param prime: the prime number for the coefficient field
        :type prime: int, defaults to 547
        """
        self.prime = prime
        self.p = d.cohomology_persistence(self.f, prime, True)

        self.dgms = d.init_diagrams(self.p, self.f)
        # self.dgms = []
        # for dim in range(self.phnd+1):
        #     tmp_dgm = []
        #     for pt in dgms[dim]:
        #         tmp_pt = []
        #         if self.rule == "alpha":
        #             birth = sqrt(pt.birth); death = sqrt(pt.death)
        #         else:
        #             birth = pt.birth; death = pt.death
        #         tmp_cocycle = str(p.cocycle(pt.data)).split('+')
        #         cocycle = []
        #         for tmp in tmp_cocycle:
        #             c,i = tmp.split('*')
        #             cocycle.append((int(c),int(i)))
        #         tmp_dgm.append([birth, death, cocycle])
        #     self.dgms.append(tmp_dgm)



    def collect_nd_simplices(self):
        """Collect the simplices of different dimensions separately.
        """
        for dim in range(self.nd+1):
            X_k = []; X_k_v = []; X_k_volume = []; i = 0;
            for j in range(len(self.f_simplices)):
                s = self.f_simplices[j];
                v = self.f_values[j]
                volume = self.f_volumes[j]
                if len(s) == dim+1:
                    X_k.append(s)
                    X_k_v.append(v)
                    X_k_volume.append(volume)
                    self.simplex_dic[tuple(s)] = i
                    i += 1
            self.nd_simplex_collection[dim] = X_k
            self.nd_simplex_filt[dim] = np.asarray(X_k_v)
            self.nd_simplex_volume[dim] = np.asarray(X_k_volume)

    def construct_weighted_combinatorial_laplacian(self, dim, filt_value, type):
        """Construct weighted combinatorial laplacian

        :param dim: the dimension of the laplacian
        :type dim: int
        :param filt_value: the filtration value for constructing the matrix
        :type filt_value: float
        :param type: full laplacian or only the lower part
        :type type: str
        """
        
        n_X_k = len(self.nd_simplex_collection[dim])
        n_X_k_minus_1 = len(self.nd_simplex_collection[dim-1])
        n_X_k_plus_1 = len(self.nd_simplex_collection[dim+1])
        if n_X_k > 0:
            ns_k = np.max(np.where(self.nd_simplex_filt[dim]<=filt_value)[0]) + 1
        else:
            ns_k = 0
        if n_X_k_plus_1 > 0:
            ns_k_plus_1 = np.max(np.concatenate((np.array([-1]), np.where(self.nd_simplex_filt[dim+1]<=filt_value)[0]), axis=0)) + 1
        else:
            ns_k_plus_1 = 0
        if n_X_k_minus_1 > 0:
            ns_k_minus_1 = np.max(np.where(self.nd_simplex_filt[dim-1]<=filt_value)[0]) + 1
        else:
            ns_k_minus_1 = 0

        I = np.arange(ns_k); V = np.ones(ns_k);
        left_mat = sparse.coo_matrix((V, (I,I)), shape=(ns_k, n_X_k))
        I = np.arange(ns_k_minus_1); V = np.ones(ns_k_minus_1);
        right_mat = sparse.coo_matrix((V, (I,I)), shape=(n_X_k_minus_1, ns_k_minus_1))
        cobmat_k = left_mat*self.wcobmats[dim]*right_mat

        I = np.arange(ns_k_plus_1); V = np.ones(ns_k_plus_1);
        left_mat = sparse.coo_matrix((V, (I,I)), shape=(ns_k_plus_1, n_X_k_plus_1))
        I = np.arange(ns_k); V = np.ones(ns_k);
        right_mat = sparse.coo_matrix((V, (I,I)), shape=(n_X_k, ns_k))
        cobmat_k_plus_1 = left_mat*self.wcobmats[dim+1]*right_mat

        if (not 0 in cobmat_k_plus_1.shape) and (type == "full"):
            wcLmat = cobmat_k_plus_1.transpose() * cobmat_k_plus_1 + cobmat_k * cobmat_k.transpose()
        else:
            wcLmat = cobmat_k * cobmat_k.transpose()

        return wcLmat

    def assemble_boundary_matrix(self, dim):
        """Construct the coboundary matrices of the final complex

        :param dim: dimension k, C^{k-1} -> C^k
        :type dim: int
        """
        X_k = self.nd_simplex_collection[dim]
        X_k_vol = self.nd_simplex_volume[dim]
        X_k_minus_1_vol = self.nd_simplex_volume[dim-1]
        X_k_minus_1 = self.nd_simplex_collection[dim-1]
        I = []; J = []; V = []; wV = [];
        for i in range(len(X_k)):
            sigma = X_k[i]; wsigma = X_k_vol[i]
            for k in range(len(sigma)):
                face = [sigma[j] for j in range(len(sigma)) if j!=k]
                I.append(self.simplex_dic[tuple(face)])
                J.append(i)
                V.append((-1.0)**k)
                wV.append((-1.0)**k*wsigma/X_k_minus_1_vol[self.simplex_dic[tuple(face)]])
        cobmat = sparse.coo_matrix((V,(J, I)), \
               shape=(len(X_k), len(X_k_minus_1)))
        wcobmat = sparse.coo_matrix((wV,(J, I)), \
               shape=(len(X_k), len(X_k_minus_1))) 
        self.cobmats[dim] = cobmat
        self.wcobmats[dim] = wcobmat

    def integer_lifting(self, c):
        int_c = []
        for i in c:
            if int(i) > ( int(self.prime)-1 )/2:
                ii = i - float(self.prime)
                int_c.append(ii)
            else:
                int_c.append(i)
        return int_c

    def smooth(self, dim, cocycle, filt_value, lap_type):
        """Generate smoothed cocycles

        :param dim: dimension of the cocycle
        :type dim: int
        :param cocycle: the original representative cocycle with coefficients and simplices
        :type cocycle: [list of scalars, list of tuples]
        :param filt_value: the filtration value
        :type filt_value: float
        :param lap_type: the type of Laplacian to measure smoothness. 1. 'combinatorial_full', combinatorial Laplacian, 
            2. 'combinatorial_low', the lower combinatorial Laplacian,
        :type lap_type: str
        """
        cocycle_coef = cocycle[0]
        cocycle_coef = self.integer_lifting(cocycle_coef)
        cocycle_simp = cocycle[1]
        cocycle_dic = {}
        for i in range(len(cocycle_coef)):
            cocycle_dic[tuple(cocycle_simp[i])] = cocycle_coef[i]
        ns_k = np.max(np.where(self.nd_simplex_filt[dim]<=filt_value)[0]) + 1
        ns_k_minus_1 = np.max(np.where(self.nd_simplex_filt[dim-1]<=filt_value)[0]) + 1
        X_k = self.nd_simplex_collection[dim];
        X_k_minus_1 = self.nd_simplex_collection[dim-1]
        cocycle_org = np.zeros([ns_k], float)
        for i in range(ns_k):
            if tuple(X_k[i]) in cocycle_dic.keys():
                cocycle_org[i] = cocycle_dic[tuple(X_k[i])]

        I = np.arange(ns_k); V = np.ones(ns_k);
        left_mat = sparse.coo_matrix((V, (I,I)), shape=(ns_k, len(X_k)))
        I = np.arange(ns_k_minus_1); V = np.ones(ns_k_minus_1);
        right_mat = sparse.coo_matrix((V, (I,I)), shape=(len(X_k_minus_1), ns_k_minus_1))
        cobmat = left_mat*self.cobmats[dim]*right_mat

        if lap_type == "combinatorial_full":
            Lmat = self.construct_weighted_combinatorial_laplacian(dim, filt_value, type="full")
        elif lap_type =="combinatorial_low":
            Lmat = self.construct_weighted_combinatorial_laplacian(dim, filt_value, type="low")
        f = lsqr((Lmat)*cobmat, -(Lmat)*cocycle_org)[0]

        cocycle_smooth = cocycle_org + cobmat*f
        return cocycle_smooth, cocycle_org

    def construct_smoothed_cocycles(self, dim, thrsh, lap_type, nsp=None, lsp=None):
        """Construct smoothed cocycles.

        :param dim: dimension
        :type dim: int
        :param thrsh: only bars longer than threshold is considered and smoothed
        :type thrsh: float
        :param lap_type: the type of Laplacian to measure smoothness. 1. 'combinatorial_full', weighted combinatorial Laplacian; 
            2. 'combinatorial_low', the lower combinatorial laplacian
        :type lap_type: str
        :param nsp: number of points separated equally to take for each bar
        :type nsp: int
        :param lsp: the stepsize for taking points for each bar
        :type lsp: float
        """
        tmp_record = []
        tmp_org_record = []
        if dim > 0:
            for tmp_pt in self.dgms[dim]:
                pt = []
                if self.rule == 'alpha':
                    birth = sqrt(tmp_pt.birth); death = sqrt(tmp_pt.death)
                else:
                    birth = tmp_pt.birth; death = tmp_pt.death
                pt.extend([birth, death])
                # For alpha filtration, sqrt is already taken in pcoh computation
                if pt[1] - pt[0] < thrsh: continue
                tmp_cocycle = str(self.p.cocycle(tmp_pt.data)).split('+')
                cocycle = []
                for tmp in tmp_cocycle:
                    c,i = tmp.split('*')
                    cocycle.append((int(c),int(i)))
                pt.append(cocycle)
                if not self.rips_max == None:
                    pt[1] = min(pt[1], self.rips_max)
                if pt[1] > 10000.0: pt[1] = self.f_values[-1]
                bar_len = pt[1] - pt[0]
                if not nsp==None:
                    bar_step = bar_len/float(nsp)
                elif not lsp==None:
                    nsp = np.ceil(bar_len/lsp)
                    bar_step = bar_len/float(nsp)
                sp_pts = [pt[0] + 0.5*bar_step + float(i)*bar_step for i in range(nsp)]
                instance = [pt[0], pt[1]]
                org_instance = [pt[0], pt[1]]
                for filt_v in sp_pts:
                    rf_len = np.max(np.where(np.asarray(self.f_values) <= filt_v)[0])
                    tmp_cocycle = [(c,i) for (c,i) in pt[2] if i <= rf_len]
                    cocycle_coef = []; cocycle_simp = [];
                    for c in tmp_cocycle:
                        cocycle_coef.append(float(c[0]))
                        cocycle_simp.append(self.f_simplices[c[1]])
                    smoothed_cocycle, org_cocycle = self.smooth(dim, [cocycle_coef, cocycle_simp], filt_v, lap_type)
                    # print(smoothed_cocycle)
                    instance.append([filt_v, smoothed_cocycle])
                    org_instance.append([filt_v, org_cocycle])
                
                tmp_record.append(instance)
                tmp_org_record.append(org_instance)
        elif dim == 0:
            print( "Use UnionFindFor0D.py" )

        self.pcoh_cocycles[dim] = tmp_record
        self.org_cocycles[dim] = tmp_org_record

    def compute_enriched_barcode(self, f, dim):
        """Generate the enriched barcodes

        Returns: 
            1. the original barcode,
            2. the filtration values examined for each bar,
            3. the corresponding function values for each bar.

        :param f: the information on the nodes
        "type f: class:`numpy.ndarray`
        :param dim: dimension of barcode
        :type dim: int
        """
        simplices = self.nd_simplex_collection[dim]
        simplices_f = []
        for s in simplices:
            simplices_f.append(np.mean(f[np.array(s,int)]))
        simplices_f = np.array(simplices_f)
        barcode = []
        filt_values = []
        feature_values = []
        for tmp in self.pcoh_cocycles[dim]:
            b = tmp[0]; d = tmp[1]
            barcode.append([b,d])
            tmp_filt_values = []
            tmp_feature_values = []
            for cocycle in tmp[2:]:
                tmp_filt_values.append(cocycle[0])
                tmp_ns = len(cocycle[1])
                tmp_feature_values.append(np.sum(simplices_f[:tmp_ns] * np.abs(cocycle[1]))/np.sum(np.abs(cocycle[1])))
            filt_values.append(tmp_filt_values)
            feature_values.append(tmp_feature_values)
        barcode = np.array(barcode)
        filt_values = np.array(filt_values)
        feature_values = np.array(feature_values)
        return barcode, filt_values, feature_values
                
    def plot_enriched_barcode(self, barcode, filt_values, feature_values, dims=None, vminmax=None, colorbar_ticks=None):
        """Plot the enriched barcode

        :param barcode: an array of original barcode
        :type barcode: class:`numpy.ndarray`
        :param filt_values: the filtration values examined for each bar
        :type filt_values: class:`numpy.ndarray`
        :param feature_values: the value reflecting the additional information
        :type feature_values: class:`numpy.ndarray`
        """
        if dims is None:
            dims = np.zeros([barcode.shape[0]], int)
        possible_dims = np.sort(list(set(list(dims))))
        left = np.min(barcode[:,0]) - 0.15
        right = np.max(barcode[:,1]) + 0.15
        step = (right-left)/float(barcode.shape[0])
        start = 0.0
        if vminmax is None:
            vmin = np.min(feature_values)
            vmax = np.max(feature_values)
            vminmax = [vmin, vmax]
        cnorm = mpl.colors.Normalize(vmin=vminmax[0], vmax=vminmax[1])
        scalarMap = mpl.cm.ScalarMappable(norm=cnorm, cmap=mpl.cm.jet)
        npt = filt_values.shape[1]
        cnt = 0
        for current_dim in possible_dims:
            if cnt > 0:
                start += 5.0*step
                plt.plot([left, right],[start, start], c='k')
                start += 5.0*step
            cnt += 1
            tmp_ind = np.where(dims==current_dim)[0]
            current_barcode = barcode[tmp_ind,:]
            current_feature_values = feature_values[tmp_ind,:]
            current_filt_values = filt_values[tmp_ind,:]
            Index = np.argsort(current_barcode[:,0])
            for i in Index:
                endpoints = [current_barcode[i,0]]
                for j in range(npt-1):
                    endpoints.append(0.5*(current_filt_values[i,j]+current_filt_values[i,j+1]))
                endpoints.append(current_barcode[i,1])
                for j in range(npt):
                    colorVal = scalarMap.to_rgba(current_feature_values[i,j])
                    plt.plot([endpoints[j], endpoints[j+1]],[start, start], c = colorVal, linewidth=3)
                start += step
        scalarMap.set_array(feature_values.reshape(-1))
        if colorbar_ticks is None:
            plt.colorbar(scalarMap)
        else:
            plt.colorbar(scalarMap, ticks=colorbar_ticks)
        plt.xlim(left, right)
        plt.ylim(-0.1, start+0.1)
        plt.tick_params(
        axis='y',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        left='off',
        labelleft='off') # labels along the bottom edge are off

    def plot_cocycle_1D(self, cocycle, vminmax):
        if vminmax is None:
            vmin, vmax = [np.min(np.abs(cocycle)), np.max(np.abs(cocycle))]
        else:
            vmin, vmax = vminmax
        # scalarMap = mpl.cm.ScalarMappable(norm=cnorm, cmap=mpl.cm.coolwarm)
        edge_weights = np.abs(cocycle)
        pt_weights = np.zeros(self.X.shape[0], float)
        edge_alphas = edge_weights/np.max(edge_weights)
        for i in range(len(cocycle)):
            i_pt, j_pt = self.nd_simplex_collection[1][i]
            pt_weights[i_pt] += edge_weights[i]
            pt_weights[j_pt] += edge_weights[i]
        
        if self.nd == 2:
            for i in range(len(cocycle)):
                i_pt, j_pt = self.nd_simplex_collection[1][i]
                [x1,y1] = self.X[i_pt][:]
                [x2,y2] = self.X[j_pt][:]
                plt.plot([x1,x2], [y1,y2], c='grey', alpha=edge_alphas[i], linewidth=0.5)
            plt.scatter(self.X[:,0], self.X[:,1], c=pt_weights, cmap='coolwarm', zorder=10)
            plt.colorbar()
            plt.xticks([]); plt.yticks([])
            plt.axis('equal'); plt.axis('off')
        elif self.nd == 3:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection = '3d')
            ax.scatter(self.X[:,0], self.X[:,1], self.X[:,2], color=pt_weights, marker = 'o')
            for i in range(len(cocycle)):
                i_pt, j_pt = self.nd_simplex_collection[1][i]
                [x1,y1,z1] = self.X[i_pt,:]
                [x2,y2,z2] = self.X[j_pt,:]
                if edge_weights[i] > 0.1:
                    ax.plot([x1,x2],[y1,y2],[z1,z2], c = 'grey', alpha=edge_alphas[i])
    
    def plot_cocycle_2D(self, cocycle):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection = '3d')
        pt_weights = np.zeros([self.X.shape[0]], float)
        for i in range(len(cocycle)):
            tmp_ind = np.array( self.nd_simplex_collection[2][i], int )
            pt_weights[tmp_ind] += np.abs(cocycle[i])
        sizes = 1.0 + pt_weights * 100.0
        sizes[np.where(sizes>10)] = 10
        ax.scatter(self.X[:,0], self.X[:,1], self.X[:,2], c=self.X[:,0], marker = 'o', s=sizes, cmap=mpl.cm.jet)
        for i in range(len(cocycle)):
            if abs(cocycle[i]) > 0.005:
                # colorVal = scalarMap.to_rgba(abs(cocycle[i]))
                [x1,y1,z1] = self.X[self.nd_simplex_collection[2][i][0]][:3]
                [x2,y2,z2] = self.X[self.nd_simplex_collection[2][i][1]][:3]
                [x3,y3,z3] = self.X[self.nd_simplex_collection[2][i][2]][:3]
                tri = [[np.array([x1,y1,z1]),np.array([x2,y2,z2]),np.array([x3,y3,z3])]]
                ax.add_collection3d(Poly3DCollection(tri, facecolors='grey',alpha=0.7))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.set_xlim(-0.1,4.1)
        ax.set_ylim(-0.1,2.1)
        ax.set_zlim(-0.1,2.1)
        ax.plot([-0.099,-0.099], [-0.099,-0.099], [-0.099,2.099], color='k')
        ax.plot([4.099,4.099], [-0.099,-0.099], [-0.099,2.099], color='k')
        ax.plot([-0.099,-0.099], [2.099,2.099], [-0.099,2.099], color='k')
        ax.plot([4.099,4.099], [2.099,2.099], [-0.099,2.099], color='k')

        ax.plot([-0.099,-0.099], [-0.099,2.099], [-0.099,-0.099], color='k')
        ax.plot([4.099,4.099], [-0.099,2.099], [-0.099,-0.099], color='k')
        ax.plot([4.099,4.099], [-0.099,2.099], [2.099,2.099], color='k')
        ax.plot([-0.099,-0.099], [-0.099,2.099], [2.099,2.099], color='k')

        ax.plot([-0.099,4.099], [-0.099,-0.099], [-0.099,-0.099], color='k')
        ax.plot([-0.099,4.099], [2.099,2.099], [-0.099,-0.099], color='k')
        ax.plot([-0.099,4.099], [-0.099,-0.099], [2.099,2.099], color='k')
        ax.plot([-0.099,4.099], [2.099,2.099], [2.099,2.099], color='k')

        ax.view_init(0., -97.5)
        ax.set_axis_off()
            
def generate_coulomb_potential_enriched_barcode(pcoh_result, atm_pos, charges):
    """This generates the results for protein electrostatic example.
    """
    f = charges
    nd_simplices = pcoh_result[0]; cocycles = pcoh_result[1];
    dgm_f = []
    for dim in [1,2]:
        simplices = nd_simplices[dim]
        simplices_v = []
        simplices_abs_v = []
        # Compute Coulomb interaction and put them in simplices_v
        for s in simplices:
            value = 0.0; abs_value = 0.0;
            for ii in range(len(s)-1):
                for jj in range(ii+1, len(s)):
                    v1 = s[ii]; v2 = s[jj];
                    d = np.sqrt( (atm_pos[v1,0]-atm_pos[v2,0])**2 + (atm_pos[v1,1]-atm_pos[v2,1])**2 + (atm_pos[v1,2]-atm_pos[v2,2])**2 )
                    clb = f[v1]*f[v2]/d
                    value += clb
                    abs_value += abs(clb)
            simplices_v.append(value/float(dim+1))
            simplices_abs_v.append(abs_value/float(dim+1))
        for j in range(len(cocycles[dim])):
            cocycle = cocycles[dim][j]
            b = []
            b.append(dim)
            b.append(cocycle[0]); b.append(cocycle[1]);
            f_sum = 0.0
            f_abs_sum = 0.0
            weight_sum = 0.0
            for k in range(len(cocycle[2][1])):
                w = abs(cocycle[2][1][k])
                f_sum += w*simplices_v[k]
                f_abs_sum += w*simplices_abs_v[k]
                weight_sum += w
            if weight_sum == 0.0:
                print( '0 weight_sum' )
                f_avg = f_sum; f_abs_avg = f_abs_sum;
            else:
                if weight_sum < 1E-4:
                    print( "small weight_sum" )
                f_avg = f_sum/weight_sum
                f_abs_avg = f_abs_sum/weight_sum
            b.extend([f_sum, f_abs_sum, f_avg, f_abs_avg])
            dgm_f.append(b)
    return np.array(dgm_f)

def weighted_wasserstein_distance(barcode1, barcode2, alphas, p):
    """Computes a sequence of the wasserstein distances.

    Returns:
        a collection of the two components (barcode and the function) w.r.t. the given alphas

    :param barcode1: enriched barcode with the columns being birth, death, and the non-geometric information
    :type barcode1: class:`numpy.ndarray`
    :param barcode2: the other enriched barcode to compare with
    :type barcode2: class:`numpy.ndarray`
    :param alphas: the weights to consider in [0,1], when alpha=0, it returns the p-Wasserstein distance for traditional barcodes
    :type alphas: class:`numpy.ndarray`
    :param p: the power for the Wasserstein distance
    :type p: float
    """
    m = barcode1.shape[0]; n = barcode2.shape[0];
    ghost_barcode1 = np.zeros(barcode1.shape)
    for i in range(ghost_barcode1.shape[0]):
        ghost_barcode1[i][0] = 0.5*(barcode1[i][0]+barcode1[i][1])
        ghost_barcode1[i][1] = 0.5*(barcode1[i][0]+barcode1[i][1])
    ghost_barcode2 = np.zeros(barcode2.shape)
    for i in range(ghost_barcode2.shape[0]):
        ghost_barcode2[i][0] = 0.5*(barcode2[i][0]+barcode2[i][1])
        ghost_barcode2[i][1] = 0.5*(barcode2[i][0]+barcode2[i][1])
    N = barcode1.shape[0] + barcode2.shape[0]
    cost_bar = np.zeros([N,N], float)
    cost_fld = np.zeros([N,N], float)
    mbarcode1 = np.concatenate((barcode1, ghost_barcode2), axis=0)
    mbarcode2 = np.concatenate((barcode2, ghost_barcode1), axis=0)


    for i in range(N):
        for j in range(N):
            if i < m or j < n:
                c_bar = pow(max(abs(mbarcode1[i][0]-mbarcode2[j][0]), abs(mbarcode1[i][1]-mbarcode2[j][1])), p)
                c_fld = pow(abs(mbarcode1[i][2]-mbarcode2[j][2]), p)
                cost_bar[i,j] = c_bar; cost_fld[i,j] = c_fld;

    DistanceContour = []
    for alpha in alphas:
        cost = (1.0-alpha)*cost_bar + alpha*cost_fld
        row_ind, col_ind = linear_sum_assignment(cost)
        dis_bar = pow(cost_bar[row_ind, col_ind].sum(), 1.0/float(p))
        dis_fld = pow(cost_fld[row_ind, col_ind].sum(), 1.0/float(p))
        # print( dis_bar, dis_fld )
        DistanceContour.append([dis_bar, dis_fld])
    DistanceContour = np.asarray(DistanceContour)
    return DistanceContour


def main():
    name = sys.argv[1]
    # X = np.loadtxt(name+".pts")[:,:2]
    X = np.array([[0,0],[0,1],[1,1],[1,0]], float)
    # X = np.array([[0,0],[0,1],[1.0,1.0],[1,0],[0.5,-0.5],[1.5,0.5],[0.5,1.5],[-0.5,0.5]])
    # X = np.array([[0,0],[-0.5,np.sqrt(3)/2],[0.5,np.sqrt(3)/2],[0,np.sqrt(3)]])
    # X = np.array([[0,0],[0,1],[1,1],[1,0],[0.5,-0.75],[1.75,0.5],[0.5,1.75],[-0.75,0.5]])
    a = PCF(X)
    a.filtration('rips_euc', rips_max=2)
    # a.filtration('alpha')
    a.compute_volume()
    a.pcoh(prime = 547)
    a.collect_nd_simplices()
    a.construct_weight_matrix(1)
    a.assemble_boundary_matrix(1)
    a.assemble_boundary_matrix(2)
    # a.smooth(1, [[1.0,1.0],[[38,90],[8,9]]], 0.5)
    a.construct_smoothed_cocycles(1, 0.1, "combinatorial_full", nsp=1)
    b,c,d = a.compute_enriched_barcode(np.loadtxt(name+".pts")[:,-1], 1)
    print(b,c,d)
    a.plot_enriched_barcode(b,c,d,(-0.5,0.5))
    # with open(name + '_pcoh.pkl', 'w') as f:
    #     pickle.dump([a.nd_simplex_collection, a.pcoh_cocycles], f)

if __name__ == "__main__":
    main()
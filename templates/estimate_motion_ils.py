import numpy as np
from numpy.linalg import det, inv, svd
from rpy_from_dcm import rpy_from_dcm
from dcm_from_rpy import dcm_from_rpy
from estimate_motion_ls import estimate_motion_ls

def estimate_motion_ils(Pi, Pf, Si, Sf, iters):
    """
    Estimate motion from 3D correspondences.
  
    The function estimates the 6-DOF motion of a boby, given a series 
    of 3D point correspondences. This method relies on NLS.
    
    Arrays Pi and Pf store corresponding landmark points before and after 
    a change in pose.  Covariance matrices for the points are stored in Si 
    and Sf.

    Parameters:
    -----------
    Pi  - 3xn np.array of points (intial - before motion).
    Pf  - 3xn np.array of points (final - after motion).
    Si  - 3x3xn np.array of landmark covariance matrices.
    Sf  - 3x3xn np.array of landmark covariance matrices.

    Outputs:
    --------
    Tfi  - 4x4 np.array, homogeneous transform matrix, frame 'i' to frame 'f'.
    """
    # Initial guess...
    Tfi = estimate_motion_ls(Pi, Pf, Si, Sf)
    C = Tfi[:3, :3]
    I = np.eye(3)
    rpy = rpy_from_dcm(C).reshape(3, 1)

    # Iterate.
    for j in np.arange(iters):
        A = np.zeros((6, 6))
        B = np.zeros((6, 1))
        Wj = np.zeros((6, 6))
        #--- FILL ME IN ---
        # Partial derivatives of the rotation matrix
        dRdr, dRdp, dRdy = dcm_jacob_rpy(C)

        for i in np.arange(Pi.shape[1]):
            # Jacobian Matrix 
            Jj = np.hstack((dRdr @ Pi[:,i].reshape((3, 1)), dRdp @ Pi[:,i].reshape((3,1)), dRdy @ Pi[:,i].reshape((3,1))))
            Jj = Jj.reshape((3,3))
            # Hj = [Jj I]
            Hj = np.hstack((Jj, I))
            # Qj = Qcj - R*Qpj + Jj * theta
            Qj = Pf[:,i].reshape((3, 1)) - C @ Pi[:,i].reshape((3, 1)) +  Jj @ rpy
            # Covariance 
            Wj[:3, :3] = Si[:, :, i]
            Wj[3:, 3:] = Sf[:, :, i]
            W = inv(Hj @ Wj @ Hj.T)

            A += Hj.T @ W @ Hj 
            B += Hj.T @ W @ Qj 

        #------------------

        # Solve system and check stopping criteria if desired...
        theta = inv(A) @ B
        rpy = theta[0:3].reshape(3, 1)
        C = dcm_from_rpy(rpy)
        t = theta[3:6].reshape(3, 1)

    Tfi = np.vstack((np.hstack((C, t)), np.array([[0, 0, 0, 1]])))

    return Tfi

def dcm_jacob_rpy(C):
     # Rotation - convenient!
    cp = np.sqrt(1 - C[2, 0]*C[2, 0])
    cy = C[0, 0]/cp
    sy = C[1, 0]/cp

    dRdr = C@np.array([[ 0,   0,   0],
                       [ 0,   0,  -1],
                       [ 0,   1,   0]])

    dRdp = np.array([[ 0,    0, cy],
                     [ 0,    0, sy],
                     [-cy, -sy,  0]])@C

    dRdy = np.array([[ 0,  -1,  0],
                     [ 1,   0,  0],
                     [ 0,   0,  0]])@C
    
    return dRdr, dRdp, dRdy
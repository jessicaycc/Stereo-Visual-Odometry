import numpy as np
from numpy.linalg import inv, norm

def triangulate(Kl, Kr, Twl, Twr, pl, pr, Sl, Sr):
    """
    Triangulate 3D point position from camera projections.

    The function computes the 3D position of a point landmark from the 
    projection of the point into two camera images separated by a known
    baseline.

    Parameters:
    -----------
    Kl   - 3 x 3 np.array, left camera intrinsic calibration matrix.
    Kr   - 3 x 3 np.array, right camera intrinsic calibration matrix.
    Twl  - 4x4 np.array, homogeneous pose, left camera in world frame.
    Twr  - 4x4 np.array, homogeneous pose, right camera in world frame.
    pl   - 2x1 np.array, point in left camera image.
    pr   - 2x1 np.array, point in right camera image.
    Sl   - 2x2 np.array, left image point covariance matrix.
    Sr   - 2x2 np.array, right image point covariance matrix.

    Returns:
    --------
    Pl  - 3x1 np.array, closest point on ray from left camera  (in world frame).
    Pr  - 3x1 np.array, closest point on ray from right camera (in world frame).
    P   - 3x1 np.array, estimated 3D landmark position in the world frame.
    S   - 3x3 np.array, covariance matrix for estimated 3D point.
    """
    #--- FILL ME IN ---

    # Compute baseline (right camera translation minus left camera translation).
    tl = Twl[:3,3].reshape(3,1)
    tr = Twr[:3,3].reshape(3,1)
    b = tr - tl

    # Unit vectors projecting from optical center to image plane points.
    # Use variables rayl and rayr for the rays.
    cl = Twl[:3, :3]
    cr = Twr[:3, :3]
    rl = cl @ inv(Kl) @ np.vstack((pl, 1))
    rr = cr @ inv(Kr) @ np.vstack((pr, 1))


    rayl = np.divide(rl, norm(rl))
    rayr = np.divide(rr, norm(rr))
    

    # Projected segment lengths.
    # Use variables ml and mr for the segment lengths.
    ml = (np.dot(b.T, rayl) - np.dot(b.T, rayr) * np.dot(rayl.T, rayr)) / (1 - (np.dot(rayl.T, rayr))**2)
    mr = np.dot(rayl.T, rayr) * ml - np.dot(b.T,rayr)
    
    # Segment endpoints.
    # User variables Pl and Pr for the segment endpoints.
    Pl = tl + rayl * ml
    Pr = tr + rayr * mr

    # Now fill in with appropriate ray Jacobians. These are 
    # 3x4 matrices, but two columns are zeros (because the right
    # ray direction is not affected by the left image point and 
    # vice versa).
    drayl = np.zeros((3, 4))  # Jacobian left ray w.r.t. image points.
    drayr = np.zeros((3, 4))  # Jacobian right ray w.r.t. image points.

    # Add code here...
    # Jacobian of the unnormalized rays
    drl = np.hstack(((cl @ inv(Kl))[:, :2], np.zeros((3, 2))))
    drr = np.hstack(( np.zeros((3, 2)),(cr @ inv(Kr))[:, :2]))

    # Jacobian of the norm rays 
    d_norml = np.hstack((np.transpose(rl) @ (cl @ inv(Kl))[:, :2], np.zeros((1, 2))))
    d_normr = np.hstack((np.zeros((1, 2)), np.transpose(rr) @ (cr @ inv(Kr))[:, :2] ))

    d_norml = (1/norm(rl)) * d_norml
    d_normr = (1/norm(rr)) * d_normr
    
    # Combine the two above to get the Jacobian of r_hat
    drayl = (norm(rl) * drl - rl @ d_norml)/(norm(rl))**2
    drayr = (norm(rr) * drr - rr @ d_normr)/(norm(rr))**2
    #------------------

    # Compute dml and dmr (partials wrt segment lengths).
    u = np.dot(b.T, rayl) - np.dot(b.T, rayr)*np.dot(rayl.T, rayr)
    v = 1 - np.dot(rayl.T, rayr)**2

    du = (b.T@drayl).reshape(1, 4) - \
         (b.T@drayr).reshape(1, 4)*np.dot(rayl.T, rayr) - \
         np.dot(b.T, rayr)*((rayr.T@drayl) + (rayl.T@drayr)).reshape(1, 4)
 
    dv = -2*np.dot(rayl.T, rayr)*((rayr.T@drayl).reshape(1, 4) + \
        (rayl.T@drayr).reshape(1, 4))

    m = np.dot(b.T, rayr) - np.dot(b.T, rayl)@np.dot(rayl.T, rayr)
    n = np.dot(rayl.T, rayr)**2 - 1

    dm = (b.T@drayr).reshape(1, 4) - \
         (b.T@drayl).reshape(1, 4)*np.dot(rayl.T, rayr) - \
         np.dot(b.T, rayl)@((rayr.T@drayl) + (rayl.T@drayr)).reshape(1, 4)
    dn = -dv

    dml = (du*v - u*dv)/v**2
    dmr = (dm*n - m*dn)/n**2

    # Finally, compute Jacobian for P w.r.t. image points.
    JP = (ml*drayl + rayl*dml + mr*drayr + rayr*dmr)/2

    #--- FILL ME IN ---

    # 3D point.
    P = (Pl + Pr)/2

    # 3x3 landmark point covariance matrix (need to form
    # the 4x4 image plane covariance matrix first).
    diag = np.zeros((4,4))
    diag[:2, :2] = Sl
    diag[2:, 2:] = Sr
    S = JP @ diag @ np.transpose(JP)
    #------------------

    return Pl, Pr, P, S
import math
import numpy as np 
from numpy.linalg import inv
import os
import argparse

def get_relative_6dof(p1, r1, p2, r2, rotation_mode='axang', correction=None):
    """ 
    Relative pose between two cameras. 
    
    Arguments:
        p1, p2: world coordinate of camera 1/2
        r1, r2: rotation of camera 1/2 relative to world frame
        rotation_mode: rotation convention for r1, r2. Default 'axang'
        correction: rotation matrix correction applied when camera was mounted 
                    on the arm differently from other times
    Returns:
        t: 6 degree of freedom pose from (p1, r1) to (p2, r2) 
    """

    # Convert to rotation matrix
    if rotation_mode == "axang":
        r1 = axang_to_rotm(r1, with_magnitude=True)
        r2 = axang_to_rotm(r2, with_magnitude=True)
    elif rotation_mode == "euler":
        r1 = euler_to_rotm(r1)
        r2 = euler_to_rotm(r2)
    elif rotation_mode == "rotm":
        r1 = r1 
        r2 = r2 

    if correction is not None:
        r1 = r1 @ correction
        r2 = r2 @ correction

    r1 = r1.transpose() 
    r2 = r2.transpose()

    # Ensure translations are column vectors
    p1 = np.float32(p1).reshape(3,1)
    p2 = np.float32(p2).reshape(3,1)

    # Concatenate to transformation matrices
    T1 = np.vstack([np.hstack([r1, p1]), [0,0,0,1]])
    T2 = np.vstack([np.hstack([r2, p2]), [0,0,0,1]])
    
    relative_pose = inv(T1) @ T2    # [4,4] transform matrix
    rotation = relative_pose[0:3, 0:3]
    rotation = rotm_to_euler(rotation)
    translation = relative_pose[0:3, 3]

    return np.hstack((translation, rotation))


def rotm_to_euler(R):
    """
    Rotation matrix to euler angles.
    DCM angles are decomposed into Z-Y-X euler rotations.
    """ 

    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
    singular = sy < 1e-6
 
    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0

    return np.array([x, y, z])


def axang_to_rotm(r, with_magnitude=False):
    """ 
    Axis angle representation to rotation matrix. 
        Expect 3-vector for with_magnitude=True.
        Expect 4-vector for with_magnitude=False.
    """

    if with_magnitude:
        theta = np.linalg.norm(r) + 1e-15
        r = r / theta 
        r = np.append(r, theta)

    kx, ky, kz, theta = r

    ctheta = math.cos(theta)
    stheta = math.sin(theta)
    vtheta = 1 - math.cos(theta)

    R = np.float32([
        [kx*kx*vtheta + ctheta,     kx*ky*vtheta - kz*stheta,   kx*kz*vtheta + ky*stheta],
        [kx*ky*vtheta + kz*stheta,  ky*ky*vtheta + ctheta,      ky*kz*vtheta - kx*stheta],
        [kx*kz*vtheta - ky*stheta,  ky*kz*vtheta + kx*stheta,   kz*kz*vtheta + ctheta   ]
    ])

    return R


def euler_to_rotm(alpha, beta, gamma):
    """
    Euler angle representation to rotation matrix. Rotation is composed in Z-Y-X order.
    
    Arguments:
        Gamma: rotation about z
        Alpha: rotation about x
        Beta: rotation about y
    """

    Rx = np.float32([
        [1,  0,                0              ],
        [0,  math.cos(alpha), -math.sin(alpha)],
        [0,  math.sin(alpha),  math.cos(alpha)]
    ])

    Ry = np.float32([
        [math.cos(beta),  0,    math.sin(beta)],
        [0,               1,    0             ],
        [-math.sin(beta), 0,    math.cos(beta)]
    ])

    Rz = np.float32([
        [math.cos(gamma), -math.sin(gamma), 0],
        [math.sin(gamma), math.cos(gamma),  0],
        [0,               0,                1]
    ])

    R = Rz @ Ry @ Rx
    return R


def process_grountruth_relative_poses(sequence_dir, force_recalculate=False):
    """
    Process a directory of '000000x.txt' absolute poses to a numpy array of relative poses.
    """
    relative_pose_file = os.path.join(sequence_dir, "poses_gt_relative.npy")
    absolute_pose_file = os.path.join(sequence_dir, "poses_gt_absolute.npy")
    
    corr = np.array([
        [-1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])

    if os.path.exists(relative_pose_file) and not force_recalculate:
        print("Using existing pose array saved at {}".format(relative_pose_file))
        return (np.load(relative_pose_file), np.load(absolute_pose_file))

    
    relative_poses = []
    absolute_poses = []
    
    pose_files = [f for f in os.listdir(sequence_dir) if f.endswith(".txt")]
    pose_files = [os.path.join(sequence_dir, f) for f in sorted(pose_files)]
    previous_pose = None
    first_pose = None
    
    for p in pose_files:
        # print(p)
        f = open(p, 'r')
        pose = f.readlines()[0]
        pose = pose.replace('(', '').replace(')', '').replace(']', '').replace('[', '')
        pose = np.fromstring(pose, sep=',')
        
        if first_pose is None:
            first_pose = pose
                
        pose_absolute = get_relative_6dof(first_pose[0:3], first_pose[3:], pose[0:3], pose[3:], correction=corr)
        absolute_poses.append(pose_absolute)
        
        if previous_pose is None:
            previous_pose = pose
            continue
            
        pose_relative = get_relative_6dof(previous_pose[0:3], previous_pose[3:], pose[0:3], pose[3:], correction=corr)
        relative_poses.append(pose_relative)
        previous_pose = pose

    relative_poses = np.array(relative_poses)
    absolute_poses = np.array(absolute_poses)
    np.save(relative_pose_file, relative_poses)
    np.save(absolute_pose_file, absolute_poses)
    
    return relative_poses, absolute_poses


def visualise_ground_truth_trajectories(rel, savename=None):
    import matplotlib.pyplot as plt
    xs = rel[:, 0]
    ys = rel[:, 1]
    zs = rel[:, 2]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot3D(xs, ys, zs, '.')
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    if savename:
        fig.savefig(savename)
        


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="process pose files")

    parser.add_argument("input_folder", type=str, default=None,
                        help="Folder where the pose and epirect files are present.")
    args = parser.parse_args()

    input_folder = args.input_folder
    print("input folder: {}".format(input_folder))
    absolute, relative = process_grountruth_relative_poses(input_folder, force_recalculate=True)
    visualise_ground_truth_trajectories(relative, savename=os.path.join(input_folder, "trajectory.png"))

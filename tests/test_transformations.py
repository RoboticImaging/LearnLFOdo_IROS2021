import numpy as np
import utils
import matplotlib.pyplot as plt

if __name__ == "__main__":
    pose_w_1 = np.array([[1., 0., 0., 1],
                         [0., 1., 0., 2],
                         [0., 0., 1., 3],
                         [0., 0., 0., 1]], dtype=np.float)

    pose_1_2 = np.array([[1.0000000, 0.0000000, 0.0000000, 3],
                         [0.0000000, 1.0000000, 0.0000000, 1],
                         [0.0000000, 0.5000000, 1.0000000, 2],
                         [0., 0., 0., 1.]], dtype=np.float)

    rot_x = np.array([[1.0000000, 0.0000000, 0.0000000, 0],
                         [0.0000000, 0.8660254, -0.5000000, 0],
                         [0.0000000, 0.5000000, 0.8660254, 0],
                         [0., 0., 0., 1]], dtype=np.float)

    rot_y = np.array([[0.7071068,  0.0000000,  0.7071068, 0],
                  [0.0000000,  1.0000000,  0.0000000, 0],
                  [-0.7071068,  0.0000000,  0.7071068, 0],
                  [0.,  0.,  0., 1]], dtype=np.float)

    rot_z = np.array([[0.5000000, -0.8660254, 0.0000000, 0],
                  [0.8660254, 0.5000000, 0.0000000, 0],
                  [0.0000000, 0.0000000, 1.0000000, 0],
                  [0., 0., 0., 1]], dtype=np.float)

    pose_1_5 = pose_1_2.copy()
    pose_1_2 = pose_1_2 @ rot_x @ rot_y @ rot_z

    ## CW 90
    transform = np.array([[0., -1.,  0, 0],
                          [1., 0., 0., 0],
                          [0., 0., 1., 0],
                          [0., 0., 0., 1]], dtype=np.float)
    ## CCW 90
    # transform = np.array([[0., 1., 0, 0],
    #                       [-1., 0., 0., 0],
    #                       [0., 0., 1., 0],
    #                       [0., 0., 0., 1]], dtype=np.float)

    ## 180
    # transform = np.array([[-1., 0., 0, 0],
    #                       [0., -1., 0., 0],
    #                       [0., 0., 1., 0],
    #                       [0., 0., 0., 1]], dtype=np.float)

    ## hflip
    transform = np.array([[-1., 0., 0, 0],
                          [0., 1., 0., 0],
                          [0., 0., 1., 0],
                          [0., 0., 0., 1]], dtype=np.float)

    pose_1_3_ = transform @ pose_1_2
    pose_1_3 = pose_1_3_.copy()
    pose_1_3[:, 0] = -1 * pose_1_3_[:, 1]
    pose_1_3[:, 1] = pose_1_3_[:, 0]

    pose_1_4 = transform @ pose_1_2 @ transform.transpose()         # this is how rotations will be applied

    pose_w_2 = pose_w_1 @ pose_1_2
    pose_w_3 = pose_w_1 @ pose_1_3
    pose_w_4 = pose_w_1 @ pose_1_4

    pose_1_5 = pose_1_5 @ rot_x @ rot_y.transpose() @ rot_z.transpose()
    pose_1_5[0, 3] *= -1

    pose_w_5 = pose_w_1 @ pose_1_5

    # These two should be the same
    rel_t1_t3 = utils.get_relative_6dof(pose_w_1[0:3, 3], pose_w_1[0:3, 0:3], pose_w_3[0:3, 3], pose_w_3[0:3, 0:3],
                                 rotation_mode='rotm', return_as_mat=True)
    rel_t1_t4 = utils.get_relative_6dof(pose_w_1[0:3, 3], pose_w_1[0:3, 0:3], pose_w_4[0:3, 3], pose_w_4[0:3, 0:3],
                                 rotation_mode='rotm', return_as_mat=True)
    rel_t1_t5 = utils.get_relative_6dof(pose_w_1[0:3, 3], pose_w_1[0:3, 0:3], pose_w_5[0:3, 3], pose_w_5[0:3, 0:3],
                                        rotation_mode='rotm', return_as_mat=True)

    print("rel_t1_t2")
    print(pose_1_2)
    # print("rel_t1_t3")
    # print(rel_t1_t3)
    # print("rel_t1_t4")
    # print(rel_t1_t4)
    print("rel_t1_t5")
    print(rel_t1_t5)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=-90, azim=-90)

    utils.draw_axes_at(pose_w_1, ax, scale=1, frame_label=1, arrowstyle='->')
    utils.draw_axes_at(pose_w_2, ax, scale=1, frame_label=2, arrowstyle='->')
    # utils.draw_axes_at(pose_w_3, ax, scale=1, frame_label=3, arrowstyle='->')
    utils.draw_axes_at(pose_w_4, ax, scale=1.5, frame_label=4, arrowstyle='-')   # this should be the same as before
    utils.draw_axes_at(pose_w_5, ax, scale=1.5, frame_label=5, arrowstyle='-')  # this should be the same as before

    ax.plot3D(pose_w_1[0, 3], pose_w_1[1, 3], pose_w_1[2, 3], "r.")
    ax.plot3D(pose_w_2[0, 3], pose_w_2[1, 3], pose_w_2[2, 3], "b.")
    ax.plot3D(pose_w_3[0, 3], pose_w_3[1, 3], pose_w_3[2, 3], "k.")
    ax.plot3D(pose_w_4[0, 3], pose_w_4[1, 3], pose_w_4[2, 3], "g.")     # This should be the same as before
    ax.plot3D(pose_w_5[0, 3], pose_w_5[1, 3], pose_w_5[2, 3], "y.")


    ax.set_xlim([-6, 6])
    ax.set_ylim([-5, 5])
    ax.set_zlim([-5, 5])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    plt.show()
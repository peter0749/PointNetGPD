#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import sys
import re
import pickle  # todo: current pickle file are using format 3 witch is not compatible with python2
from meshpy.obj_file import ObjFile
from meshpy.sdf_file import SdfFile
from dexnet.grasping import GraspableObject3D
import numpy as np
from dexnet.visualization.visualizer3d import DexNetVisualizer3D as Vis
from dexnet.grasping import RobotGripper
from autolab_core import YamlConfig
try:
    from mayavi import mlab
    mlab.init_notebook('x3d', 800, 800)
except ImportError:
    print("can not import mayavi")
    mlab = None
from dexnet.grasping import GpgGraspSampler  # temporary way for show 3D gripper using mayavi
import pcl
import glob
from IPython.display import display


# In[2]:


# global configurations:
home_dir = os.environ['HOME']
yaml_config = YamlConfig(home_dir + "/code/grasp-pointnet/dex-net/test/config.yaml")
gripper_name = 'robotiq_85'
gripper = RobotGripper.load(gripper_name, home_dir + "/code/grasp-pointnet/dex-net/data/grippers")
ags = GpgGraspSampler(gripper, yaml_config)
save_fig = False  # save fig as png file
show_fig = True  # show the mayavi figure
generate_new_file = False  # whether generate new file for collision free grasps
check_pcd_grasp_points = False

if not show_fig:
    mlab.options.offscreen = True

def get_file_name(file_dir_):
    file_list = []
    for root, dirs, files in os.walk(file_dir_):
        # print(root)  # current path
        if root.count('/') == file_dir_.count('/')+1:
            file_list.append(root)
        # print(dirs)  # all the directories in current path
        # print(files)  # all the files in current path
    file_list.sort()
    return file_list


def get_pickle_file_name(file_dir):
    pickle_list_ = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if os.path.splitext(file)[1] == '.pickle':
                pickle_list_.append(os.path.join(root, file))
    return pickle_list_


def fuzzy_finder(user_input, collection):
    suggestions = []
    # pattern = '.*'.join(user_input)  # Converts 'djm' to 'd.*j.*m'
    pattern = user_input
    regex = re.compile(pattern)  # Compiles a regex.
    for item in collection:
        match = regex.search(item)  # Checks if the current item matches the regex.
        if match:
            suggestions.append(item)
    return suggestions


def open_pickle_and_obj(name_to_open_):
    pickle_names_ = get_pickle_file_name(home_dir + "/code/grasp-pointnet/dex-net/apps/generated_grasps")
    suggestion_pickle = fuzzy_finder(name_to_open_, pickle_names_)
    if len(suggestion_pickle) != 1:
        print("Pickle file suggestions:", suggestion_pickle)
        exit("Name error for pickle file!")
    pickle_m_ = pickle.load(open(suggestion_pickle[0], 'rb'))

    file_dir = home_dir + "/dataset/ycb_meshes_google/objects"
    file_list_all = get_file_name(file_dir)
    new_sug = re.findall(r'_\d+', suggestion_pickle[0], flags=0)
    new_sug = new_sug[0].split('_')
    new_sug = new_sug[1]
    suggestion = fuzzy_finder(new_sug, file_list_all)

    # very dirty way to support name with "-a, -b etc."
    if len(suggestion) != 1:
        new_sug = re.findall(r'_\d+\W\w', suggestion_pickle[0], flags=0)
        new_sug = new_sug[0].split('_')
        new_sug = new_sug[1]
        suggestion = fuzzy_finder(new_sug, file_list_all)
        if len(suggestion) != 1:
            exit("Name error for obj file!")
    object_name_ = suggestion[0][len(file_dir) + 1:]
    ply_name_ = suggestion[0] + "/google_512k/nontextured.ply"
    if not check_pcd_grasp_points:
        of = ObjFile(suggestion[0] + "/google_512k/nontextured.obj")
        sf = SdfFile(suggestion[0] + "/google_512k/nontextured.sdf")
        mesh = of.read()
        sdf = sf.read()
        obj_ = GraspableObject3D(sdf, mesh)
    else:
        cloud_path = home_dir + "/code/grasp-pointnet/pointGPD/data/ycb_rgbd/" + object_name_ + "/clouds/"
        pcd_files = glob.glob(cloud_path + "*.pcd")
        obj_ = pcd_files
        obj_.sort()
    return pickle_m_, obj_, ply_name_, object_name_


def display_object(obj_):
    """display object only using mayavi"""
    Vis.figure(bgcolor=(1, 1, 1), size=(1000, 1000))
    Vis.mesh(obj_.mesh.trimesh, color=(0.5, 0.5, 0.5), style='surface')
    Vis.show()


def display_gripper_on_object(obj_, grasp_):
    """display both object and gripper using mayavi"""
    # transfer wrong was fixed by the previews comment of meshpy modification.
    # gripper_name = 'robotiq_85'
    # home_dir = os.environ['HOME']
    # gripper = RobotGripper.load(gripper_name, home_dir + "/code/grasp-pointnet/dex-net/data/grippers")
    # stable_pose = self.dataset.stable_pose(object.key, 'pose_1')
    # T_obj_world = RigidTransform(from_frame='obj', to_frame='world')
    t_obj_gripper = grasp_.gripper_pose(gripper)

    stable_pose = t_obj_gripper
    grasp_ = grasp_.perpendicular_table(stable_pose)

    Vis.figure(bgcolor=(1, 1, 1), size=(1000, 1000))
    Vis.gripper_on_object(gripper, grasp_, obj_,
                          gripper_color=(0.25, 0.25, 0.25),
                          # stable_pose=stable_pose,  # .T_obj_world,
                          plot_table=False)
    Vis.show()


def display_grasps(grasps, graspable, color):
    approach_normal = grasps.rotated_full_axis[:, 0]
    approach_normal = approach_normal/np.linalg.norm(approach_normal)
    major_pc = grasps.configuration[3:6]
    major_pc = major_pc/np.linalg.norm(major_pc)
    minor_pc = np.cross(approach_normal, major_pc)
    center_point = grasps.center
    grasp_bottom_center = -ags.gripper.hand_depth * approach_normal + center_point
    hand_points = ags.get_hand_points(grasp_bottom_center, approach_normal, major_pc)
    local_hand_points = ags.get_hand_points(np.array([0, 0, 0]), np.array([1, 0, 0]), np.array([0, 1, 0]))
    if_collide = ags.check_collide(grasp_bottom_center, approach_normal,
                                   major_pc, minor_pc, graspable, local_hand_points)
    if not if_collide and (show_fig or save_fig):
        ags.show_grasp_3d(hand_points, color=color)
        return True
    elif not if_collide:
        return True
    else:
        return False


def show_selected_grasps_with_color(m, ply_name_, title, obj_):
    m_good = m[m[:, 1] <= 0.4]
    m_good = m_good[np.random.choice(len(m_good), size=25, replace=True)]
    m_bad = m[m[:, 1] >= 1.8]
    m_bad = m_bad[np.random.choice(len(m_bad), size=25, replace=True)]
    collision_grasp_num = 0
    bg_obj = None
    

    if save_fig or show_fig:
        # fig 1: good grasps
        mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0.7, 0.7, 0.7), size=(1000, 1000))
        gg = mlab.pipeline.surface(mlab.pipeline.open(ply_name_))
        for a in m_good:
            # display_gripper_on_object(obj, a[0])  # real gripper
            collision_free = display_grasps(a[0], obj_, color='d')  # simulated gripper
            if not collision_free:
                collision_grasp_num += 1

        if save_fig:
            mlab.savefig("good_"+title+".png")
            mlab.close()
        elif show_fig:
            mlab.title(title, size=0.5)

        # fig 2: bad grasps
        mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0.7, 0.7, 0.7), size=(1000, 1000))
        bg = mlab.pipeline.surface(mlab.pipeline.open(ply_name_))

        for a in m_bad:
            # display_gripper_on_object(obj, a[0])  # real gripper
            collision_free = display_grasps(a[0], obj_, color=(1, 0, 0))
            if not collision_free:
                collision_grasp_num += 1

        if save_fig:
            mlab.savefig("bad_"+title+".png")
            mlab.close()
        elif show_fig:
            mlab.title(title, size=0.5)
            mlab.show()
    elif generate_new_file:
        # only to calculate collision:
        collision_grasp_num = 0
        ind_good_grasp_ = []
        for i_ in range(len(m)):
            collision_free = display_grasps(m[i_][0], obj_, color=(1, 0, 0))
            if not collision_free:
                collision_grasp_num += 1
            else:
                ind_good_grasp_.append(i_)
        collision_grasp_num = str(collision_grasp_num)
        collision_grasp_num = (4-len(collision_grasp_num))*" " + collision_grasp_num
        print("collision_grasp_num =", collision_grasp_num, "| object name:", title)
        return ind_good_grasp_, gg, bg
    return gg, bg


def get_grasp_points_num(m, obj_):
    has_points_ = []
    ind_points_ = []
    for i_ in range(len(m)):
        grasps = m[i_][0]
        # from IPython import embed;embed()
        approach_normal = grasps.rotated_full_axis[:, 0]
        approach_normal = approach_normal / np.linalg.norm(approach_normal)
        major_pc = grasps.configuration[3:6]
        major_pc = major_pc / np.linalg.norm(major_pc)
        minor_pc = np.cross(approach_normal, major_pc)
        center_point = grasps.center
        grasp_bottom_center = -ags.gripper.hand_depth * approach_normal + center_point
        # hand_points = ags.get_hand_points(grasp_bottom_center, approach_normal, major_pc)
        local_hand_points = ags.get_hand_points(np.array([0, 0, 0]), np.array([1, 0, 0]), np.array([0, 1, 0]))

        has_points_tmp, ind_points_tmp = ags.check_collision_square(grasp_bottom_center, approach_normal,
                                                                    major_pc, minor_pc, obj_, local_hand_points,
                                                                    "p_open")
        ind_points_tmp = len(ind_points_tmp)  # here we only want to know the number of in grasp points.
        has_points_.append(has_points_tmp)
        ind_points_.append(ind_points_tmp)
    return has_points_, ind_points_


if __name__ == '__main__':
    name_to_open = '006_mustard_bottle'
    grasps_with_score, obj, ply_name, obj_name = open_pickle_and_obj(name_to_open)
    assert(len(grasps_with_score) > 0)
    with_score = isinstance(grasps_with_score[0], tuple) or isinstance(grasps_with_score[0], list)
    if with_score:
        grasps_with_score = np.array(grasps_with_score)
        gg, bg = show_selected_grasps_with_color(grasps_with_score, ply_name, obj_name, obj)
        display(gg)
        display(bg)


# In[ ]:





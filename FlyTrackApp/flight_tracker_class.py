import os
import numpy as np
import math
import time
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sys
from time import sleep
sys.path.append(os.getcwd()+'/build')
import FlightTracker_lib

class Flight_Tracker_Class():

    # initialization

    def __init__(self):

        self.N_cam = None
        self.mov_nr = None

        self.start_point = None
        self.mid_point = None
        self.end_point = None
        self.trigger_mode = ""

        self.session_loc = ""
        self.mov_name = ""
        self.cam_name = ""
        self.cal_loc = ""
        self.cal_name = ""
        self.bckg_loc = ""
        self.bckg_name = ""
        self.bckg_img_format = ""
        self.frame_name = ""
        self.frame_img_format = ""
        self.sol_loc = ""
        self.sol_name = ""
        self.model_loc = ""
        self.model_name = ""

        self.nx = None
        self.ny = None
        self.nz = None
        self.ds = None
        self.x0 = None
        self.y0 = None
        self.z0 = None

        self.N_threads = None

        # C++ class
        self.flt = FlightTracker_lib.FLT()

    def set_parameters(self):

        # Set class parameters
        self.flt.set_N_cam(self.N_cam)
        self.flt.set_start_point(self.start_point)
        self.flt.set_mid_point(self.mid_point)
        self.flt.set_end_point(self.end_point)
        self.flt.set_trigger_mode(self.trigger_mode)
        self.flt.set_session_loc(self.session_loc)
        self.flt.set_mov_name(self.mov_name)
        self.flt.set_mov_nr(self.mov_nr)
        self.flt.set_cam_name(self.cam_name)
        self.flt.set_cal_loc(self.cal_loc)
        self.flt.set_cal_name(self.cal_name)
        self.flt.set_bckg_loc(self.bckg_loc)
        self.flt.set_bckg_name(self.bckg_name)
        self.flt.set_bckg_img_format(self.bckg_img_format)
        self.flt.set_frame_name(self.frame_name)
        self.flt.set_frame_img_format(self.frame_img_format)
        self.flt.set_sol_loc(self.sol_loc)
        self.flt.set_sol_name(self.sol_name)
        self.flt.set_model_loc(self.model_loc)

    def get_parameters(self):

        # Return class parameters
        print "------------------------------------"
        print "C++ Flight Tracker Class parameters:"
        print "------------------------------------"
        print ""
        print "Number of cameras: " + str(self.flt.get_N_cam())
        print "Start frame: " + str(self.flt.get_start_point())
        print "Mid frame: " + str(self.flt.get_mid_point())
        print "End frame: " + str(self.flt.get_end_point())
        print "Trigger mode: " + str(self.flt.get_trigger_mode())
        print "Session folder: " + str(self.flt.get_session_loc())
        print "Movie name: " + str(self.flt.get_mov_name())
        print "Camera name: " + str(self.flt.get_cam_name())
        print "Calibration folder: " + str(self.flt.get_cal_loc())
        print "Calibration name: " + str(self.flt.get_cal_name())
        print "Background folder: " + str(self.flt.get_bckg_loc())
        print "Background name: " + str(self.flt.get_bckg_name())
        print "Background image format: " + str(self.flt.get_bckg_img_format())
        print "Frame name: " + str(self.flt.get_frame_name())
        print "Frame image format: " + str(self.flt.get_frame_img_format())
        print "Solution location: " + str(self.flt.get_sol_loc())
        print "Solution file name: " + str(self.flt.get_sol_name())
        print "Model location: " + str(self.flt.get_model_loc())
        print "Model name: " + str(self.flt.get_model_name())
        print ""
        print "------------------------------------"

    def set_session_parameters(self):
        self.flt.set_session_par()

    # FrameLoader

    def init_frame_loader(self):
        self.flt.init_frame_loader()

    def load_frame(self,frame_nr):
        self.flt.load_single_frame(frame_nr)

    def load_frame_batch(self,batch_nr,start_frame,end_frame):
        self.flt.load_frame_batch(batch_nr,start_frame,end_frame)

    def get_image_size(self):
        return self.flt.return_image_size()

    def get_frame(self):
        frame = []

        for i in range(0,self.N_cam):
            frame.append(self.flt.return_single_frame(i))

        return frame

    def show_frame(self):
        imgs = []

        for i in range(0,self.N_cam):
            imgs.append(self.flt.return_single_frame(i))

        f = plt.figure()

        plt.subplot(131)
        plt.imshow(imgs[0], cmap='gray')
        plt.title('camera 1')

        plt.subplot(132)
        plt.imshow(imgs[1], cmap='gray')
        plt.title('camera 2')

        plt.subplot(133)
        plt.imshow(imgs[2], cmap='gray')
        plt.title('camera 3')

        f.set_figheight(12)
        f.set_figwidth(30)

        plt.show()

    def raw_frame_interactor(self,frame_nr):

        self.load_frame(frame_nr)

        frame = self.get_frame()

        f = plt.figure()

        plt.subplot(131)
        plt.imshow(frame[0], cmap='gray')
        plt.title('camera 1')

        plt.subplot(132)
        plt.imshow(frame[1], cmap='gray')
        plt.title('camera 2')

        plt.subplot(133)
        plt.imshow(frame[2], cmap='gray')
        plt.title('camera 3')

        f.set_figheight(12)
        f.set_figwidth(30)

        plt.show()

    # FocalGrid

    def set_grid_param(self):
        self.flt.set_N_threads(self.N_threads)
        self.flt.set_nx(self.nx)
        self.flt.set_ny(self.ny)
        self.flt.set_nz(self.nz)
        self.flt.set_ds(self.ds)
        self.flt.set_x0(self.x0)
        self.flt.set_y0(self.y0)
        self.flt.set_z0(self.z0)
        self.flt.set_grid_param()

    def get_grid_param(self):
        print ""
        print "Focal Grid parameters:"
        print "N threads: " + str(self.flt.get_N_threads())
        print "Nx: " + str(self.flt.get_nx())
        print "Ny: " + str(self.flt.get_ny())
        print "Nz: " + str(self.flt.get_nz())
        print "ds: " + str(self.flt.get_ds())
        print "x0: " + str(self.flt.get_x0())
        print "y0: " + str(self.flt.get_y0())
        print "z0: " + str(self.flt.get_z0())
        print ""

    def init_focal_grid(self):
        self.flt.init_focal_grid()

    def construct_focal_grid(self):
        print 'calculating focal grid'
        start = time.time()
        print self.flt.construct_focal_grid()
        print time.time()-start

    def convert_3D_point_2_uv(self,x_in,y_in,z_in):
        return self.flt.xyz_2_uv(x_in,y_in,z_in)

    def drag_point_3D(self,cam_nr, u_now, v_now, u_prev, v_prev, x_prev, y_prev, z_prev):
        return self.flt.drag_point_3d(cam_nr, u_now, v_now, u_prev, v_prev, x_prev, y_prev, z_prev)

    # ModelClass

    def add_model(self,folder_name,file_name):
        self.flt.add_model(folder_name,file_name)

    def clear_model_list(self):
        self.flt.clear_model_list()

    def set_model_loc(self,model_loc):
        self.model_loc = model_loc
        self.flt.set_model_loc(model_loc)

    def get_model_loc(self):
        return self.flt.get_model_loc()

    def load_model(self,model_ind):
        print "load model"
        print model_ind
        self.flt.load_model(model_ind)

    def set_model_scale(self,scale_list):
        self.flt.set_model_scale(scale_list)

    def set_model_origin(self,x0,y0,z0):
        self.flt.set_model_origin(x0,y0,z0)

    def set_body_length(self,body_length):
        self.flt.set_body_length(body_length)

    def set_wing_length(self,wing_length):
        self.flt.set_wing_length(wing_length)

    def set_model_init_state(self):
        self.flt.set_model_init_state()

    #def return_model_pcl(self):
    #    return self.flt.return_model_pointcloud()

    def return_parents(self):
        return self.flt.return_parent_structure()

    def return_model_state(self):
        return self.flt.return_model_state()

    def return_stl_list(self):
        return self.flt.return_stl_list()

    def return_model_scale(self):
        return self.flt.return_model_scale()

    def return_transformation_matrices(self):
        return self.flt.return_transform_mats()

    # ImagSegm

    def set_segmentation_param(self,body_thresh,wing_thresh,Sigma,K,min_body_size,min_wing_size,tethered):
        self.flt.set_seg_param(body_thresh,wing_thresh,Sigma,K,min_body_size,min_wing_size,tethered)

    def segment_single_frame(self):
        self.flt.seg_single_frame()

    def segment_frame_batch(self):
        self.flt.seg_frame_batch()

    def return_segmented_frame(self):
        frame = []

        for i in range(0,self.N_cam):
            frame.append(self.flt.return_seg_frame(i))

        return frame

    # InitState

    def set_weight_xsi(self,weight_xsi):
        self.flt.set_weight_xsi(weight_xsi)

    def set_weight_theta(self,weight_theta):
        self.flt.set_weight_theta(weight_theta)

    def set_weight_length(self,weight_length):
        self.flt.set_weight_length(weight_length)

    def set_weight_volume(self,weight_volume):
        self.flt.set_weight_volume(weight_volume)

    def set_cone_angle(self,cone_angle):
        self.flt.set_cone_angle(cone_angle)

    def set_cone_height(self,cone_height):
        self.flt.set_cone_height(cone_height)

    def project_single_frame_2_pcl(self):
        self.flt.single_frame_2_pcl()

    def find_initial_state(self):
        self.flt.find_initial_state()

    def return_segment_pointcloud(self):
        return self.flt.return_segment_pcl()

    def return_bounding_boxes(self):
        return self.flt.return_bboxes()

    def return_blr_pointclouds(self):
        return self.flt.return_blr_pcl()

    def return_blr_bounding_boxes(self):
        return self.flt.return_blr_bboxes()

    def return_m_body(self):
        return self.flt.return_m_body()

    def return_m_wing_L(self):
        return self.flt.return_m_wing_L()

    def return_m_wing_R(self):
        return self.flt.return_m_wing_R()

    # MultiBodyICP

    def perform_single_frame_icp(self):
        return self.flt.icp_on_single_frame()

    def return_projected_model_pcl(self):
        return self.flt.return_projected_model_pcl()

    def return_projected_model_frames(self):
        frame = []

        for i in range(0,self.N_cam):
            frame.append(self.flt.return_projected_frames(i))

        return frame

    #def return_projected_model_pcl(self):
    #    return self.flt.return_model_pcl()

    #def return_model_imgs(self):
    #    img_list = []
    #    for i in range(self.N_cam):
    #        img_list.append(self.flt.return_model_images(i))
    #    return img_list

    def show_model_imgs(self,img_list):

        f = plt.figure()

        plt.subplot(131)
        plt.imshow(img_list[0], cmap='gray')
        plt.title('camera 1')

        plt.subplot(132)
        plt.imshow(img_list[1], cmap='gray')
        plt.title('camera 2')

        plt.subplot(133)
        plt.imshow(img_list[2], cmap='gray')
        plt.title('camera 3')

        f.set_figheight(12)
        f.set_figwidth(30)

        plt.show()

    def show_seg_imgs(self,img_list):

        f = plt.figure()

        plt.subplot(131)
        im1 = plt.imshow(img_list[0])
        im1.set_cmap('jet')
        plt.title('camera 1')

        plt.subplot(132)
        im2 = plt.imshow(img_list[1])
        im2.set_cmap('jet')
        plt.title('camera 2')

        plt.subplot(133)
        im3 = plt.imshow(img_list[2])
        im3.set_cmap('jet')
        plt.title('camera 3')

        f.set_figheight(12)
        f.set_figwidth(30)

        plt.show()
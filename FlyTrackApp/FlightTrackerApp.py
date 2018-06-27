from __future__ import print_function
import sys
import vtk
#from PyQt5 import QtCore
#from PyQt5 import QtGui 
#from PyQt5 import QtWidgets
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
from PyQt5 import Qt
from PyQt5.QtWidgets import QMainWindow, QApplication, QWidget, QAction, QTableWidget,QTableWidgetItem,QVBoxLayout
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import copy
import time

from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

from flight_tracker_ui import Ui_mainWindow
from flight_tracker_class import Flight_Tracker_Class
from ModelSelectWidget import ModelSelectWidget
from ScaleModelWidget import ScaleModelWidget
from ImageSegmentWidget import ImageSegmentWidget
from BoundingBoxWidget import BBoxWidget
#from flight_tracker_vis_class import FlightTrackerVisualization

#--------------------------------------------------------------------------------------------------
#
# Flight Tracker App
#
#--------------------------------------------------------------------------------------------------

class FlightTrackerApp(QtWidgets.QMainWindow, Ui_mainWindow):

	def __init__(self, parent=None):
		super(FlightTrackerApp,self).__init__(parent)
		self.setupUi(self)
		self.initialize()
		self.connectActions()

	def initialize(self):
		self.init_flight_tracker()
		self.add_model_select_gui()
		self.add_focal_grid_gui()
		self.add_scale_gui()
		self.add_image_segment_gui()
		self.add_bbox_gui()
		self.add_mbicp_gui()
		self.add_frame_proj_gui()

	def connectActions(self):
		self.connect_model_select_gui()
		self.connect_focal_grid_gui()
		self.connect_scale_gui()
		self.connect_image_segment_gui()
		self.connect_bbox_gui()
		self.connect_mbicp_gui()
		self.connect_frame_proj_gui()
		
	def init_flight_tracker(self):
		# For know the startup is hard-coded, this has to be implemented in a seperate tab
		self.flt = Flight_Tracker_Class()
		self.flt.N_cam = 3
		self.flt.mov_nr = 1

		self.flt.start_point = 0
		self.flt.mid_point = 8188
		self.flt.end_point = 16375
		self.flt.trigger_mode = "center"

		self.flt.session_loc = "/media/flyami/New Volume/Flyami_movies/Session_10_Jan_2018_13_32"
		self.flt.mov_name = "mov_"
		self.flt.cam_name = "cam_"
		self.flt.cal_loc = "calibration"
		self.flt.cal_name = "cam_calib.txt"
		self.flt.bckg_loc = "background"
		self.flt.bckg_name = "background_cam_"
		self.flt.bckg_img_format = "tif"
		self.flt.frame_name = "frame_"
		self.flt.frame_img_format = "bmp"
		self.flt.sol_loc = ""
		self.flt.sol_name = ""
		self.flt.model_loc = "/home/flyami/flight_tracker/models"
		self.flt.model_name = "drosophila_melanogaster_simple"

		self.flt.set_parameters()
		self.flt.set_session_parameters()
		self.flt.init_frame_loader()
		self.flt.load_frame(self.flt.start_point)

	def add_model_select_gui(self):
		self.flt.clear_model_list()
		model_location = "/home/flyami/flight_tracker/models"
		self.flt.set_model_loc(model_location)
		model_folder_list = ["drosophila_rigid_wing","drosophila_seg_wing"]
		model_file_name_list = ["drosophila_rigid_wing","drosophila_seg_wing"]
		model_label_list = ["Drosophila Melanogaster (rigid wing)", "Drosophila Melanogaster (flexible wing)"]
		for i, label in enumerate(model_label_list):
			self.flt.add_model(model_folder_list[i],model_file_name_list[i])
			self.model_comboBox.addItem(label)
			self.model_view.add_model(model_folder_list[i])
		self.model_view.loadFLT(self.flt)

	def connect_model_select_gui(self):
		self.model_view.load_model(0)
		self.model_comboBox.currentIndexChanged.connect(self.model_view.load_model)

	def add_focal_grid_gui(self):
		self.flt.N_threads = 8
		self.flt.nx = 256
		self.flt.ny = 256
		self.flt.nz = 256
		self.flt.ds = 0.040
		self.flt.x0 = 0.0
		self.flt.y0 = 0.0
		self.flt.z0 = 0.0
		self.flt.set_grid_param()

	def update_N_threads(self,n_threads):
		self.flt.N_threads = n_threads

	def update_nx(self,nx):
		self.flt.nx = nx

	def update_ny(self,ny):
		self.flt.ny = ny

	def update_nz(self,nz):
		self.flt.nz = nz

	def update_ds(self,ds):
		self.flt.ds = ds

	def update_x0(self,x0):
		self.flt.x0 = x0

	def update_y0(self,y0):
		self.flt.y0 = y0

	def update_z0(self,z0):
		self.flt.z0 = z0

	def connect_focal_grid_gui(self):

		self.n_threads_spin.setMinimum(1)
		self.n_threads_spin.setMaximum(256)
		self.n_threads_spin.setValue(self.flt.N_threads)
		self.n_threads_spin.valueChanged.connect(self.update_N_threads)

		self.nx_spin.setMinimum(2)
		self.nx_spin.setMaximum(1024)
		self.nx_spin.setValue(self.flt.nx)
		self.nx_spin.valueChanged.connect(self.update_nx)

		self.ny_spin.setMinimum(2)
		self.ny_spin.setMaximum(1024)
		self.ny_spin.setValue(self.flt.ny)
		self.ny_spin.valueChanged.connect(self.update_ny)

		self.nz_spin.setMinimum(2)
		self.nz_spin.setMaximum(1024)
		self.nz_spin.setValue(self.flt.nz)
		self.nz_spin.valueChanged.connect(self.update_nz)

		self.ds_spin.setMinimum(0.001)
		self.ds_spin.setMaximum(1.000)
		self.ds_spin.setSingleStep(0.001)
		self.ds_spin.setValue(self.flt.ds)
		self.ds_spin.valueChanged.connect(self.update_ds)

		self.x0_spin.setRange(-100.0,100.0)
		self.x0_spin.setSingleStep(0.01)
		self.x0_spin.setValue(self.flt.x0)
		self.x0_spin.valueChanged.connect(self.update_x0)

		self.y0_spin.setRange(-100.0,100.0)
		self.y0_spin.setSingleStep(0.01)
		self.y0_spin.setValue(self.flt.y0)
		self.y0_spin.valueChanged.connect(self.update_y0)

		self.z0_spin.setRange(-100.0,100.0)
		self.z0_spin.setSingleStep(0.01)
		self.z0_spin.setValue(self.flt.z0)
		self.z0_spin.valueChanged.connect(self.update_z0)

		self.set_fg_par_button.clicked.connect(self.set_focal_grid)

		self.calc_fg_button.clicked.connect(self.calc_focal_grid)

	def set_focal_grid(self):
		self.flt.set_grid_param()
		self.flt.get_grid_param()

	def calc_focal_grid(self):
		self.flt.construct_focal_grid()

	def add_scale_gui(self):
		# This is kind of ugly, need to incorporate this with the model selection in c++
		symbols = ['o','o','o','o','o','o','o']
		adj = np.array([[0,1],[0,2],[0,3],[0,4],[3,5],[4,6]])
		xyz_pos = np.array([(0.0,0.0,0.0),
			(1.0,1.0,1.0),
			(-1.0,-1.0,-1.0),
			(0.5,0.5,-0.5),
			(-0.5,-0.5,0.5),
			(1.0,1.0,-1.0),
			(-1.0,-1.0,1.0)])
		txt_color = np.array([(0,0,255),
							(0,0,255),
							(0,0,255),
							(255,0,0),
							(0,255,0),
							(255,0,0),
							(0,255,0)], 
							dtype=[('red',np.ubyte),('green',np.ubyte),('blue',np.ubyte)])
		lines = np.array([(0,0,255,255,2),
						  (0,0,255,255,2),
						  (255,0,0,255,2),
						  (0,255,0,255,2),
						  (255,0,0,255,2),
						  (0,255,0,255,2)], 
						 dtype=[('red',np.ubyte),('green',np.ubyte),('blue',np.ubyte),('alpha',np.ubyte),('width',float)])
		texts = ["0", "head", "abdomen", "joint L", "joint R", "tip L", "tip R"]
		length_calc = [[0,1],[0,1],[0,2],[3,5],[4,6]]

		self.rawFrameView.loadFLT(self.flt)
		self.rawFrameView.setPos(xyz_pos)
		self.rawFrameView.setAdj(adj)
		self.rawFrameView.setLines(lines)
		self.rawFrameView.setSymbols(symbols)
		self.rawFrameView.setTexts(texts)
		self.rawFrameView.setTextColor(txt_color)
		self.rawFrameView.setLengthCalc(length_calc)
		self.rawFrameView.add_frame(self.flt.start_point)
		self.rawFrameView.add_graph()
		self.rawFrameView.setMouseCallbacks()

		# set lengths table size
		table_texts = ["head scale","thorax scale","abdomen scale", "wing left", "wing right"]
		table_data = [1.0, 1.0, 1.0, 1.0, 1.0]
		self.length_table.setRowCount(2)
		self.length_table.setColumnCount(len(table_texts))
		for i in range(len(table_texts)):
			self.length_table.setItem(0,i,QTableWidgetItem(table_texts[i]))
			self.length_table.setItem(1,i,QTableWidgetItem(str(table_data[i])))
		self.rawFrameView.connect_table(self.length_table)

	def connect_scale_gui(self):
		self.frameSelectRaw.setMinimum(self.flt.start_point)
		self.frameSelectRaw.setMaximum(self.flt.end_point)
		self.frameSelectRaw.valueChanged.connect(self.rawFrameView.update_frame)
		self.set_par_button.clicked.connect(self.rawFrameView.update_model_scale)

	def add_image_segment_gui(self):
		# Also a bit ugly, load from a file might be better
		self.body_thresh = 50
		self.wing_thresh = 20
		self.sigma = 0.05
		self.K = 2000
		self.min_body_size = 50
		self.min_wing_size = 10
		self.tethered = False

		self.segFrameView.loadFLT(self.flt)
		self.segFrameView.setBodyThresh(self.body_thresh)
		self.segFrameView.setWingThresh(self.wing_thresh)
		self.segFrameView.setSigma(self.sigma)
		self.segFrameView.setK(self.K)
		self.segFrameView.setMinBodySize(self.min_body_size)
		self.segFrameView.setMinWingSize(self.min_wing_size)
		self.segFrameView.setTethered(self.tethered)
		self.segFrameView.add_frame(self.flt.start_point)

	def connect_image_segment_gui(self):
		self.frame_seg_spin.setMinimum(self.flt.start_point)
		self.frame_seg_spin.setMaximum(self.flt.end_point)
		self.frame_seg_spin.valueChanged.connect(self.segFrameView.update_frame)

		self.body_th_spin.setMinimum(0)
		self.body_th_spin.setMaximum(255)
		self.body_th_spin.setValue(self.body_thresh)
		self.body_th_spin.valueChanged.connect(self.segFrameView.setBodyThresh)

		self.wing_th_spin.setMinimum(0)
		self.wing_th_spin.setMaximum(255)
		self.wing_th_spin.setValue(self.wing_thresh)
		self.wing_th_spin.valueChanged.connect(self.segFrameView.setWingThresh)

		self.sigma_spin.setMinimum(0.0)
		self.sigma_spin.setMaximum(2.0)
		self.sigma_spin.setSingleStep(0.01)
		self.sigma_spin.setValue(self.sigma)
		self.sigma_spin.valueChanged.connect(self.segFrameView.setSigma)

		self.K_spin.setMinimum(0)
		self.K_spin.setMaximum(5000)
		self.K_spin.setSingleStep(100)
		self.K_spin.setValue(self.K)
		self.K_spin.valueChanged.connect(self.segFrameView.setK)

		self.min_body_spin.setMinimum(0)
		self.min_body_spin.setMaximum(10000)
		self.min_body_spin.setValue(self.min_body_size)
		self.min_body_spin.valueChanged.connect(self.segFrameView.setMinBodySize)

		self.min_wing_spin.setMinimum(0)
		self.min_wing_spin.setMaximum(10000)
		self.min_wing_spin.setValue(self.min_wing_size)
		self.min_wing_spin.valueChanged.connect(self.segFrameView.setMinWingSize)

		self.check_tethered.setChecked(self.tethered)
		self.check_tethered.stateChanged.connect(self.segFrameView.setTethered)

		self.update_button.clicked.connect(self.segFrameView.update_parameters)

	def add_bbox_gui(self):
		self.w_xsi = 0.5
		self.w_theta = 0.5
		self.w_length = 1.0
		self.w_volume = 1.0
		self.cone_angle = 20.0
		self.cone_height = 0.5
		self.bbox_view.loadFLT(self.flt)
		self.bbox_view.setWeigthXsi(self.w_xsi)
		self.bbox_view.setWeightTheta(self.w_theta)
		self.bbox_view.setWeightLength(self.w_length)
		self.bbox_view.setWeightVolume(self.w_volume)
		self.bbox_view.setConeAngle(self.cone_angle)
		self.bbox_view.setConeHeight(self.cone_height)

	def connect_bbox_gui(self):
		self.frame_bbox_spin.setMinimum(self.flt.start_point)
		self.frame_bbox_spin.setMaximum(self.flt.end_point)
		self.frame_bbox_spin.valueChanged.connect(self.bbox_view.loadFrame)

		self.wxsi_spin.setMinimum(0.0)
		self.wxsi_spin.setMaximum(1.0)
		self.wxsi_spin.setSingleStep(0.05)
		self.wxsi_spin.setValue(self.w_xsi)
		self.wxsi_spin.valueChanged.connect(self.bbox_view.setWeigthXsi)

		self.wtheta_spin.setMinimum(0.0)
		self.wtheta_spin.setMaximum(1.0)
		self.wtheta_spin.setSingleStep(0.05)
		self.wtheta_spin.setValue(self.w_theta)
		self.wtheta_spin.valueChanged.connect(self.bbox_view.setWeightTheta)

		self.wlength_spin.setMinimum(0.0)
		self.wlength_spin.setMaximum(1.0)
		self.wlength_spin.setSingleStep(0.05)
		self.wlength_spin.setValue(self.w_length)
		self.wlength_spin.valueChanged.connect(self.bbox_view.setWeightLength)

		self.wvolume_spin.setMinimum(0.0)
		self.wvolume_spin.setMaximum(1.0)
		self.wvolume_spin.setSingleStep(0.05)
		self.wvolume_spin.setValue(self.w_volume)
		self.wvolume_spin.valueChanged.connect(self.bbox_view.setWeightVolume)

		self.cone_angle_spin.setMinimum(0.0)
		self.cone_angle_spin.setMaximum(45.0)
		self.cone_angle_spin.setSingleStep(1.0)
		self.cone_angle_spin.setValue(self.cone_angle)
		self.cone_angle_spin.valueChanged.connect(self.bbox_view.setConeAngle)

		self.cone_height_spin.setMinimum(0.0)
		self.cone_height_spin.setMaximum(1.0)
		self.cone_height_spin.setSingleStep(0.05)
		self.cone_height_spin.setValue(self.cone_angle)
		self.cone_height_spin.valueChanged.connect(self.bbox_view.setConeHeight)

		self.bbox_view_btn.clicked.connect(self.bbox_view.project_image_2_pcl)

		self.blr_view_btn.clicked.connect(self.bbox_view.find_init_state)

	def add_mbicp_gui(self):
		self.mod_proj_view.loadFLT(self.flt)

	def connect_mbicp_gui(self):
		self.model_proj_spin.setMinimum(self.flt.start_point)
		self.model_proj_spin.setMaximum(self.flt.end_point)
		self.model_proj_spin.valueChanged.connect(self.mod_proj_view.load_frame)

		self.show_dest_pts.clicked.connect(self.mod_proj_view.find_dest_pcls)
		self.show_src_pts.clicked.connect(self.mod_proj_view.find_src_pcls)

	def add_frame_proj_gui(self):
		self.frame_proj_view.loadFLT(self.flt)
		self.frame_proj_view.add_frame(self.flt.start_point)

	def connect_frame_proj_gui(self):
		self.frame_proj_spin.setMinimum(self.flt.start_point)
		self.frame_proj_spin.setMaximum(self.flt.end_point)
		self.frame_proj_spin.valueChanged.connect(self.frame_proj_view.load_frame)

		self.calc_proj.clicked.connect(self.frame_proj_view.calculate_projections)

def appMain():
	app = QtWidgets.QApplication(sys.argv)
	mainWindow = FlightTrackerApp()
	mainWindow.show()
	app.exec_()

# -------------------------------------------------------------------------------------------------
if __name__ == '__main__':

	appMain()
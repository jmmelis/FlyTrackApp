from __future__ import print_function
import sys
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QMainWindow, QApplication, QWidget, QAction, QTableWidget,QTableWidgetItem,QVBoxLayout
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import copy
import time

class ImageSegmentWidget(pg.GraphicsWindow):

	def __init__(self, parent=None):
		pg.GraphicsWindow.__init__(self)
		self.setParent(parent)

		self.w_sub = self.addLayout(row=0,col=0)

		self.v_list = []
		self.img_list = []

	def loadFLT(self,flt):
		self.flt = flt
		self.N_cam = self.flt.N_cam
		self.image_size = self.flt.get_image_size()

	def setBodyThresh(self,body_thresh):
		self.body_thresh = body_thresh

	def setWingThresh(self,wing_thresh):
		self.wing_thresh = wing_thresh

	def setSigma(self,sigma):
		self.sigma = sigma

	def setK(self,K):
		self.K = K

	def setMinBodySize(self,min_body_size):
		self.min_body_size = min_body_size

	def setMinWingSize(self,min_wing_size):
		self.min_wing_size = min_wing_size

	def setTethered(self,tethered):
		self.tethered = tethered

	def add_frame(self,frame_nr):
		self.flt.load_frame(frame_nr)
		self.flt.set_body_length(1.0)
		self.flt.set_wing_length(1.0)
		self.flt.set_model_origin(0.0,0.0,0.0)
		self.flt.set_segmentation_param(self.body_thresh,self.wing_thresh,self.sigma,self.K,self.min_body_size,self.min_wing_size,self.tethered)
		self.flt.segment_single_frame()
		frame_list = self.flt.return_segmented_frame()
		for i, frame in enumerate(frame_list):
			self.v_list.append(self.w_sub.addViewBox(row=1,col=i,lockAspect=True))
			frame_jet = self.jet_color(np.transpose(np.flipud(frame)))
			self.img_list.append(pg.ImageItem(frame_jet))
			self.v_list[i].addItem(self.img_list[i])
			self.v_list[i].disableAutoRange('xy')
			self.v_list[i].autoRange()
	
	def update_frame(self,frame_nr):
		self.flt.load_frame(frame_nr)
		self.flt.set_segmentation_param(self.body_thresh,self.wing_thresh,self.sigma,self.K,self.min_body_size,self.min_wing_size,self.tethered)
		self.flt.segment_single_frame()
		frame_list = self.flt.return_segmented_frame()
		for i, frame in enumerate(frame_list):
			frame_jet = self.jet_color(np.transpose(np.flipud(frame)))
			self.img_list[i].setImage(frame_jet)

	def update_parameters(self):
		self.flt.set_segmentation_param(self.body_thresh,self.wing_thresh,self.sigma,self.K,self.min_body_size,self.min_wing_size,self.tethered)
		self.flt.segment_single_frame()
		frame_list = self.flt.return_segmented_frame()
		for i, frame in enumerate(frame_list):
			frame_jet = self.jet_color(np.transpose(np.flipud(frame)))
			self.img_list[i].setImage(frame_jet)

	def jet_color(self,frame):
		norm = mpl.colors.Normalize()
		color_img = plt.cm.jet(norm(frame.astype(float)))*255
		return color_img
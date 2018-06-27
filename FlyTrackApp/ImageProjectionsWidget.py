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

class ImageProjectionsWidget(pg.GraphicsWindow):

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

	def add_frame(self,frame_nr):
		self.flt.load_frame(frame_nr)
		self.flt.segment_single_frame()
		frame_list = self.flt.return_segmented_frame()
		for i, frame in enumerate(frame_list):
			self.v_list.append(self.w_sub.addViewBox(row=1,col=i,lockAspect=True))
			frame_jet = self.jet_color(np.transpose(np.flipud(frame)))
			self.img_list.append(pg.ImageItem(frame_jet))
			self.v_list[i].addItem(self.img_list[i])
			self.v_list[i].disableAutoRange('xy')
			self.v_list[i].autoRange()

	def load_frame(self,frame_nr):
		self.flt.load_frame(frame_nr)

	def calculate_projections(self):
		self.flt.segment_single_frame()
		self.flt.project_single_frame_2_pcl()
		self.flt.find_initial_state()
		frame_list = self.flt.return_projected_model_frames()
		for i, frame in enumerate(frame_list):
			frame_jet = self.jet_color(np.transpose(np.flipud(frame)))
			self.img_list[i].setImage(frame_jet)

	def jet_color(self,frame):
		norm = mpl.colors.Normalize()
		color_img = plt.cm.jet(norm(frame.astype(float)))*255
		return color_img
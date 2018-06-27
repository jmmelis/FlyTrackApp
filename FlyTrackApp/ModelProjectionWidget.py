from __future__ import print_function
import sys
import vtk
from PyQt5 import QtCore, QtGui
from PyQt5 import Qt
import numpy as np
import os
import copy
import time

from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

class ModelProjectionWidget(Qt.QFrame):

	def __init__(self, parent=None):
		Qt.QFrame.__init__(self, parent)

		self.vl = Qt.QVBoxLayout()
		self.vtkWidget = QVTKRenderWindowInteractor(self)
		self.vl.addWidget(self.vtkWidget)

		self.ren = vtk.vtkRenderer()
		self.vtkWidget.GetRenderWindow().AddRenderer(self.ren)
		self.iren = self.vtkWidget.GetRenderWindow().GetInteractor()

		orig_axes = vtk.vtkAxesActor()
		orig_axes.SetAxisLabels(0)

		self.ren.AddActor(orig_axes)

		self.ren.ResetCamera()

		# Set the background color
		self.background = (0.1,0.2,0.4)
		self.ren.SetBackground(*self.background)

		self.setLayout(self.vl)

		self.show()

		self.iren.Initialize()
		self.ren.ResetCameraClippingRange()
		self.vtkWidget.Render()

	def clear_window(self):
		actors = self.ren.GetActors()
		for actor in actors:
			self.ren.RemoveActor(actor)

	def loadFLT(self,flt):
		self.flt = flt

	'''
	def load_stl_files(self,file_loc,file_list):
		os.chdir(file_loc)
		self.stl_list = []
		self.mapper_list = []
		self.stl_actor_list = []
		self.stl_properties = []
		for stl_file in file_list:
			stl_reader = vtk.vtkSTLReader()
			stl_reader.SetFileName(stl_file)
			self.stl_list.append(stl_reader)
			mapper = vtk.vtkPolyDataMapper()
			mapper.ScalarVisibilityOff()
			mapper.SetInputConnection(stl_reader.GetOutputPort())
			self.mapper_list.append(mapper)
			stl_actor = vtk.vtkActor()
			stl_actor.SetMapper(mapper)
			self.stl_actor_list.append(stl_actor)
			self.stl_properties.append(stl_actor.GetProperty())

	def show_stl_files(self):
		M = self.flt.return_transformation_matrices()
		for i, actor in enumerate(self.stl_actor_list):
			M_vtk = vtk.vtkMatrix4x4()
			M_vtk.SetElement(0,0,M[0,i])
			M_vtk.SetElement(0,1,M[1,i])
			M_vtk.SetElement(0,2,M[2,i])
			M_vtk.SetElement(0,3,M[3,i])
			M_vtk.SetElement(1,0,M[4,i])
			M_vtk.SetElement(1,1,M[5,i])
			M_vtk.SetElement(1,2,M[6,i])
			M_vtk.SetElement(1,3,M[7,i])
			M_vtk.SetElement(2,0,M[8,i])
			M_vtk.SetElement(2,1,M[9,i])
			M_vtk.SetElement(2,2,M[10,i])
			M_vtk.SetElement(2,3,M[11,i])
			M_vtk.SetElement(3,0,M[12,i])
			M_vtk.SetElement(3,1,M[13,i])
			M_vtk.SetElement(3,2,M[14,i])
			M_vtk.SetElement(3,3,M[15,i])
			actor.SetUserMatrix(M_vtk)
			actor.Modified()
			self.ren.AddActor(actor)

	def load_model(self,model_folder):
		self.clear_window()
		model_loc = self.flt.get_model_loc()
		file_loc = model_loc + '/' + model_folder
		stl_list = self.flt.return_stl_list()
		self.load_stl_files(file_loc,stl_list)
		self.set_model_state()
		self.show_stl_files()
	'''

	def load_pointcloud(self,pointCloud,pcl_in):
		for k in range(pcl_in.shape[1]):
			#point = pcl_in[:,k]
			point = np.array([pcl_in[1,k],pcl_in[2,k],pcl_in[3,k],pcl_in[0,k]])
			normal = np.array([pcl_in[4,k],pcl_in[5,k],pcl_in[6,k]])
			pointCloud.addPoint(point)
			#pointCloud.addNormal(point,normal,0.1)
		return pointCloud

	def show_pointcloud(self,pcl_in):
		pointCloud = self.VtkPointCloud(np.amax(pcl_in[0,:]))
		pointCloud = self.load_pointcloud(pointCloud,pcl_in)
		self.ren.AddActor(pointCloud.vtkActor)

	def load_frame(self,frame_nr):
		self.clear_window()
		self.flt.load_frame(frame_nr)

	def find_dest_pcls(self):
		self.flt.segment_single_frame()
		self.flt.project_single_frame_2_pcl()
		self.flt.find_initial_state()
		blr_pcl = self.flt.return_blr_pointclouds()
		#M_body = self.flt.return_m_body()
		#M_wing_L = self.flt.return_m_wing_L()
		#M_wing_R = self.flt.return_m_wing_R()
		if blr_pcl.shape[1]>1:
			self.show_pointcloud(blr_pcl)

	def find_src_pcls(self):
		mdl_pcl = self.flt.return_projected_model_pcl()
		if mdl_pcl.shape[1]>1:
			self.show_pointcloud(mdl_pcl)

	'''
	def show_bboxes(self,corner_points):
		N_boxes = corner_points.shape[1]
		for i in range(N_boxes):
			corner_mat = np.empty([8,3])
			for j in range(8):
				corner_mat[j,0] = corner_points[j*3,i]
				corner_mat[j,1] = corner_points[j*3+1,i]
				corner_mat[j,2] = corner_points[j*3+2,i]
			box = self.BoundingBox()
			box.addBox(corner_mat)
			self.ren.AddActor(box.vtkActor)

	def load_stl_files(self,file_loc,file_list):
		os.chdir(file_loc)
		self.stl_list = []
		self.mapper_list = []
		self.stl_actor_list = []
		self.stl_properties = []
		for stl_file in file_list:
			stl_reader = vtk.vtkSTLReader()
			stl_reader.SetFileName(stl_file)
			self.stl_list.append(stl_reader)
			mapper = vtk.vtkPolyDataMapper()
			mapper.ScalarVisibilityOff()
			mapper.SetInputConnection(stl_reader.GetOutputPort())
			self.mapper_list.append(mapper)
			stl_actor = vtk.vtkActor()
			stl_actor.SetMapper(mapper)
			self.stl_actor_list.append(stl_actor)
			self.stl_properties.append(stl_actor.GetProperty())

	def show_stl_files(self):
		M = self.flt.return_transformation_matrices()
		#M_list = self.calculate_icp_transform()
		scale_vec = self.flt.return_model_scale()
		for i, actor in enumerate(self.stl_actor_list):
			actor.SetScale(scale_vec[i],scale_vec[i],scale_vec[i])
			actor.Modified()
			M_vtk = vtk.vtkMatrix4x4()
			#M_vtk.SetElement(0,0,M_list[i][0,0])
			#M_vtk.SetElement(0,1,M_list[i][0,1])
			#M_vtk.SetElement(0,2,M_list[i][0,2])
			#M_vtk.SetElement(0,3,M_list[i][0,3])
			#M_vtk.SetElement(1,0,M_list[i][1,0])
			#M_vtk.SetElement(1,1,M_list[i][1,1])
			#M_vtk.SetElement(1,2,M_list[i][1,2])
			#M_vtk.SetElement(1,3,M_list[i][1,3])
			#M_vtk.SetElement(2,0,M_list[i][2,0])
			#M_vtk.SetElement(2,1,M_list[i][2,1])
			#M_vtk.SetElement(2,2,M_list[i][2,2])
			#M_vtk.SetElement(2,3,M_list[i][2,3])
			#M_vtk.SetElement(3,0,M_list[i][3,0])
			#M_vtk.SetElement(3,1,M_list[i][3,1])
			#M_vtk.SetElement(3,2,M_list[i][3,2])
			#M_vtk.SetElement(3,3,M_list[i][3,3])
			M_vtk.SetElement(0,0,M[0,i])
			M_vtk.SetElement(0,1,M[1,i])
			M_vtk.SetElement(0,2,M[2,i])
			M_vtk.SetElement(0,3,M[3,i])
			M_vtk.SetElement(1,0,M[4,i])
			M_vtk.SetElement(1,1,M[5,i])
			M_vtk.SetElement(1,2,M[6,i])
			M_vtk.SetElement(1,3,M[7,i])
			M_vtk.SetElement(2,0,M[8,i])
			M_vtk.SetElement(2,1,M[9,i])
			M_vtk.SetElement(2,2,M[10,i])
			M_vtk.SetElement(2,3,M[11,i])
			M_vtk.SetElement(3,0,M[12,i])
			M_vtk.SetElement(3,1,M[13,i])
			M_vtk.SetElement(3,2,M[14,i])
			M_vtk.SetElement(3,3,M[15,i])
			actor.SetUserMatrix(M_vtk)
			actor.Modified()
			self.ren.AddActor(actor)

	def calculate_icp_transform(self):
		M_init = self.flt.return_transformation_matrices()
		M_align = self.flt.perform_single_frame_icp()
		M_out_list = []
		for i in range(M_init.shape[1]):
			M_i = np.empty([4,4])
			M_a = np.empty([4,4])
			for j in range(4):
				for k in range(4):
					M_i[j][k] = M_init[j*4+k,i]
					M_a[j][k] = M_align[j*4+k,i]
			M_out = np.zeros([4,4])
			M_out[0:3,0:3] = np.dot(M_i[0:3,0:3],M_a[0:3,0:3])
			#M_out[3,0:3] = np.dot(M_i[0:3,0:3],M_a[3,0:3])+M_i[3,0:3]
			M_out[3,0:3] = M_i[3,0:3]
			M_out[3,3] = 1.0
			M_out_list.append(M_out)
		return M_out_list

	def load_model(self,model_folder):
		#self.clear_window()
		model_loc = self.flt.get_model_loc()
		file_loc = model_loc + '/' + model_folder
		stl_list = self.flt.return_stl_list()
		self.load_stl_files(file_loc,stl_list)
		self.flt.set_model_init_state()
		self.show_stl_files()
	'''

	class VtkPointCloud:
		def __init__(self,scalar_range):
			self.vtkPolyData = vtk.vtkPolyData()
			self.clearPoints()
			mapper = vtk.vtkPolyDataMapper()
			mapper.SetInputData(self.vtkPolyData)
			mapper.SetColorModeToDefault()
			mapper.SetScalarRange(0.0,scalar_range)
			mapper.SetScalarVisibility(1)
			self.vtkActor = vtk.vtkActor()
			self.vtkActor.SetMapper(mapper)

		def addPoint(self,point):
			pointID = self.vtkPoints.InsertNextPoint(point[0:3])
			self.vtkDepth.InsertNextValue(point[3])
			self.vtkCells.InsertNextCell(1)
			self.vtkCells.InsertCellPoint(pointID)
			self.vtkCells.Modified()
			self.vtkPoints.Modified()
			self.vtkDepth.Modified()

		def addNormal(self,point,normal,scale):
			pointID1 = self.vtkPoints.InsertNextPoint(point[0:3])
			pointID2 = self.vtkPoints.InsertNextPoint([point[0]+scale*normal[0],point[1]+scale*normal[1],point[2]+scale*normal[2]])
			self.vtkDepth.InsertNextValue(point[3])
			self.vtkCells.InsertNextCell(2)
			self.vtkCells.InsertCellPoint(pointID1)
			self.vtkCells.InsertCellPoint(pointID2)
			self.vtkCells.Modified()
			self.vtkPoints.Modified()
			self.vtkDepth.Modified()

		def clearPoints(self):
			self.vtkPoints = vtk.vtkPoints()
			self.vtkCells = vtk.vtkCellArray()
			self.vtkDepth = vtk.vtkDoubleArray()
			self.vtkDepth.SetName('DepthArray')
			self.vtkPolyData.SetPoints(self.vtkPoints)
			self.vtkPolyData.SetVerts(self.vtkCells)
			#self.vtkPolyData.SetLines(self.vtkCells)
			self.vtkPolyData.GetPointData().SetScalars(self.vtkDepth)
			self.vtkPolyData.GetPointData().SetActiveScalars('DepthArray')

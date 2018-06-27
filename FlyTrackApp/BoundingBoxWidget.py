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

class BBoxWidget(Qt.QFrame):

	def __init__(self, parent=None):
		Qt.QFrame.__init__(self, parent)

		self.vl = Qt.QVBoxLayout()
		self.vtkWidget = QVTKRenderWindowInteractor(self)
		self.vl.addWidget(self.vtkWidget)

		self.ren = vtk.vtkRenderer()
		self.vtkWidget.GetRenderWindow().AddRenderer(self.ren)
		self.iren = self.vtkWidget.GetRenderWindow().GetInteractor()

		'''
		# Create source
		source = vtk.vtkSphereSource()
		source.SetCenter(0, 0, 0)
		source.SetRadius(5.0)

		# Create a mapper
		mapper = vtk.vtkPolyDataMapper()
		mapper.SetInputConnection(source.GetOutputPort())

		# Create an actor
		actor = vtk.vtkActor()
		actor.SetMapper(mapper)
		'''

		# Place axes 
		#transform = vtk.vtkTransform()
		#transform.Translate(1.0, 0.0 ,0.0)

		orig_axes = vtk.vtkAxesActor()
		orig_axes.SetAxisLabels(0)

		self.ren.AddActor(orig_axes)

		#self.ren.AddActor(actor)

		self.ren.ResetCamera()

		# Set the background color
		self.background = (0.1,0.2,0.4)
		self.ren.SetBackground(*self.background)

		self.setLayout(self.vl)

		self.show()

		self.iren.Initialize()
		#self.iren.Start()
		self.ren.ResetCameraClippingRange()
		self.vtkWidget.Render()

		#------------------------------
		'''

		# stl model parameters
		self.model_name = ""
		self.stl_list = []
		self.model_loc = ""
		self.stl_src = []
		self.stl_actors = []

		# point parameters
		self.pointcloud_list = []
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

	def show_axes(self,M):
		axes = vtk.vtkAxesActor()
		axes.SetAxisLabels(0)
		M_vtk = vtk.vtkMatrix4x4()
		M_vtk.SetElement(0,0,M[0,0])
		M_vtk.SetElement(0,1,M[0,1])
		M_vtk.SetElement(0,2,M[0,2])
		M_vtk.SetElement(0,3,M[0,3])
		M_vtk.SetElement(1,0,M[1,0])
		M_vtk.SetElement(1,1,M[1,1])
		M_vtk.SetElement(1,2,M[1,2])
		M_vtk.SetElement(1,3,M[1,3])
		M_vtk.SetElement(2,0,M[2,0])
		M_vtk.SetElement(2,1,M[2,1])
		M_vtk.SetElement(2,2,M[2,2])
		M_vtk.SetElement(2,3,M[2,3])
		M_vtk.SetElement(3,0,M[3,0])
		M_vtk.SetElement(3,1,M[3,1])
		M_vtk.SetElement(3,2,M[3,2])
		M_vtk.SetElement(3,3,M[3,3])
		axes.SetUserMatrix(M_vtk)
		axes.Modified()
		self.ren.AddActor(axes)

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

	def clear_window(self):
		actors = self.ren.GetActors()
		for actor in actors:
			self.ren.RemoveActor(actor)

	def loadFLT(self,flt):
		self.flt = flt

	def setWeigthXsi(self,w_xsi):
		self.flt.set_weight_xsi(w_xsi)

	def setWeightTheta(self,w_theta):
		self.flt.set_weight_theta(w_theta)

	def setWeightLength(self,w_length):
		self.flt.set_weight_length(w_length)

	def setWeightVolume(self,w_volume):
		self.flt.set_weight_volume(w_volume)

	def setConeAngle(self,cone_angle):
		self.flt.set_cone_angle(cone_angle)

	def setConeHeight(self,cone_height):
		self.flt.set_cone_height(cone_height)

	def loadFrame(self,frame_nr):
		self.flt.load_frame(frame_nr)

	def project_image_2_pcl(self):
		self.clear_window()
		self.flt.segment_single_frame()
		self.flt.project_single_frame_2_pcl()
		seg_pcl = self.flt.return_segment_pointcloud()
		print(seg_pcl.shape)
		if seg_pcl.shape[1]>0:
			self.show_pointcloud(seg_pcl)
			bbox_mat = self.flt.return_bounding_boxes()
			self.show_bboxes(bbox_mat)

	def find_init_state(self):
		self.clear_window()
		self.flt.segment_single_frame()
		self.flt.project_single_frame_2_pcl()
		self.flt.find_initial_state()
		blr_pcl = self.flt.return_blr_pointclouds()
		print(blr_pcl.shape)
		M_body = self.flt.return_m_body()
		print(M_body)
		M_wing_L = self.flt.return_m_wing_L()
		print(M_wing_L)
		M_wing_R = self.flt.return_m_wing_R()
		print(M_wing_R)
		if blr_pcl.shape[1]>1:
			self.show_pointcloud(blr_pcl)
			bbox_mat = self.flt.return_blr_bounding_boxes()
			self.show_bboxes(bbox_mat)
			self.load_model("drosophila_rigid_wing")
			self.show_axes(M_body)
			self.show_axes(M_wing_L[:,0:4])
			self.show_axes(M_wing_R[:,0:4])

	'''
	def set_state(self,M_body,M_wing_L,M_wing_R):
		state_mat = np.zeros([7,5])
		state_mat[0:4,0] = self.get_quaternion(M_body)
		state_mat[4,0] = M_body[3,0]
		state_mat[5,0] = M_body[3,1]
		state_mat[6,0] = M_body[3,2]
		state_mat[0:4,1] = self.get_quaternion(M_body)
		state_mat[4,1] = M_body[3,0]
		state_mat[5,1] = M_body[3,1]
		state_mat[6,1] = M_body[3,2]
		state_mat[0:4,2] = self.get_quaternion(M_body)
		state_mat[4,2] = M_body[3,0]
		state_mat[5,2] = M_body[3,1]
		state_mat[6,2] = M_body[3,2]
		state_mat[0:4,3] = self.get_quaternion(M_wing_L)
		state_mat[4,3] = M_wing_L[3,0]
		state_mat[5,3] = M_wing_L[3,1]
		state_mat[6,3] = M_wing_L[3,2]
		state_mat[0:4,4] = self.get_quaternion(M_wing_R)
		state_mat[4,4] = M_wing_R[3,0]
		state_mat[5,4] = M_wing_R[3,1]
		state_mat[6,4] = M_wing_R[3,2]
		self.flt.set_state()

	def get_quaternion(M):
		q = np.zeros([4,1])
		q[0] = 1.0
		if M[3,3] == 1.0:
			q[0] = 0.5*np.sqrt(M[0,0]+M[1,1]+M[2,2])
			q[1] = (M[2,3]-M[3,2])/(4.0*q[0])
			q[2] = (M[3,1]-M[1,3])/(4.0*q[0])
			q[3] = (M[1,2]-M[2,1])/(4.0*q[0])
			q = (1.0/np.linalg.norm(q))*q
		return q

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

	def show_model(self):
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

	class BoundingBox:
		def __init__(self):
			self.mapper = vtk.vtkPolyDataMapper()
			self.vtkActor = vtk.vtkActor()
			self.vtkActor.SetMapper(self.mapper)

		def addBox(self,corner_points):
			# Add a bounding box
			points = vtk.vtkPoints()
			points.SetNumberOfPoints(8)
			points.SetPoint(0,corner_points[0,0],corner_points[0,1],corner_points[0,2])
			points.SetPoint(1,corner_points[1,0],corner_points[1,1],corner_points[1,2])
			points.SetPoint(2,corner_points[2,0],corner_points[2,1],corner_points[2,2])
			points.SetPoint(3,corner_points[3,0],corner_points[3,1],corner_points[3,2])
			points.SetPoint(4,corner_points[4,0],corner_points[4,1],corner_points[4,2])
			points.SetPoint(5,corner_points[5,0],corner_points[5,1],corner_points[5,2])
			points.SetPoint(6,corner_points[6,0],corner_points[6,1],corner_points[6,2])
			points.SetPoint(7,corner_points[7,0],corner_points[7,1],corner_points[7,2])
			lines = vtk.vtkCellArray()
			lines.InsertNextCell(5)
			lines.InsertCellPoint(0)
			lines.InsertCellPoint(1)
			lines.InsertCellPoint(2)
			lines.InsertCellPoint(3)
			lines.InsertCellPoint(0)
			lines.InsertNextCell(5)
			lines.InsertCellPoint(4)
			lines.InsertCellPoint(5)
			lines.InsertCellPoint(6)
			lines.InsertCellPoint(7)
			lines.InsertCellPoint(4)
			lines.InsertNextCell(5)
			lines.InsertCellPoint(0)
			lines.InsertCellPoint(4)
			lines.InsertCellPoint(7)
			lines.InsertCellPoint(3)
			lines.InsertCellPoint(0)
			lines.InsertNextCell(5)
			lines.InsertCellPoint(1)
			lines.InsertCellPoint(5)
			lines.InsertCellPoint(6)
			lines.InsertCellPoint(2)
			lines.InsertCellPoint(1)
			lines.InsertNextCell(5)
			lines.InsertCellPoint(0)
			lines.InsertCellPoint(1)
			lines.InsertCellPoint(5)
			lines.InsertCellPoint(4)
			lines.InsertCellPoint(0)
			lines.InsertNextCell(5)
			lines.InsertCellPoint(3)
			lines.InsertCellPoint(2)
			lines.InsertCellPoint(6)
			lines.InsertCellPoint(7)
			lines.InsertCellPoint(3)
			polygon = vtk.vtkPolyData()
			polygon.SetPoints(points)
			polygon.SetLines(lines)
			self.mapper.SetInputData(polygon)
			self.mapper.Update()
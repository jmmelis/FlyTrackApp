import vtk
import sys
import os
import time
import numpy as np

# flight tracker visualization class

class FlightTrackerVisualization:

	def __init__(self):

		# window parameters
		self.window_name = "Model"
		self.background = (0.1,0.2,0.4)
		self.window_sz = (600, 600)

		# stl model parameters
		self.model_name = ""
		self.stl_list = []
		self.model_loc = ""
		self.stl_src = []
		self.stl_actors = []

		# point parameters
		self.pointcloud_list = []

		# Create the Renderer, RenderWindow, and RenderWindowInteractor
		self.ren = vtk.vtkRenderer()
		self.ren_win = vtk.vtkRenderWindow()
		self.ren_win.AddRenderer(self.ren)
		self.iren = vtk.vtkRenderWindowInteractor()
		self.iren.SetRenderWindow(self.ren_win)

		# Set the background color and window size
		self.ren_win.SetWindowName(self.window_name)
		self.ren.SetBackground(*self.background)
		self.ren_win.SetSize(*self.window_sz)

		# Render
		self.iren.Initialize()
		self.ren.ResetCameraClippingRange()
		self.ren_win.Render()

	def load_model(self,model_name,model_loc,stl_list):

		self.model_name = model_name
		self.stl_list = stl_list
		self.model_loc = model_loc + '/' + model_name
		self.ren_win.SetWindowName(model_name)

		os.chdir(self.model_loc)
		
		for stl_file in stl_list:
			sr = vtk.vtkSTLReader()
			sr.SetFileName(stl_file)
			self.stl_src.append(sr)
			stl_mapper = vtk.vtkPolyDataMapper()
			stl_mapper.ScalarVisibilityOff()
			stl_mapper.SetInputConnection(sr.GetOutputPort())
			stl_actor = vtk.vtkActor()
			stl_actor.SetMapper(stl_mapper)
			self.stl_actors.append(stl_actor)
			stl_props = stl_actor.GetProperty()
			stl_actor.SetPosition(0,0,0)
			stl_props.SetInterpolationToGouraud()
			stl_mapper.Update()
			self.ren.AddActor(stl_actor)

		self.ren_win.Render()

	def set_state_model(self,state,parents,scale):

		for i in range(state.shape[1]):

			old_val = -1
			j = 0

			transformation = vtk.vtkTransform()

			while parents[i,j] > old_val:
				ind = parents[i,j]
				elem_mat = vtk.vtkMatrix4x4()
				elem_mat.SetElement(0,0,(2.0*state[0,ind]**2-1.0+2.0*state[1,ind]**2))
				elem_mat.SetElement(0,1,(2.0*state[1,ind]*state[2,ind]+2.0*state[0,ind]*state[3,ind]))
				elem_mat.SetElement(0,2,(2.0*state[1,ind]*state[3,ind]-2.0*state[0,ind]*state[2,ind]))
				elem_mat.SetElement(1,0,(2.0*state[1,ind]*state[2,ind]-2.0*state[0,ind]*state[3,ind]))
				elem_mat.SetElement(1,1,(2.0*state[0,ind]**2-1.0+2.0*state[2,ind]**2))
				elem_mat.SetElement(1,2,(2.0*state[2,ind]*state[3,ind]+2.0*state[0,ind]*state[1,ind]))
				elem_mat.SetElement(2,0,(2.0*state[1,ind]*state[3,ind]+2.0*state[0,ind]*state[2,ind]))
				elem_mat.SetElement(2,1,(2.0*state[2,ind]*state[3,ind]-2.0*state[0,ind]*state[1,ind]))
				elem_mat.SetElement(2,2,(2.0*state[0,ind]**2-1.0+2.0*state[3,ind]**2))
				elem_mat.SetElement(0,3,state[4,ind]*scale[ind])
				elem_mat.SetElement(1,3,state[5,ind]*scale[ind])
				elem_mat.SetElement(2,3,state[6,ind]*scale[ind])

				transformation.Concatenate(elem_mat)
				old_val = parents[i,j]
				j+=1

			self.stl_actors[i].SetUserMatrix(transformation.GetMatrix())

		self.ren_win.Render()

	def add_pointcloud(self,pcl_in):

		N = pcl_in.shape[1]

		#points = vtk.vtkPointSource()
		points = vtk.vtkPoints()
		points.SetNumberOfPoints(N)
		polydata = vtk.vtkPolyData()

		for i in range(N):
			points.InsertNextPoint(pcl_in[:,i])
			#points.SetRadius(0.005)

		polydata.SetPoints(points)
		mapper = vtk.vtkPolyDataMapper()
		mapper.SetInputData(polydata)
		#mapper.SetInputData(points)
		#mapper.SetInputConnection(points.GetOutputPort())
		actor = vtk.vtkActor()
		actor.SetMapper(mapper)
		self.ren.AddActor(actor)

		#for i in range(N_points):
		#	#point = vtk.vtkSphereSource()
		#	#point.SetCenter(pcl_in[:,i])
		#	#point.SetRadius(0.005)
		#	point = vtk.vtkPointSource()
		#	point.SetCenter(pcl_in[:,i])
		#	point.SetNumberOfPoints(1);
		#	mapper = vtk.vtkPolyDataMapper()
		#	mapper.ScalarVisibilityOff()
		#	mapper.SetInputConnection(point.GetOutputPort())
		#	actor = vtk.vtkActor()
		#	actor.SetMapper(mapper)
		#	props = actor.GetProperty()
		#	self.ren.AddActor(actor)

	def start_interaction_window(self):
		self.ren_win.Render()
		self.iren.Start()

	def kill_interaction_window(self):
		del self.ren_win, self.iren

	def load_pointcloud(self,pointCloud,pcl_in):
		for k in range(pcl_in.shape[1]):
			point = pcl_in[:,k]
			pointCloud.addPoint(point)

		return pointCloud

	def show_pointcloud(self,pcl_in):
		pointCloud = self.VtkPointCloud(np.amax(pcl_in[3,:]))
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

		def clearPoints(self):
			self.vtkPoints = vtk.vtkPoints()
			self.vtkCells = vtk.vtkCellArray()
			self.vtkDepth = vtk.vtkDoubleArray()
			self.vtkDepth.SetName('DepthArray')
			self.vtkPolyData.SetPoints(self.vtkPoints)
			self.vtkPolyData.SetVerts(self.vtkCells)
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





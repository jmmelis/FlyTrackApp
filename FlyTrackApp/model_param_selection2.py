import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
import sys
import os
import numpy as np
import copy
import vtk
import time

pg.mkQApp()

## Define main window class from template
path = '/home/flyami/flight_tracker'
uiFile = os.path.join(path, 'par_sel.ui')
WindowTemplate, TemplateBaseClass = pg.Qt.loadUiType(uiFile)

from flight_tracker_class import Flight_Tracker_Class# Select model parameters using dragpoints

class Graph(pg.GraphItem):
    def __init__(self,graph_nr):
    	self.graph_nr = graph_nr
        self.dragPoint = None
        self.dragOffset = None
        self.textItems = []
        pg.GraphItem.__init__(self)
        self.scatter.sigClicked.connect(self.clicked)
        self.onMouseDragCb = None
        
    def setData(self, **kwds):
        self.text = kwds.pop('text', [])
        self.data = copy.deepcopy(kwds)
        
        if 'pos' in self.data:
            npts = self.data['pos'].shape[0]
            self.data['data'] = np.empty(npts, dtype=[('index', int)])
            self.data['data']['index'] = np.arange(npts)
        self.setTexts(self.text,self.data)
        self.updateGraph()
        
    def setTexts(self, text, data):
        for i in self.textItems:
            i.scene().removeItem(i)
        self.textItems = []
        #for t in text:
        for i,t in enumerate(text):
            item = pg.TextItem(t)
            if len(data.keys())>0:
            	item.setColor(data['textcolor'][i])
            self.textItems.append(item)
            item.setParentItem(self)
        
    def updateGraph(self):
        pg.GraphItem.setData(self, **self.data)
        for i,item in enumerate(self.textItems):
            item.setPos(*self.data['pos'][i])

    def setOnMouseDragCallback(self, callback):
    	self.onMouseDragCb = callback
        
    def mouseDragEvent(self, ev):
        if ev.button() != QtCore.Qt.LeftButton:
            ev.ignore()
            return
        
        if ev.isStart():
            # We are already one step into the drag.
            # Find the point(s) at the mouse cursor when the button was first 
            # pressed:
            pos = ev.buttonDownPos()
            pts = self.scatter.pointsAt(pos)
            if len(pts) == 0:
                ev.ignore()
                return
            self.dragPoint = pts[0]
            ind = pts[0].data()[0]
            self.dragOffset = self.data['pos'][ind] - pos
        elif ev.isFinish():
            self.dragPoint = None
            return
        else:
            if self.dragPoint is None:
                ev.ignore()
                return
        
        ind = self.dragPoint.data()[0]
        self.data['pos'][ind] = ev.pos() + self.dragOffset
        self.updateGraph()
        ev.accept()
        if self.onMouseDragCb:
        	PosData = self.data['pos'][ind]
        	PosData = np.append(PosData,ind)
        	PosData = np.append(PosData,self.graph_nr)
        	self.onMouseDragCb(PosData)
        
    def clicked(self, pts):
        print("clicked: %s" % pts)

class model_param_GUI(TemplateBaseClass):

	def __init__(self,flt,symbols,adj,xyz_pos,txt_color,lines,texts,length_calc,start_frame):

		self.flt = flt

		self.win = QtGui.QMainWindow()
		self.win.setWindowTitle('Select model parameters')

		self.cw = QtGui.QWidget()

		self.layout = QtGui.QGridLayout()
		self.cw.setLayout(self.layout)
		self.win.setCentralWidget(cw)

		self.layout.addWidget

		self.w = pg.GraphicsWindow(size=(1200,600),border=True)
		self.w.setWindowTitle('body and wing length selection')
		#pg.setConfigOptions(antialias=True)
		self.w_sub = self.w.addLayout(row=0,col=0)

		self.N_cam = flt.N_cam

		self.N_drag_points = xyz_pos.shape[0]

		self.image_size = flt.get_image_size()

		self.xyz_pos = xyz_pos

		self.pos = []

		for i in range(self.N_cam):
			self.pos.append(np.empty([self.N_drag_points,2]))
			for j in range(self.N_drag_points):
				uv_points = self.flt.convert_3D_point_2_uv(xyz_pos[j,0],xyz_pos[j,1],xyz_pos[j,2])
				self.pos[i][j,0] = uv_points[i,0]
				self.pos[i][j,1] = self.image_size[i,1]-uv_points[i,1]

		self.adj = adj
		self.symbols = symbols
		self.txt_color = txt_color
		self.lines = lines
		self.texts = texts
		self.length_calc = length_calc
		self.lengths = []
		for i in range(len(self.length_calc)):
			self.lengths.append(0.0)

		self.v_list = []
		self.img_list = []
		self.graph_list = []

		self.add_frame(start_frame)

		self.update_frame(start_frame)

		self.add_graph()

		def onMouseDragCallback(data):
			self.update_graph(data)

		for i in range(self.N_cam):
			self.graph_list[i].setOnMouseDragCallback(onMouseDragCallback)

		# Add spinbox
		self.frame_sel = pg.SpinBox(value=0, int=True, minStep=1, step=1)
		self.w_sub.addWidget(self.frame_sel)
		self.frame_sel.sigValueChanged.connect(valueChanged(self.update_frame))


	def add_frame(self,frame_nr):
		self.flt.load_frame(frame_nr)
		frame_list = self.flt.get_frame()
		for i, frame in enumerate(frame_list):
			self.v_list.append(self.w_sub.addViewBox(row=1,col=i,lockAspect=True))
			self.img_list.append(pg.ImageItem(np.transpose(np.flipud(frame))))
			self.v_list[i].addItem(self.img_list[i])
			self.v_list[i].disableAutoRange('xy')
			self.v_list[i].autoRange()

	def update_frame(self,frame_nr):
		self.flt.load_frame(frame_nr)
		frame_list = self.flt.get_frame()
		for i, frame in enumerate(frame_list):
			self.img_list[i].setImage(np.transpose(np.flipud(frame)))

	def add_graph(self):
		for i in range(self.N_cam):
			self.graph_list.append(Graph(i))
			self.v_list[i].addItem(self.graph_list[i])
			self.graph_list[i].setData(pos=self.pos[i], adj=self.adj, pen=self.lines, size=3, symbol=self.symbols, pxMode=False, text=self.texts, textcolor=self.txt_color)

	def calculate_lengths(self):
		for i,calc in enumerate(self.length_calc):
			self.lengths[i] = np.sqrt((self.xyz_pos[calc[0],0]-self.xyz_pos[calc[1],0])**2+
									(self.xyz_pos[calc[0],1]-self.xyz_pos[calc[1],1])**2+
									(self.xyz_pos[calc[0],2]-self.xyz_pos[calc[1],2])**2)

	def update_graph(self,drag_data):

		# Calculate new 3D position and new uv positions

		cam_nr = int(drag_data[3])
		point_nr = int(drag_data[2])
		u_now = int(drag_data[0])
		v_now = int(self.image_size[cam_nr,1]-drag_data[1])
		u_prev = int(self.pos[cam_nr][point_nr,0])
		v_prev = int(self.image_size[cam_nr,1]-self.pos[cam_nr][point_nr,1])
		x_prev = self.xyz_pos[point_nr,0]
		y_prev = self.xyz_pos[point_nr,1]
		z_prev = self.xyz_pos[point_nr,2]

		new_pos = flt.drag_point_3D(cam_nr, u_now, v_now, u_prev, v_prev, x_prev, y_prev, z_prev)

		# Update new 3D position and new uv positions

		self.xyz_pos[point_nr,:] = new_pos[:,0]

		self.calculate_lengths()
		
		for i in range(self.N_cam):
			self.pos[i][point_nr,0] = new_pos[0,i+1]
			self.pos[i][point_nr,1] = self.image_size[i,1]-new_pos[1,i+1]

		for i in range(self.N_cam):
			self.graph_list[i].data['pos'][point_nr,:] = self.pos[i][point_nr,:]
			self.graph_list[i].updateGraph()
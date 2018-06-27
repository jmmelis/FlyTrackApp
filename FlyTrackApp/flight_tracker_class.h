#ifndef FLIGHT_TRACKER_CLASS_H
#define FLIGHT_TRACKER_CLASS_H

#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include <iostream>
#include <string>
#include <stdint.h>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <armadillo>

// Data structures
#include "session_param.h"
#include "frames.h"
#include "vox_grid.h"
#include "model.h"

// Sub-classes
#include "frame_loader.h"
#include "focal_grid.h"
#include "model_class.h"
#include "image_segmentation.h"
#include "find_initial_state.h"
#include "multi_body_icp.h"

namespace p = boost::python;
namespace np = boost::python::numpy;

using namespace std;

class FLT
{
	
	public:

		// structures
		struct session_param session;
		struct frames frame_batch;
		struct vox_grid voxel_grid;
		struct model insect_model;

		// Classes
		FrameLoader fl;
		FocalGrid fg;
		ModelClass mdl;
		ImagSegm seg;
		InitState init;
		MultiBodyICP mbicp;

		// Number of threads (cores):
		int N_threads = 0;

		// File locations

		string session_loc;
		string mov_name;
		string cam_name;
		string cal_loc;
		string cal_name;
		string bckg_loc;
		string bckg_name;
		string bckg_img_format;
		string frame_name;
		string frame_img_format;
		string sol_loc;
		string sol_name;
		string model_loc;

		// FrameLoader parameters

		int N_cam;
		int mov_nr;

		int start_point;
		int mid_point;
		int end_point;
		string trigger_mode;

		vector<tuple<int, int>> image_size;
		vector<arma::Col<int>> bckg_images;
		vector<arma::Col<int>> single_raw_frame;
		vector<arma::Mat<int>> raw_frames;
		vector<int> chrono_frame_nr;

		// FocalGrid parameters

		int N_x;
		int N_y;
		int N_z;
		double ds;
		double x_0;
		double y_0;
		double z_0;

		vector<arma::Col<double>> calibration_parameters;
		vector<arma::Mat<double>> X_xyz;
		vector<arma::Mat<double>> X_uv;
		arma::Mat<double> uv_offset;
		vector<voxel_prop> voxel_list;


		FLT();

		// Set and Get functions
		void SetSessionLoc(string SessionLoc);
		string GetSessionLoc();
		void SetMovName(string MovName);
		string GetMovName();
		void SetCamName(string CamName);
		string GetCamName();
		void SetCalLoc(string CalLoc);
		string GetCalLoc();
		void SetCalName(string CalName);
		string GetCalName();
		void SetNrOfCam(int CamNr);
		int GetNrOfCam();
		void SetMovNr(int MovNr);
		int GetMovNr();
		void SetBckgLoc(string BckgLoc);
		string GetBckgLoc();
		void SetBckgName(string BckgName);
		string GetBckgName();
		void SetBckgImageFormat(string BckgImageFormat);
		string GetBckgImageFormat();
		void SetFrameName(string FrameName);
		string GetFrameName();
		void SetFrameImageFormat(string FrameImageFormat);
		string GetFrameImageFormat();
		void SetStartPoint(int StartPoint);
		int GetStartPoint();
		void SetMidPoint(int MidPoint);
		int GetMidPoint();
		void SetEndPoint(int EndPoint);
		int GetEndPoint();
		void SetTriggerMode(string TriggerMode);
		string GetTriggerMode();
		void SetNx(int new_Nx);
		int GetNx();
		void SetNy(int new_Ny);
		int GetNy();
		void SetNz(int new_Nz);
		int GetNz();
		void SetDs(double new_ds);
		double GetDs();
		void SetX0(double new_x0);
		double GetX0();
		void SetY0(double new_y0);
		double GetY0();
		void SetZ0(double new_z0);
		double GetZ0();
		void SetSolLoc(string SolLoc);
		string GetSolLoc();
		void SetSolName(string SolName);
		string GetSolName();
		void SetNumberOfThreads(int N);
		int GetNumberOfThreads();
		void SetModelLoc(string ModelLoc);
		string GetModelLoc();
		void AddModel(string ModelFolder, string ModelName);
		void ClearModelList();

		// Set session parameters
		bool SetSessionParam();

		// FrameLoader functions
		bool InitFrameLoader();
		bool LoadSingleFrame(int frame_nr);
		bool LoadFrameBatch(int batch_nr, int start_frame, int end_frame);
		np::ndarray ReturnSingleFrame(int cam_nr);
		np::ndarray ReturnImageSize();

		// FocalGrid functions
		bool SetGridParameters();
		bool ConstructGrid();
		np::ndarray XYZ2UV(double x_in, double y_in, double z_in);
		np::ndarray DragPoint3D(int cam_nr, int u_now, int v_now, int u_prev, int v_prev, double x_prev, double y_prev, double z_prev);

		// ModelClass functions
		bool LoadModel(int model_ind);
		bool SetModelScale(p::object& scale_list);
		bool SetModelOrigin(double x_in, double y_in, double z_in);
		bool SetBodyLength(double body_length);
		bool SetWingLength(double wing_length);
		bool SetInitState();
		//np::ndarray ReturnModelPointcloud();
		np::ndarray ReturnParentStructure();
		np::ndarray ReturnModelState();
		p::list ReturnSTLFileList();
		np::ndarray ReturnModelScale();
		np::ndarray ReturnTransformationMatrices();

		// ImagSegm functions
		void SetImagSegmParam(int body_thresh, int wing_thresh, double Sigma, int K, int min_body_size, int min_wing_size, bool Tethered);
		bool SegmentSingleFrame();
		bool SegmentFrameBatch();
		np::ndarray ReturnSegmentedFrame(int cam_nr);

		// InitState functions
		void SetWeightXsi(double w_xsi);
		void SetWeightTheta(double w_theta);
		void SetWeightLength(double w_length);
		void SetWeightVolume(double w_volume);
		void SetConeAngle(double cone_angle);
		void SetConeHeight(double cone_height);
		bool ProjectSingleFrame2PCL();
		bool ProjectFrameBatch2PCL();
		bool FindInitialState();
		np::ndarray ReturnSingleSegmentPCL();
		np::ndarray ReturnBBoxSingleFrame();
		np::ndarray ReturnBLRPCL();
		np::ndarray ReturnBBoxBLR();
		np::ndarray ReturnMBody();
		np::ndarray ReturnMWingL();
		np::ndarray ReturnMWingR();
		
		// MultiBodyICP functions
		np::ndarray PerformICPSingleFrame();
		np::ndarray ReturnProjectedModelPCL();
		np::ndarray ReturnProjectedFrame(int cam_nr);
		//np::ndarray ReturnProjectedModel();
		//np::ndarray ReturnModelImages(int cam_nr);

		// Auxiliary functions
		vector<double> py_list_to_std_vector( p::object& iterable );

};

#endif
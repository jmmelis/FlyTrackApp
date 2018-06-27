#include "flight_tracker_class.h"

#include <string>
#include <stdint.h>
#include <stdlib.h>
#include <chrono>
#include <thread>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <dirent.h>
#include <vector>
#include <boost/python.hpp>
#include <boost/python/module.hpp>
#include <boost/python/def.hpp>
#include <boost/python/extract.hpp>
#include <boost/python/numpy.hpp>

using namespace std;

namespace p = boost::python;
namespace np = boost::python::numpy;

// ------------------------------------------------------------------------------------------

FLT::FLT() {
}

// ------------------------------------------------------------------------------------------


// Set and get functions:

void FLT::SetSessionLoc(string SessionLoc) {
	session_loc = SessionLoc;
}

string FLT::GetSessionLoc() {
	return session_loc;
}

void FLT::SetMovName(string MovName) {
	mov_name = MovName;
}

string FLT::GetMovName() {
	return mov_name;
}

void FLT::SetCamName(string CamName) {
	cam_name = CamName;
}

string FLT::GetCamName() {
	return cam_name;
}

void FLT::SetCalLoc(string CalLoc) {
	cal_loc = CalLoc;
}

string FLT::GetCalLoc() {
	return cal_loc;
}

void FLT::SetCalName(string CalName) {
	cal_name = CalName;
}

string FLT::GetCalName() {
	return cal_name;
}

void FLT::SetNrOfCam(int CamNr) {
	N_cam = CamNr;
}

int FLT::GetNrOfCam() {
	return N_cam;
}

void FLT::SetMovNr(int MovNr) {
	mov_nr = MovNr;
}

int FLT::GetMovNr() {
	return mov_nr;
}

void FLT::SetBckgLoc(string BckgLoc) {
	bckg_loc = BckgLoc;
}

string FLT::GetBckgLoc() {
	return bckg_loc;
}

void FLT::SetBckgName(string BckgName) {
	bckg_name = BckgName;
}

string FLT::GetBckgName() {
	return bckg_name;
}

void FLT::SetBckgImageFormat(string BckgImageFormat) {
	bckg_img_format = BckgImageFormat;
}

string FLT::GetBckgImageFormat() {
	return bckg_img_format;
}

void FLT::SetFrameName(string FrameName) {
	frame_name = FrameName;
}

string FLT::GetFrameName() {
	return frame_name;
}

void FLT::SetFrameImageFormat(string FrameImageFormat) {
	frame_img_format = FrameImageFormat;
}

string FLT::GetFrameImageFormat() {
	return frame_img_format;
}

void FLT::SetStartPoint(int StartPoint) {
	start_point = StartPoint;
}

int FLT::GetStartPoint() {
	return start_point;
}

void FLT::SetMidPoint(int MidPoint) {
	mid_point = MidPoint;
}

int FLT::GetMidPoint() {
	return mid_point;
}

void FLT::SetEndPoint(int EndPoint) {
	end_point = EndPoint;
}

int FLT::GetEndPoint() {
	return end_point;
}

void FLT::SetTriggerMode(string TriggerMode) {
	trigger_mode = TriggerMode;
}

string FLT::GetTriggerMode() {
	return trigger_mode;
}

void FLT::SetNx(int new_Nx) {
	N_x = new_Nx;
}

int FLT::GetNx() {
	return N_x;
}

void FLT::SetNy(int new_Ny) {
	N_y = new_Ny;
}

int FLT::GetNy() {
	return N_y;
}

void FLT::SetNz(int new_Nz) {
	N_z = new_Nz;
}

int FLT::GetNz() {
	return N_z;
}

void FLT::SetDs(double new_ds) {
	ds = new_ds;
}

double FLT::GetDs() {
	return ds;
}

void FLT::SetX0(double new_x0) {
	x_0 = new_x0;
}

double FLT::GetX0() {
	return x_0;
}

void FLT::SetY0(double new_y0) {
	y_0 = new_y0;
}

double FLT::GetY0() {
	return y_0;
}

void FLT::SetZ0(double new_z0) {
	z_0 = new_z0;
}

double FLT::GetZ0() {
	return z_0;
}

void FLT::SetSolLoc(string SolLoc) {
	sol_loc = SolLoc;
}

string FLT::GetSolLoc() {
	return sol_loc;
}

void FLT::SetSolName(string SolName) {
	sol_name = SolName;
}

string FLT::GetSolName() {
	return sol_name;
}

void FLT::SetNumberOfThreads(int N) {
	N_threads = N;
}

int FLT::GetNumberOfThreads() {
	return N_threads;
}

void FLT::SetModelLoc(string ModelLoc) {
	model_loc = ModelLoc;
	session.model_loc = ModelLoc;
}

string FLT::GetModelLoc() {
	return model_loc;
}

void FLT::AddModel(string ModelFolder, string ModelName) {
	session.model_folders.push_back(ModelFolder);
	session.model_file_names.push_back(ModelName);
}

void FLT::ClearModelList() {
	session.model_folders.clear();
	session.model_file_names.clear();
}

// -----------------------------------------

// Set session_param

bool FLT::SetSessionParam() {

	bool success = true;

	try {

		session.N_cam = N_cam;
		session.mov_nr = mov_nr;
		session.start_point = start_point;
		session.mid_point = mid_point;
		session.end_point = end_point;
		session.trigger_mode = trigger_mode;
		session.session_loc = session_loc;
		session.mov_name = mov_name;
		session.cam_name = cam_name;
		session.cal_loc = cal_loc;
		session.cal_name = cal_name;
		session.bckg_loc = bckg_loc;
		session.bckg_name = bckg_name;
		session.bckg_img_format = bckg_img_format;
		session.frame_name = frame_name;
		session.frame_img_format = frame_img_format;
		session.sol_loc = sol_loc;
		session.sol_name = sol_name;

		vector<int> chrono_frame_nr;

		if (trigger_mode == "start") {
			for (int i = start_point; i<=end_point; i++) {
				chrono_frame_nr.push_back(i);
			}
		}
		else if (trigger_mode == "center") {
			for (int i = mid_point; i<=end_point; i++) {
				chrono_frame_nr.push_back(i);
			}
			for (int j = start_point; j<mid_point; j++) {
				chrono_frame_nr.push_back(j);
			}
		}
		else if (trigger_mode == "end") {
			for (int i = start_point; i<=end_point; i++) {
				chrono_frame_nr.push_back(i);
			}
		}

		session.chrono_frame_nr = chrono_frame_nr;
	}
	catch (...) {
		success = false;
	}

	return success;

}

// FrameLoader functions

bool FLT::InitFrameLoader() {

	bool success = true;

	if (fl.LoadBackground(session, frame_batch)==false) {
		success = false;
	}

	return success;
}

bool FLT::LoadSingleFrame(int frame_nr) {

	bool success = true;

	if (fl.LoadSingleFrame(session, frame_batch, frame_nr)==false) {
		success = false;
	}

	return success;
}

bool FLT::LoadFrameBatch(int batch_nr, int start_frame, int end_frame) {

	bool success = true;

	if (fl.LoadFrameBatch(session, frame_batch, start_frame, end_frame, batch_nr)==false) {
		success = false;
	}

	return success;
}

np::ndarray FLT::ReturnSingleFrame(int cam_nr) {

	return fl.ReturnSingleFrame(session, frame_batch, cam_nr);

}

np::ndarray FLT::ReturnImageSize() {

	p::tuple shape = p::make_tuple(session.N_cam,2);
	np::dtype dtype = np::dtype::get_builtin<int>();
	np::ndarray size_mat = np::zeros(shape,dtype);

	for (int i=0; i<session.N_cam; i++) {
		size_mat[i][0] = get<0>(frame_batch.image_size[i]);
		size_mat[i][1] = get<1>(frame_batch.image_size[i]);
	}

	return size_mat;
}

// FocalGrid functions

bool FLT::SetGridParameters() {

	bool success = true;

	try {
		voxel_grid.N_cam = N_cam;
		voxel_grid.N_threads = N_threads;
		voxel_grid.nx = N_x;
		voxel_grid.ny = N_y;
		voxel_grid.nz = N_z;
		voxel_grid.ds = ds;
		voxel_grid.x0 = x_0;
		voxel_grid.y0 = y_0;
		voxel_grid.z0 = z_0;

		if (fg.LoadCalibration(session, voxel_grid, frame_batch)==false) {
			success = false;
		}
	}
	catch (...) {
		success = false;
	}

	return success;
}

bool FLT::ConstructGrid() {

	bool success = true;

	try {
		if (fg.ConstructFocalGrid(voxel_grid)==false) {
			success = false;
		}
	}
	catch (...) {
		success = false;
	}

	return success;
}

np::ndarray FLT::XYZ2UV(double x_in, double y_in, double z_in) {

	arma::Col<double> xyz = {x_in, y_in, z_in, 1.0};

	arma::Mat<int> uv_mat = fg.TransformXYZ2UV(xyz);

	p::tuple shape = p::make_tuple(session.N_cam,2);
	np::dtype dtype = np::dtype::get_builtin<int>();
	np::ndarray point_mat = np::zeros(shape,dtype);

	for (int i=0; i<session.N_cam; i++) {
		point_mat[i][0] = uv_mat(0,i);
		point_mat[i][1] = uv_mat(1,i);
	}

	return point_mat;
}

np::ndarray FLT::DragPoint3D(int cam_nr, int u_now, int v_now, int u_prev, int v_prev, double x_prev, double y_prev, double z_prev) {

	tuple<arma::Mat<int>, arma::Col<double>> new_point;

	arma::Col<double> xyz_pos_prev = {x_prev,y_prev,z_prev,1.0};
	arma::Col<double> uv_pos_prev = {double(u_prev),double(v_prev),1.0};
	arma::Col<double> uv_pos_now = {double(u_now),double(v_now),1.0};

	new_point = fg.RayCasting(cam_nr, xyz_pos_prev, uv_pos_prev, uv_pos_now);

	p::tuple shape = p::make_tuple(3,session.N_cam+1);
	np::dtype dtype = np::dtype::get_builtin<double>();
	np::ndarray point_mat = np::zeros(shape,dtype);

	point_mat[0][0] = get<1>(new_point)(0);
	point_mat[1][0] = get<1>(new_point)(1);
	point_mat[2][0] = get<1>(new_point)(2);

	for (int i=0; i<session.N_cam; i++) {
		point_mat[0][i+1] = get<0>(new_point)(0,i);
		point_mat[1][i+1] = get<0>(new_point)(1,i);
		point_mat[2][i+1] = get<0>(new_point)(2,i);
	}

	return point_mat;
}

// Model functions

bool FLT::LoadModel(int model_ind) {

	bool success = true;

	try{
		if (mdl.LoadModel(session, insect_model, model_ind)==false) {
			success = false;
		}
	}
	catch (...) {
		success = false;
	}

	return success;
}

bool FLT::SetModelScale(p::object& scale_list) {

	bool success = true;

	try {

		vector<double> scale_vector = FLT::py_list_to_std_vector(scale_list);

		int N_seg = scale_vector.size();

		arma::Col<double> scale_vec(N_seg);

		for (int i=0; i<N_seg; i++) {
			scale_vec(i) = scale_vector[i];
		}

		mdl.SetScale(insect_model, scale_vec);

		mdl.SetStartState(insect_model);

	}
	catch (...) {
		success = false;
	}

	return success;
}

bool FLT::SetModelOrigin(double x_in, double y_in, double z_in) {

	bool success = true;

	try {

		arma::Col<double> origin_loc = {x_in, y_in, z_in};

		mdl.SetOrigin(insect_model, origin_loc);

	}
	catch (...) {
		success = false;
	}

	return success;

}

bool FLT::SetBodyLength(double body_length) {

	bool success = true;

	try {

		mdl.SetBodyLength(insect_model, body_length);

	}
	catch (...) {
		success = false;
	}

	return success;
}

bool FLT::SetWingLength(double wing_length) {

	bool success = true;

	try {

		mdl.SetWingLength(insect_model, wing_length);

	}
	catch (...) {
		success = false;
	}

	return success;
}

bool FLT::SetInitState() {

	bool success = true;

	try{
		mdl.SetInitState(insect_model, frame_batch);
	}
	catch(...) {
		success = false;
	}
	return success;
}

np::ndarray FLT::ReturnParentStructure() {

	int N_segs = insect_model.N_parts;

	p::tuple shape = p::make_tuple(N_segs,N_segs);
	np::dtype dtype = np::dtype::get_builtin<int>();
	np::ndarray parent_mat = np::zeros(shape,dtype);

	for (int i=0; i<N_segs; i++) {

		int N_parents = insect_model.parents[i].size();

		for (int j=0; j<N_parents; j++) {
			parent_mat[i][j] = insect_model.parents[i][j];
		}
	}

	return parent_mat;
}

/*
np::ndarray FLT::ReturnModelPointcloud() {

	vector<tuple<double,double,double,int>> model_pcl;

	model_pcl = mbicp.GetModelPointCloud(insect_model);

	int N = model_pcl.size();

	p::tuple shape = p::make_tuple(4,N);
	np::dtype dtype = np::dtype::get_builtin<double>();
	np::ndarray pcl_out = np::zeros(shape,dtype);

	for (int i=0; i<N; i++) {
		pcl_out[0][i] = get<0>(model_pcl[i]);
		pcl_out[1][i] = get<1>(model_pcl[i]);
		pcl_out[2][i] = get<2>(model_pcl[i]);
		pcl_out[3][i] = (double) get<3>(model_pcl[i]);
	}

	return pcl_out;
}
*/

np::ndarray FLT::ReturnModelState() {

	int N_segs = insect_model.N_parts;

	p::tuple shape = p::make_tuple(7,N_segs);
	np::dtype dtype = np::dtype::get_builtin<double>();
	np::ndarray state_out = np::zeros(shape,dtype);

	for (int i=0; i<N_segs; i++) {
		state_out[0][i] = insect_model.state[i](0);
		state_out[1][i] = insect_model.state[i](1);
		state_out[2][i] = insect_model.state[i](2);
		state_out[3][i] = insect_model.state[i](3);
		state_out[4][i] = insect_model.state[i](4);
		state_out[5][i] = insect_model.state[i](5);
		state_out[6][i] = insect_model.state[i](6);
	}

	return state_out;
}

np::ndarray FLT::ReturnModelScale() {

	int N_segs = insect_model.N_parts;

	p::tuple shape = p::make_tuple(N_segs,1);
	np::dtype dtype = np::dtype::get_builtin<double>();
	np::ndarray scale_out = np::zeros(shape,dtype);

	for (int i=0; i<N_segs; i++) {
		scale_out[i] = insect_model.scale[i];
		cout << insect_model.scale[i] << endl;
	}

	return scale_out;

}

p::list FLT::ReturnSTLFileList() {

	p::list name_list;

	vector<string> stl_file_list = mdl.ReturnSTLNames(insect_model);

	for (int p=0; p<stl_file_list.size(); p++) {
		name_list.append(stl_file_list[p]);
	}

	return name_list;
}

np::ndarray FLT::ReturnTransformationMatrices() {

	vector<arma::Mat<double>> M_vec = mdl.ReturnTransformMatrices(insect_model);

	int N_segs = insect_model.N_parts;

	p::tuple shape = p::make_tuple(16,N_segs);
	np::dtype dtype = np::dtype::get_builtin<double>();
	np::ndarray mat_out = np::zeros(shape,dtype);

	for (int i=0; i<N_segs; i++) {
		mat_out[0][i] = M_vec[i](0,0);
		mat_out[1][i] = M_vec[i](0,1);
		mat_out[2][i] = M_vec[i](0,2);
		mat_out[3][i] = M_vec[i](0,3);
		mat_out[4][i] = M_vec[i](1,0);
		mat_out[5][i] = M_vec[i](1,1);
		mat_out[6][i] = M_vec[i](1,2);
		mat_out[7][i] = M_vec[i](1,3);
		mat_out[8][i] = M_vec[i](2,0);
		mat_out[9][i] = M_vec[i](2,1);
		mat_out[10][i] = M_vec[i](2,2);
		mat_out[11][i] = M_vec[i](2,3);
		mat_out[12][i] = M_vec[i](3,0);
		mat_out[13][i] = M_vec[i](3,1);
		mat_out[14][i] = M_vec[i](3,2);
		mat_out[15][i] = M_vec[i](3,3);
	}

	return mat_out;
}

// ImagSegm functions

void FLT::SetImagSegmParam(int body_thresh, int wing_thresh, double Sigma, int K, int min_body_size, int min_wing_size, bool Tethered) {

	arma::Col<double> xyz_pos = {insect_model.origin_loc(0), insect_model.origin_loc(1), insect_model.origin_loc(2), 1.0};

	arma::Mat<int> uv_mat = fg.TransformXYZ2UV(xyz_pos);

	vector<tuple<double,double>> Origin;

	for (int n=0; n<session.N_cam; n++) {
		Origin.push_back(make_tuple(uv_mat(0,n),uv_mat(1,n)));
	}

	double BodyLength = insect_model.body_length/voxel_grid.ds;
	double WingLength = insect_model.wing_length/voxel_grid.ds;

	seg.SetBodySegmentationParam(body_thresh, 2, 5, Sigma, K, min_body_size, BodyLength, Origin, Tethered);
	seg.SetWingSegmentationParam(wing_thresh, 2, Sigma, K, min_wing_size, WingLength);

}

bool FLT::SegmentSingleFrame() {

	bool success = true;

	try {

		seg.SegmentSingleFrame(frame_batch);

	}
	catch (...) {
		success = false;
	}

	return success;

}

bool FLT::SegmentFrameBatch() {

	bool success = true;

	try {

		seg.SegmentFrameBatch(frame_batch);

	}
	catch (...) {
		success = false;
	}

	return success;
}

np::ndarray FLT::ReturnSegmentedFrame(int cam_nr) {

	return seg.ReturnSegFrame(session, frame_batch, cam_nr);

}

// InitState functions

void FLT::SetWeightXsi(double w_xsi) {
	init.SetWeightXsi(w_xsi);
}

void FLT::SetWeightTheta(double w_theta) {
	init.SetWeightTheta(w_theta);
}

void FLT::SetWeightLength(double w_length) {
	init.SetWeightLength(w_length);
}

void FLT::SetWeightVolume(double w_volume) {
	init.SetWeightVolume(w_volume);
}

void FLT::SetConeAngle(double cone_angle) {
	init.SetConeAngle(cone_angle);
}

void FLT::SetConeHeight(double cone_height) {
	init.SetConeHeight(cone_height);
}

bool FLT::ProjectSingleFrame2PCL() {

	bool success = true;

	try {

		init.ProjectSingleFrame(fg, frame_batch, voxel_grid);

	}
	catch (...) {
		success = false;
	}

	return success;
}

bool FLT::ProjectFrameBatch2PCL() {

	bool success = true;

	try {

		init.ProjectFrameBatch(fg, frame_batch, voxel_grid);

	}
	catch (...) {
		success = false;
	}

	return success;

}

bool FLT::FindInitialState() {

	bool success = true;

	try {

		init.FindInitialStateSingleFrame(frame_batch,insect_model);

	}
	catch (...) {
		success = false;
	}

	return success;
}

np::ndarray FLT::ReturnSingleSegmentPCL() {

	return init.ReturnSinglePCL(frame_batch);

}

np::ndarray FLT::ReturnBBoxSingleFrame() {

	return init.ReturnBBoxSinglePCL(frame_batch);

}

np::ndarray FLT::ReturnBLRPCL() {

	return init.ReturnBLRPCL(frame_batch);

}

np::ndarray FLT::ReturnBBoxBLR() {
	return init.ReturnBBoxBLR(frame_batch);
}

np::ndarray FLT::ReturnMBody() {
	return init.ReturnMBody(frame_batch);
}

np::ndarray FLT::ReturnMWingL() {
	return init.ReturnMWingL(frame_batch);
}

np::ndarray FLT::ReturnMWingR() {
	return init.ReturnMWingR(frame_batch);
}

// MultiBodyICP functions

np::ndarray FLT::PerformICPSingleFrame() {

	//vector<arma::Mat<double>> M_align = mbicp.SingleFrameICP(frame_batch, insect_model);

	//int vec_size = M_align.size();

	int vec_size = 10;

	p::tuple shape = p::make_tuple(16,vec_size);
	np::dtype dtype = np::dtype::get_builtin<double>();
	np::ndarray M_out = np::zeros(shape,dtype);

	/*
	for (int i=0; i<vec_size; i++) {
		for (int j=0; j<4; j++) {
			for (int k=0; k<4; k++) {
				M_out[j*4+k][i] = M_align[i](j,k);
			}
		}
	}
	*/

	return M_out;
}

np::ndarray FLT::ReturnProjectedModelPCL() {

	//arma::Mat<double> segment_pcls = mbicp.ReturnSRCPCL(frame_batch, insect_model, voxel_grid, fg);
	arma::Mat<double> segment_pcls = mbicp.SingleFrameICP(frame_batch, insect_model, voxel_grid, fg);

	int N_points = segment_pcls.n_cols;

	p::tuple shape = p::make_tuple(7,N_points);
	np::dtype dtype = np::dtype::get_builtin<double>();
	np::ndarray array_out = np::zeros(shape,dtype);

	for (int i=0; i<N_points; i++) {

		array_out[0][i] = segment_pcls(0,i);
		array_out[1][i] = segment_pcls(1,i);
		array_out[2][i] = segment_pcls(2,i);
		array_out[3][i] = segment_pcls(3,i);
		array_out[4][i] = segment_pcls(4,i);
		array_out[5][i] = segment_pcls(5,i);
		array_out[6][i] = segment_pcls(6,i);

	}

	return array_out;
}

np::ndarray FLT::ReturnProjectedFrame(int cam_nr) {

	vector<arma::Mat<int>> proj_frames = mbicp.ReturnProjectedImages(frame_batch, insect_model, voxel_grid, fg);

	int N_rows = proj_frames[cam_nr].n_rows;
	int N_cols = proj_frames[cam_nr].n_cols;

	p::tuple shape = p::make_tuple(N_rows,N_cols);
	np::dtype dtype = np::dtype::get_builtin<double>();
	np::ndarray frame_out = np::zeros(shape,dtype);

	for (int i=0; i<N_rows; i++) {
		for (int j=0; j<N_cols; j++) {
			frame_out[i][j] = proj_frames[cam_nr](i,j);
		}
	}

	return frame_out;
}

/*
np::ndarray FLT::ReturnProjectedModel() {

	vector<arma::Col<int>> frames_in = mbicp.ProjectModel2Image(insect_model, voxel_grid, fg);

	vector<tuple<double,double,double,int>> model_pcl = mbicp.ProjectImage2PointCloud(frames_in, voxel_grid, fg);

	int N = model_pcl.size();

	p::tuple shape = p::make_tuple(4,N);
	np::dtype dtype = np::dtype::get_builtin<double>();
	np::ndarray pcl_out = np::zeros(shape,dtype);

	for (int i=0; i<N; i++) {
		pcl_out[0][i] = get<0>(model_pcl[i]);
		pcl_out[1][i] = get<1>(model_pcl[i]);
		pcl_out[2][i] = get<2>(model_pcl[i]);
		pcl_out[3][i] = (double) get<3>(model_pcl[i]);
	}

	return pcl_out;
}

np::ndarray FLT::ReturnModelImages(int cam_nr) {

	vector<arma::Col<int>> frames_out = mbicp.ProjectModel2Image(insect_model, voxel_grid, fg);

	int N_rows = get<0>(frame_batch.image_size[cam_nr]);
	int N_cols = get<1>(frame_batch.image_size[cam_nr]);

	p::tuple shape = p::make_tuple(N_rows,N_cols);
	np::dtype dtype = np::dtype::get_builtin<int>();
	np::ndarray img_out = np::zeros(shape,dtype);

	for (int i=0; i<N_cols; i++) {
		for (int j=0; j<N_rows; j++) {
				img_out[j][i] = frames_out[cam_nr](i*N_rows+j);
		}
	}

	return img_out;
}
*/

// Auxiliary functions

vector<double> FLT::py_list_to_std_vector( p::object& iterable ) {
    return vector<double>( p::stl_input_iterator< double >(iterable), p::stl_input_iterator< double >( ));
}
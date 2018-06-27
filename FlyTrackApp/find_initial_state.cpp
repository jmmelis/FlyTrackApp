#include "find_initial_state.h"

#include "session_param.h"
#include "frames.h"
#include "vox_grid.h"

#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include <string>
#include <stdint.h>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <thread>
#include <chrono>
#include <algorithm>
#include <sys/types.h>
#include <sys/stat.h>
#include <dirent.h>
#include <armadillo>
#include <pcl/features/moment_of_inertia_estimation.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/vtk_lib_io.h>
#include <pcl/point_types.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/features/normal_3d.h>
#include <pcl/surface/gp3.h>
#include <pcl/visualization/cloud_viewer.h>
#include <boost/thread/thread.hpp>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/console/parse.h>
#include <pcl/common/transforms.h>
#include <pcl/PCLPointCloud2.h>

#include "focal_grid.h"

#define PI 3.14159265

using namespace std;

namespace p = boost::python;
namespace np = boost::python::numpy;

//-----------------------------------------------------------------------
//
// Find the initial state of the fly using minimum bounding boxes
//
//-----------------------------------------------------------------------

InitState::InitState() {
	// empty
}

void InitState::SetWeightXsi(double WXsi) {
	w_xsi = WXsi;
}

void InitState::SetWeightTheta(double WTheta) {
	w_theta = WTheta;
}

void InitState::SetWeightLength(double WLength) {
	w_length = WLength;
}

void InitState::SetWeightVolume(double WVolume) {
	w_volume = WVolume;
}

void InitState::SetConeAngle(double ConeAngle) {
	cone_angle = (ConeAngle/180.0)*PI;
}

void InitState::SetConeHeight(double ConeHeight) {
	cone_height = ConeHeight;
}

void InitState::ProjectSingleFrame(FocalGrid &fg, frames &frame_in, vox_grid &vox) {

	// Project a single segmented frame to a set of pointclouds:

	frame_in.single_frame_pcl.clear();

	vector<tuple<int,double,double,double,double,double,double>> pcl_out = fg.ProjectImage2Cloud(frame_in.single_seg_frame, vox);

	frame_in.single_frame_pcl = pcl_out;

}

void InitState::ProjectFrameBatch(FocalGrid &fg, frames &frame_in, vox_grid &vox) {

	// Project a segmented frame batch to pointclouds:

	frame_in.frame_batch_pcl.clear();

	int N_frames = frame_in.seg_frames[0].n_cols;

	int N_cam = frame_in.seg_frames.size();

	for (int i=0; i<N_frames; i++) {

		vector<arma::Col<int>> frame_now;

		for (int j=0; j<N_cam; j++) {
			frame_now.push_back(frame_in.seg_frames[j].col(i));
		}

		vector<tuple<int,double,double,double,double,double,double>> pcl_out = fg.ProjectImage2Cloud(frame_now, vox);

		frame_in.frame_batch_pcl.push_back(pcl_out);

	}

}

void InitState::FindInitialStateSingleFrame(frames &frame_in, model &mod) {

	frame_in.pcl_init_single_frame.clear();
	frame_in.M_init_single_frame.clear();

	double wing_length = mod.wing_length;

	vector<tuple<int,double,double,double,double,double,double>> pcl_in = frame_in.single_frame_pcl;

	arma::Mat<double> pcl_mat = InitState::ConvertVectorPCLArma(pcl_in);

	vector<tuple<struct bbox, arma::Mat<double>>> bbox_vector = InitState::FindBoundingBoxes(pcl_mat);

	vector<tuple<arma::Mat<double>,arma::Mat<double>>> pcl_vector = InitState::FindBodyandWing(bbox_vector,wing_length);

	frame_in.pcl_init_single_frame.push_back(get<0>(pcl_vector[0]));
	frame_in.pcl_init_single_frame.push_back(get<0>(pcl_vector[1]));
	frame_in.pcl_init_single_frame.push_back(get<0>(pcl_vector[2]));
	frame_in.M_init_single_frame.push_back(get<1>(pcl_vector[0]));
	frame_in.M_init_single_frame.push_back(get<1>(pcl_vector[1]));
	frame_in.M_init_single_frame.push_back(get<1>(pcl_vector[2]));

}

void InitState::FindInitialStateFrameBatch(frames &frame_in, model &mod) {

	frame_in.pcl_init_frame_batch.clear();
	frame_in.M_init_frame_batch.clear();

	double wing_length = mod.wing_length;

	int N_batch = frame_in.raw_frames[0].n_cols;

	frame_in.pcl_init_frame_batch.resize(N_batch);
	frame_in.M_init_frame_batch.resize(N_batch);

	for (int i=0; i<N_batch; i++) {

		vector<tuple<int,double,double,double,double,double,double>> pcl_in = frame_in.frame_batch_pcl[i];

		arma::Mat<double> pcl_mat = InitState::ConvertVectorPCLArma(pcl_in);

		vector<tuple<struct bbox, arma::Mat<double>>> bbox_vector = InitState::FindBoundingBoxes(pcl_mat);

		vector<tuple<arma::Mat<double>,arma::Mat<double>>> pcl_vector = InitState::FindBodyandWing(bbox_vector,wing_length);

		frame_in.pcl_init_frame_batch[i].push_back(get<0>(pcl_vector[0]));
		frame_in.pcl_init_frame_batch[i].push_back(get<0>(pcl_vector[1]));
		frame_in.pcl_init_frame_batch[i].push_back(get<0>(pcl_vector[2]));
		frame_in.M_init_frame_batch[i].push_back(get<1>(pcl_vector[0]));
		frame_in.M_init_frame_batch[i].push_back(get<1>(pcl_vector[1]));
		frame_in.M_init_frame_batch[i].push_back(get<1>(pcl_vector[2]));

	}

}

vector<tuple<arma::Mat<double>,arma::Mat<double>>> InitState::FindBodyandWing(vector<tuple<struct bbox, arma::Mat<double>>> &bbox_vector, double wing_length) {

	int N_segments = bbox_vector.size();

	arma::Mat<double> M_body;
	arma::Mat<double> M_wing_L;
	arma::Mat<double> M_wing_R;

	arma::Mat<double> body_pcl;
	arma::Mat<double> wing_L_pcl;
	arma::Mat<double> wing_R_pcl;

	if (N_segments>0) {

		struct bbox body_box = get<0>(bbox_vector[0]);
		body_pcl = get<1>(bbox_vector[0]);

		M_body = InitState::FindBodyRefFrame(body_box, body_pcl);

		if (N_segments>1) {

			vector<arma::Mat<double>> c_vector;
			vector<arma::Mat<double>> wing_pcl;

			for (int i=1; i<N_segments; i++) {
				arma::Mat<double> c_points;
				c_points = InitState::FindRootTip(M_body, get<0>(bbox_vector[i]));
				c_vector.push_back(c_points);
				wing_pcl.push_back(get<1>(bbox_vector[i]));
			}

			vector<arma::Mat<double>> wing_pcl_LR = InitState::FindWingPCL(wing_pcl, c_vector, wing_length);
			wing_L_pcl = wing_pcl_LR[0];
			wing_R_pcl = wing_pcl_LR[1];			
		}
		else {
			wing_L_pcl.zeros(7,1);
			wing_R_pcl.zeros(7,1);
		}
	}
	else {
		body_pcl.zeros(7,1);
		wing_L_pcl.zeros(7,1);
		wing_R_pcl.zeros(7,1);
	}

	body_pcl.row(0).fill(1.0);
	wing_L_pcl.row(0).fill(2.0);
	wing_R_pcl.row(0).fill(3.0);

	vector<arma::Mat<double>> M_vector = InitState::FindWingRefFrame(M_body, wing_L_pcl, wing_R_pcl);

	//arma::Mat<double> wing_select_L = InitState::FindWingOrientation(M_vector[0], wing_L_pcl, 0);
	//arma::Mat<double> wing_select_R = InitState::FindWingOrientation(M_vector[1], wing_R_pcl, 1);

	vector<tuple<arma::Mat<double>,arma::Mat<double>>> vector_out;

	vector_out.push_back(make_tuple(body_pcl,M_body));
	vector_out.push_back(make_tuple(wing_L_pcl,M_vector[0]));
	vector_out.push_back(make_tuple(wing_R_pcl,M_vector[1]));

	//vector_out.push_back(make_tuple(body_pcl,M_body));
	//vector_out.push_back(make_tuple(wing_L_pcl,wing_select_L));
	//vector_out.push_back(make_tuple(wing_R_pcl,wing_select_R));

	return vector_out;
}

vector<arma::Mat<double>> InitState::FindWingPCL(vector<arma::Mat<double>> wing_pcls, vector<arma::Mat<double>> wing_prop, double wing_length) {

	int N_c = wing_prop.size(); // it is expected that there is 1 candidate or more

	vector<arma::Mat<double>> wing_pcl_LR;

	// Try to find a left and right wing candidate first:

	double score = 0.0;
	tuple<int, int> LR_pair;
	LR_pair = make_tuple(-1,-1);

	for (int i=0; i<N_c; i++) {
		for (int j=0; j<i; j++) {
			if ((wing_prop[i](3,9)>(0.8*wing_length)) && (wing_prop[i](3,9)<(1.5*wing_length))) {
				if ((wing_prop[j](3,9)>(0.8*wing_length)) && (wing_prop[j](3,9)<(1.5*wing_length))) {
					arma::Col<double> tip_i = {wing_prop[i](0,9),wing_prop[i](1,9),wing_prop[i](2,9)};
					arma::Col<double> tip_j = {wing_prop[j](0,9),wing_prop[j](1,9),wing_prop[j](2,9)};
					double d_xsi = arma::dot(tip_i,tip_j)/(arma::norm(tip_i,2.0)*arma::norm(tip_j,2.0));
					double delta_xsi = 0.0;
					if (d_xsi >= -1.0 && d_xsi <= 1.0) {
						delta_xsi = acos(d_xsi);
					}
					double xsi_i = wing_prop[i](0,10);
					double xsi_j = wing_prop[j](0,10);
					double theta_avg = sqrt(pow((wing_prop[i](1,10)+wing_prop[j](1,10))/2.0,2));
					double wing_span_avg = (wing_prop[i](2,10)+wing_prop[j](2,10))/2.0;
					double volume_avg = (wing_prop[i](3,10)+wing_prop[j](3,10));
					double score_now = w_xsi*(delta_xsi/PI)+w_length*(wing_span_avg/wing_length)+w_volume*volume_avg-w_theta*((2.0*theta_avg)/PI);
					if (score_now>score) {
						score = score_now;
						if (xsi_i > xsi_j) {
							LR_pair = make_tuple(i,j);
						}
						else if (xsi_i<xsi_j) {
							LR_pair = make_tuple(j,i);
						}
						else {
							LR_pair = make_tuple(i,j);
						}
					}
				}
			}
		}
	}

	cout << "left right pair" << endl;
	cout << get<0>(LR_pair) << endl;
	cout << get<1>(LR_pair) << endl;
	cout << "" << endl;

	// If a pair has been found, form a pointcloud of all pointclouds in the vicinity of the wingtips:

	int ind_L = get<0>(LR_pair);
	int ind_R = get<1>(LR_pair);

	arma::Mat<double> wing_L_pcl;
	arma::Mat<double> wing_R_pcl;

	if (ind_L != -1) {
		wing_L_pcl = wing_pcls[ind_L];
		wing_R_pcl = wing_pcls[ind_R];

		arma::Col<double> tip_L = {wing_prop[ind_L](0,9),wing_prop[ind_L](1,9),wing_prop[ind_L](2,9)};
		arma::Col<double> tip_R = {wing_prop[ind_R](0,9),wing_prop[ind_R](1,9),wing_prop[ind_R](2,9)};

		for (int i=0; i<N_c; i++) {
			arma::Col<double> tip_i = {wing_prop[i](0,9),wing_prop[i](1,9),wing_prop[i](2,9)};
			//arma::Col<double> root_i = {wing_prop[i](0,8),wing_prop[i](1,8),wing_prop[i](2,8)};
			arma::Col<double> root_i = {wing_prop[i](0,11),wing_prop[i](1,11),wing_prop[i](2,11)};
			if (i != ind_L) {
				double delta_angle_tip = arma::dot(tip_i,tip_L)/(arma::norm(tip_i,2.0)*arma::norm(tip_L,2.0));
				double tip_angle_L = -PI;
				if (delta_angle_tip >= -1.0 && delta_angle_tip <= 1.0) {
					tip_angle_L = acos(delta_angle_tip);
				}
				double delta_angle_root = arma::dot(root_i,tip_L)/(arma::norm(root_i,2.0)*arma::norm(tip_L,2.0));
				double root_angle_L = -PI;
				if (delta_angle_root >= -1.0 && delta_angle_root <= 1.0) {
					root_angle_L = acos(delta_angle_root);
				}
				double center_dist = wing_prop[i](3,11);
				if (tip_angle_L >= 0.0 && tip_angle_L <= cone_angle) {
					if (root_angle_L >= 0.0 && root_angle_L <= cone_angle) {
						if (center_dist > ((1.0-cone_height)*wing_length)) {
							wing_L_pcl = arma::join_rows(wing_L_pcl,wing_pcls[i]);
						}
					}
				}
			}
			if (i != ind_R) {
				double delta_angle_tip = arma::dot(tip_i,tip_R)/(arma::norm(tip_i,2.0)*arma::norm(tip_R,2.0));
				double tip_angle_R = -PI;
				if (delta_angle_tip >= -1.0 && delta_angle_tip <= 1.0) {
					tip_angle_R = acos(delta_angle_tip);
				}
				double delta_angle_root = arma::dot(root_i,tip_R)/(arma::norm(root_i,2.0)*arma::norm(tip_R,2.0));
				double root_angle_R = -PI;
				if (delta_angle_root >= -1.0 && delta_angle_root <= 1.0) {
					root_angle_R = acos(delta_angle_root);
				}
				double center_dist = wing_prop[i](3,11);
				if (tip_angle_R >= 0.0 && tip_angle_R <= cone_angle) {
					if (root_angle_R >= 0.0 && root_angle_R <= cone_angle) {
						if (center_dist > ((1.0-cone_height)*wing_length)) {
							wing_R_pcl = arma::join_rows(wing_R_pcl,wing_pcls[i]);
						}
					}
				}
			}
		}
	}
	else {
		wing_L_pcl.zeros(7,1);
		wing_R_pcl.zeros(7,1);
	}

	wing_pcl_LR.push_back(wing_L_pcl);
	wing_pcl_LR.push_back(wing_R_pcl);

	return wing_pcl_LR;
}

/*
arma::Mat<double> InitState::FindWingOrientation(arma::Mat<double> M_vector, arma::Mat<double> &wing_pcl_in, int L_or_R) {

	int N_segs = 5;

	arma::Mat<double> M_out;
	M_out.zeros(4,8);

	// Get x,y,z coordinates from pcl:

	double theta_avg = 0.0;

	if (wing_pcl_in.n_cols>10 && arma::sum(M_vector.row(3)) != 0) {

		arma::Mat<double> wing_pcl;

		wing_pcl.ones(4,wing_pcl_in.n_cols);

		wing_pcl.row(0) = wing_pcl_in.row(1);
		wing_pcl.row(1) = wing_pcl_in.row(2);
		wing_pcl.row(2) = wing_pcl_in.row(3);

		// Project pcl into wing reference frame:

		arma::Mat<double> M_wing;
		M_wing.eye(4,4);

		// Transpose of the matrix

		M_wing(0,0) = M_vector(0,0);
		M_wing(0,1) = M_vector(1,0);
		M_wing(0,2) = M_vector(2,0);
		M_wing(0,3) = -M_vector(0,3);
		M_wing(1,0) = M_vector(0,1);
		M_wing(1,1) = M_vector(1,1);
		M_wing(1,2) = M_vector(2,1);
		M_wing(1,3) = -M_vector(1,3);
		M_wing(2,0) = M_vector(0,2);
		M_wing(2,1) = M_vector(1,2);
		M_wing(2,2) = M_vector(2,2);
		M_wing(2,3) = -M_vector(2,3);

		arma::Mat<double> wing_pcl_ref = M_wing*wing_pcl;

		// Perform svd on wing_pcl_ref:

		arma::Row<double> angle_y;
		angle_y.zeros(N_segs);

		arma::Row<double> chord_length;
		chord_length.zeros(N_segs);

		arma::Row<int> N_seg_points;
		N_seg_points.zeros(N_segs);

		if (L_or_R==0) {

			double y_tip = wing_pcl_ref.row(1).max();

			// Split the wing in N segments:
			for (int i=0; i<N_segs; i++) {

				// Select y-slice elements
				arma::uvec slice_y_ids = arma::find((wing_pcl_ref.row(1) > (i*(y_tip/((N_segs+1)*1.0)))) && (wing_pcl_ref.row(1) < ((i+1)*(y_tip/((N_segs+1)*1.0)))));

				if (slice_y_ids.n_rows > 2) {

					arma::Mat<double> slice_points_y = wing_pcl_ref.cols(slice_y_ids);

					// Find the maximum radius from the mid point of the section

					arma::Row<double> R1;
					R1.zeros(slice_y_ids.n_rows);

					for (int j=0; j<slice_y_ids.n_rows; j++) {
						R1(j) = sqrt(pow(slice_points_y(0,j),2.0)+pow(slice_points_y(2,j),2.0));
					}

					int max_R1 = R1.index_max();

					arma::Col<double> R1_point = slice_points_y.col(max_R1);

					// Find the maximum radius from R1:

					arma::Row<double> R2;
					R2.zeros(slice_y_ids.n_rows);

					for (int j=0; j<slice_y_ids.n_rows; j++) {
						R2(j) = sqrt(pow(slice_points_y(0,j)-R1_point(0),2.0)+pow(slice_points_y(2,j)-R1_point(2),2.0));
					}

					int max_R2 = R2.index_max();

					arma::Col<double> R2_point = slice_points_y.col(max_R2);

					// Calculate the angle between the x-axis and the vector R1_point-R2_point:

					double theta = atan2(R1_point(2)-R2_point(2),R1_point(0)-R2_point(0));

					if (isfinite(theta)) {
						angle_y(i) = theta;
						chord_length(i) = sqrt(pow(R1_point(0)-R2_point(0),2.0)+pow(R1_point(2)-R2_point(2),2.0));
					}
				}
				/*
				if (slice_y_ids.n_rows > 10) {

					// Perform eigenvalue decomposition:

					arma::Mat<double> slice_points_y = wing_pcl_ref.cols(slice_y_ids);

					arma::Mat<double> X(3,slice_y_ids.n_rows);

					X.row(0) = slice_points_y.row(0);
					X.row(1) = slice_points_y.row(1);
					X.row(2) = slice_points_y.row(2);

					arma::mat U;
					arma::vec s;
					arma::mat V;

					arma::svd(U,s,V,X);

					arma::Col<double> eig_vec_1 = U.col(0);

					double theta = atan2(eig_vec_1(2),eig_vec_1(0));

					if (isfinite(theta)) {
						angle_y(i) = theta;
						N_seg_points(i) = slice_y_ids.n_rows;
						//chord_length(i) = sqrt(pow(R1_point(0)-R2_point(0),2.0)+pow(R1_point(2)-R2_point(2),2.0));
					}
				}
			}
		}
		else {

			double y_tip = wing_pcl_ref.row(1).min();

			// Split the wing in N segments:
			for (int i=0; i<N_segs; i++) {

				// Select y-slice elements
				arma::uvec slice_y_ids = arma::find((wing_pcl_ref.row(1) < (i*(y_tip/((N_segs+1)*1.0)))) && (wing_pcl_ref.row(1) > ((i+1)*(y_tip/((N_segs+1)*1.0)))));


				
				if (slice_y_ids.n_rows > 2) {

					arma::Mat<double> slice_points_y = wing_pcl_ref.cols(slice_y_ids);

					// Find the maximum radius from the mid point of the section

					arma::Row<double> R1;
					R1.zeros(slice_y_ids.n_rows);

					for (int j=0; j<slice_y_ids.n_rows; j++) {
						R1(j) = sqrt(pow(slice_points_y(0,j),2.0)+pow(slice_points_y(2,j),2.0));
					}

					int max_R1 = R1.index_max();

					arma::Col<double> R1_point = slice_points_y.col(max_R1);

					// Find the maximum radius from R1:

					arma::Row<double> R2;
					R2.zeros(slice_y_ids.n_rows);

					for (int j=0; j<slice_y_ids.n_rows; j++) {
						R2(j) = sqrt(pow(slice_points_y(0,j)-R1_point(0),2.0)+pow(slice_points_y(2,j)-R1_point(2),2.0));
					}

					int max_R2 = R2.index_max();

					arma::Col<double> R2_point = slice_points_y.col(max_R2);

					// Calculate the angle between the x-axis and the vector R1_point-R2_point:

					double theta = atan2(R1_point(2)-R2_point(2),R1_point(0)-R2_point(0));

					if (isfinite(theta)) {
						angle_y(i) = theta;
						chord_length(i) = sqrt(pow(R1_point(0)-R2_point(0),2.0)+pow(R1_point(2)-R2_point(2),2.0));
					}
				}
				/*
				if (slice_y_ids.n_rows > 10) {

					// Perform eigenvalue decomposition:

					arma::Mat<double> slice_points_y = wing_pcl_ref.cols(slice_y_ids);

					arma::Mat<double> X(3,slice_y_ids.n_rows);

					X.row(0) = slice_points_y.row(0);
					X.row(1) = slice_points_y.row(1);
					X.row(2) = slice_points_y.row(2);

					arma::mat U;
					arma::vec s;
					arma::mat V;

					arma::svd(U,s,V,X);

					arma::Col<double> eig_vec_1 = U.col(0);

					double theta = atan2(eig_vec_1(2),eig_vec_1(0));

					if (isfinite(theta)) {
						angle_y(i) = theta;
						N_seg_points(i) = slice_y_ids.n_rows;
						//chord_length(i) = sqrt(pow(R1_point(0)-R2_point(0),2.0)+pow(R1_point(2)-R2_point(2),2.0));
					}
				}

			}
		}

		cout << angle_y << endl;
		cout << N_seg_points << endl;

		// Determine the average orientation angle:
		double cos_2_theta_avg = 0.0;
		double chord_length_sum = 0.0;
		for (int i=0; i<N_segs; i++) {
			if (chord_length(i) > 0.0) {
				cos_2_theta_avg += chord_length(i)*0.5*acos(cos(2.0*angle_y(i)));
				chord_length_sum += chord_length(i);
			}
		}

		cout << cos_2_theta_avg << endl;

		double theta_avg = 0.0;

		if (chord_length_sum>0.0) {
			theta_avg = cos_2_theta_avg/chord_length_sum;
		}

		/*
		double cos_2_theta_avg = 0.0;
		int N_seg_points_sum = 0;

		for (int i=0; i<N_segs; i++) {
			if (N_seg_points(i) > 10) {
				cos_2_theta_avg += N_seg_points(i)*0.5*acos(cos(2.0*angle_y(i)));
				N_seg_points_sum += N_seg_points(i);
			}
		}

		cout << cos_2_theta_avg << endl;

		double theta_avg = 0.0;

		if (N_seg_points_sum>10) {
			theta_avg = cos_2_theta_avg/(1.0*N_seg_points_sum);
		}

		cout << theta_avg << endl;

		// Rotate the pointcloud by theta_avg around the y-axis:
		arma::Mat<double> M_theta1;
		M_theta1.eye(4,4);
		M_theta1(0,0) = cos(theta_avg);
		M_theta1(0,2) = -sin(theta_avg);
		M_theta1(2,0) = sin(theta_avg);
		M_theta1(2,2) = cos(theta_avg);

		cout << "M theta 1" << endl;
		cout << M_theta1 << endl;

		arma::Mat<double> M_theta2;
		M_theta2.eye(4,4);
		M_theta2(0,0) = cos(theta_avg+PI);
		M_theta2(0,2) = -sin(theta_avg+PI);
		M_theta2(2,0) = sin(theta_avg+PI);
		M_theta2(2,2) = cos(theta_avg+PI);

		cout << "M theta 2" << endl;
		cout << M_theta2 << endl;

		// Return M_out, wing_grid and orient_ind:

		M_out(0,0) = M_vector(0,0);
		M_out(0,1) = M_vector(0,1);
		M_out(0,2) = M_vector(0,2);
		M_out(0,3) = M_vector(0,3);
		M_out(1,0) = M_vector(1,0);
		M_out(1,1) = M_vector(1,1);
		M_out(1,2) = M_vector(1,2);
		M_out(1,3) = M_vector(1,3);
		M_out(2,0) = M_vector(2,0);
		M_out(2,1) = M_vector(2,1);
		M_out(2,2) = M_vector(2,2);
		M_out(2,3) = M_vector(2,3);
		M_out(3,3) = 1.0;

		M_out.submat(0,0,3,3) = M_out.submat(0,0,3,3)*M_theta1;

		M_out(0,4) = M_vector(0,0);
		M_out(0,5) = M_vector(0,1);
		M_out(0,6) = M_vector(0,2);
		M_out(0,7) = M_vector(0,3);
		M_out(1,4) = M_vector(1,0);
		M_out(1,5) = M_vector(1,1);
		M_out(1,6) = M_vector(1,2);
		M_out(1,7) = M_vector(1,3);
		M_out(2,4) = M_vector(2,0);
		M_out(2,5) = M_vector(2,1);
		M_out(2,6) = M_vector(2,2);
		M_out(2,7) = M_vector(2,3);
		M_out(3,7) = 1.0;

		M_out.submat(0,4,3,7) = M_out.submat(0,4,3,7)*M_theta2;

		cout << M_out << endl;

	}
	else {

		M_out(0,0) = M_vector(0,0);
		M_out(0,1) = M_vector(0,1);
		M_out(0,2) = M_vector(0,2);
		M_out(0,3) = M_vector(0,3);
		M_out(1,0) = M_vector(1,0);
		M_out(1,1) = M_vector(1,1);
		M_out(1,2) = M_vector(1,2);
		M_out(1,3) = M_vector(1,3);
		M_out(2,0) = M_vector(2,0);
		M_out(2,1) = M_vector(2,1);
		M_out(2,2) = M_vector(2,2);
		M_out(2,3) = M_vector(2,3);
		M_out(3,3) = 1.0;

		M_out(0,4) = M_vector(0,8);
		M_out(0,5) = M_vector(0,9);
		M_out(0,6) = M_vector(0,10);
		M_out(0,7) = M_vector(0,11);
		M_out(1,4) = M_vector(1,8);
		M_out(1,5) = M_vector(1,9);
		M_out(1,6) = M_vector(1,10);
		M_out(1,7) = M_vector(1,11);
		M_out(2,4) = M_vector(2,8);
		M_out(2,5) = M_vector(2,9);
		M_out(2,6) = M_vector(2,10);
		M_out(2,7) = M_vector(2,11);
		M_out(3,7) = 1.0;

	}

	return M_out;

}
*/


vector<arma::Mat<double>> InitState::FindWingRefFrame(arma::Mat<double> M_body, arma::Mat<double> wing_L_pcl, arma::Mat<double> wing_R_pcl) {

	// Fit oriented bounding boxes to the pointclouds:

	arma::Mat<double> M_L(4,16); // 4 possible orientations left wing

	if (arma::sum(wing_L_pcl.row(1)) != 0.0) {
		struct bbox bbox_L = InitState::BoundingBox(wing_L_pcl);
		arma::Mat<double> c_points_L = InitState::FindRootTip(M_body, bbox_L);
		arma::Col<double> root_L(3);
		arma::Col<double> tip_L(3);
		root_L(0) = 0.25*(c_points_L(0,0)+c_points_L(0,1)+c_points_L(0,2)+c_points_L(0,3));
		root_L(1) = 0.25*(c_points_L(1,0)+c_points_L(1,1)+c_points_L(1,2)+c_points_L(1,3));
		root_L(2) = 0.25*(c_points_L(2,0)+c_points_L(2,1)+c_points_L(2,2)+c_points_L(2,3));
		tip_L(0) = 0.25*(c_points_L(0,4)+c_points_L(0,5)+c_points_L(0,6)+c_points_L(0,7));
		tip_L(1) = 0.25*(c_points_L(1,4)+c_points_L(1,5)+c_points_L(1,6)+c_points_L(1,7));
		tip_L(2) = 0.25*(c_points_L(2,4)+c_points_L(2,5)+c_points_L(2,6)+c_points_L(2,7));

		for (int i=0; i<4; i++) {
			arma::Mat<double> base(3,3);
			arma::Col<double> x_vec(3);
			arma::Col<double> y_vec(3);
			arma::Col<double> z_vec(3);

			x_vec(0) = c_points_L(0,i)-root_L(0);
			x_vec(1) = c_points_L(1,i)-root_L(1);
			x_vec(2) = c_points_L(2,i)-root_L(2);
			base.col(0) = x_vec/arma::norm(x_vec,2);
			y_vec(0) = tip_L(0)-root_L(0);
			y_vec(1) = tip_L(1)-root_L(1);
			y_vec(2) = tip_L(2)-root_L(2);
			base.col(1) = y_vec/arma::norm(y_vec,2);
			z_vec = arma::cross(x_vec,y_vec);
			base.col(2) = z_vec/arma::norm(z_vec,2);

			M_L(0,i*4) 	 = base(0,0);
			M_L(0,i*4+1) = base(0,1);
			M_L(0,i*4+2) = base(0,2);
			M_L(0,i*4+3) = root_L(0);
			M_L(1,i*4) 	 = base(1,0);
			M_L(1,i*4+1) = base(1,1);
			M_L(1,i*4+2) = base(1,2);
			M_L(1,i*4+3) = root_L(1);
			M_L(2,i*4) 	 = base(2,0);
			M_L(2,i*4+1) = base(2,1);
			M_L(2,i*4+2) = base(2,2);
			M_L(2,i*4+3) = root_L(2);
			M_L(3,i*4) 	 = 0.0;
			M_L(3,i*4+1) = 0.0;
			M_L(3,i*4+2) = 0.0;
			M_L(3,i*4+3) = 1.0;
		}
	}
	else {
		M_L.zeros(4,16);
	}
	
	arma::Mat<double> M_R(4,16); // 4 possible orientations right wing

	if (arma::sum(wing_R_pcl.row(1)) != 0.0) {
		struct bbox bbox_R = InitState::BoundingBox(wing_R_pcl);
		arma::Mat<double> c_points_R = InitState::FindRootTip(M_body, bbox_R);
		arma::Col<double> root_R(3);
		arma::Col<double> tip_R(3);
		root_R(0) = 0.25*(c_points_R(0,0)+c_points_R(0,1)+c_points_R(0,2)+c_points_R(0,3));
		root_R(1) = 0.25*(c_points_R(1,0)+c_points_R(1,1)+c_points_R(1,2)+c_points_R(1,3));
		root_R(2) = 0.25*(c_points_R(2,0)+c_points_R(2,1)+c_points_R(2,2)+c_points_R(2,3));
		tip_R(0) = 0.25*(c_points_R(0,4)+c_points_R(0,5)+c_points_R(0,6)+c_points_R(0,7));
		tip_R(1) = 0.25*(c_points_R(1,4)+c_points_R(1,5)+c_points_R(1,6)+c_points_R(1,7));
		tip_R(2) = 0.25*(c_points_R(2,4)+c_points_R(2,5)+c_points_R(2,6)+c_points_R(2,7));

		for (int i=0; i<4; i++) {
			arma::Mat<double> base(3,3);
			arma::Col<double> x_vec(3);
			arma::Col<double> y_vec(3);
			arma::Col<double> z_vec(3);

			x_vec(0) = c_points_R(0,i)-root_R(0);
			x_vec(1) = c_points_R(1,i)-root_R(1);
			x_vec(2) = c_points_R(2,i)-root_R(2);
			//base.row(0) = arma::trans(x_vec/arma::norm(x_vec,2));
			base.col(0) = x_vec/arma::norm(x_vec,2);
			y_vec(0) = -tip_R(0)+root_R(0);
			y_vec(1) = -tip_R(1)+root_R(1);
			y_vec(2) = -tip_R(2)+root_R(2);
			base.col(1) = y_vec/arma::norm(y_vec,2);
			z_vec = arma::cross(x_vec,y_vec);
			base.col(2) = z_vec/arma::norm(z_vec,2);

			M_R(0,i*4) 	 = base(0,0);
			M_R(0,i*4+1) = base(0,1);
			M_R(0,i*4+2) = base(0,2);
			M_R(0,i*4+3) = root_R(0);
			M_R(1,i*4) 	 = base(1,0);
			M_R(1,i*4+1) = base(1,1);
			M_R(1,i*4+2) = base(1,2);
			M_R(1,i*4+3) = root_R(1);
			M_R(2,i*4) 	 = base(2,0);
			M_R(2,i*4+1) = base(2,1);
			M_R(2,i*4+2) = base(2,2);
			M_R(2,i*4+3) = root_R(2);
			M_R(3,i*4) 	 = 0.0;
			M_R(3,i*4+1) = 0.0;
			M_R(3,i*4+2) = 0.0;
			M_R(3,i*4+3) = 1.0;
		}
	}
	else {
		M_R.zeros(4,16);
	}

	vector<arma::Mat<double>> output_vec;

	output_vec.push_back(M_L);
	output_vec.push_back(M_R);

	return output_vec;
}


arma::Mat<double> InitState::FindBodyRefFrame(struct bbox &body_box, arma::Mat<double> &body_pcl) {

	// I don't really like this, it assumes some properties of the body which are probably not universal

	arma::Mat<double> M_body = {{body_box.R_box[0][0],body_box.R_box[0][1],body_box.R_box[0][2],body_box.box_center[0]},
			{body_box.R_box[1][0],body_box.R_box[1][1],body_box.R_box[1][2],body_box.box_center[1]},
			{body_box.R_box[2][0],body_box.R_box[2][1],body_box.R_box[2][2],body_box.box_center[2]},
			{0.0, 0.0, 0.0 ,1.0}};

	arma::Mat<double> M_body_T = {{body_box.R_box[0][0],body_box.R_box[1][0],body_box.R_box[2][0]},
			{body_box.R_box[0][1],body_box.R_box[1][1],body_box.R_box[2][1]},
			{body_box.R_box[0][2],body_box.R_box[1][2],body_box.R_box[2][2]}};

	// Convert points to body reference frame:

 	int N_points = body_pcl.n_cols;

 	arma::Mat<double> body_points(3,N_points);

 	body_points.row(0) = body_pcl.row(1);
 	body_points.row(1) = body_pcl.row(2);
 	body_points.row(2) = body_pcl.row(3);

 	arma::Col<double> body_center = {body_box.box_center[0], body_box.box_center[1], body_box.box_center[2]};

 	body_points.each_col() -= body_center;

 	arma::Mat<double> body_points_T = M_body_T*body_points;

 	arma::Mat<double> seg_x1;
 	arma::Mat<double> seg_x2;
 	arma::Mat<double> seg_y1;
 	arma::Mat<double> seg_y2;

 	seg_x1 = body_points_T.cols(arma::find(body_points_T.row(0)<((1.0/2.0)*body_box.box_corners[0][0])));
 	seg_x2 = body_points_T.cols(arma::find(body_points_T.row(0)>((1.0/2.0)*body_box.box_corners[0][2])));
 	seg_y1 = body_points_T.cols(arma::find(body_points_T.row(1)<((1.0/4.0)*body_box.box_corners[1][0])));
 	seg_y2 = body_points_T.cols(arma::find(body_points_T.row(1)>((1.0/4.0)*body_box.box_corners[1][4])));

 	arma::Col<double> cg_x1 = arma::mean(seg_x1,1);
 	arma::Col<double> cg_x2 = arma::mean(seg_x2,1);
 	arma::Col<double> cg_y1 = arma::mean(seg_y1,1);
 	arma::Col<double> cg_y2 = arma::mean(seg_y2,1);

 	double Ix_1 = 0.0;
 	double Ix_2 = 0.0;
 	double Iy_1 = 0.0;
 	double Iy_2 = 0.0;

 	for (int i=0; i<seg_x1.n_cols; i++) {
 		Ix_1 = Ix_1+pow(seg_x1(1,i)-cg_x1(1),2.0)+pow(seg_x1(2,i)-cg_x1(2),2.0);
 	}

 	for (int i=0; i<seg_x2.n_cols; i++) {
 		Ix_2 = Ix_2+pow(seg_x2(1,i)-cg_x2(1),2.0)+pow(seg_x2(2,i)-cg_x2(2),2.0);
 	}

	for (int i=0; i<seg_y1.n_cols; i++) {
 		Iy_1 = Iy_1+pow(seg_y1(1,i),2.0);
 	}

 	for (int i=0; i<seg_y2.n_cols; i++) {
 		Iy_2 = Iy_2+pow(seg_y2(1,i),2.0);
 	}

 	//cout << M_body << endl;

 	if (Ix_1 > Ix_2) {
 		// Abdomen is located on negative x-axis: switch y and z axes
 		M_body = {{M_body(0,0),M_body(0,2),M_body(0,1),M_body(0,3)},
 				{M_body(1,0),M_body(1,2),M_body(1,1),M_body(1,3)},
 				{M_body(2,0),M_body(2,2),M_body(2,1),M_body(2,3)},
 				{M_body(3,0),M_body(3,2),M_body(3,1),M_body(3,3)}};
 	}
 	else if (Ix_1 < Ix_2) {
 		// Abdomen is located on the positive x-axis: rotate by 180 degrees and switch y and z axes
 		M_body = {{-M_body(0,0),M_body(0,2),M_body(0,1),M_body(0,3)},
 				{-M_body(1,0),M_body(1,2),M_body(1,1),M_body(1,3)},
 				{-M_body(2,0),M_body(2,2),M_body(2,1),M_body(2,3)},
 				{-M_body(3,0),M_body(3,2),M_body(3,1),M_body(3,3)}};
 	}
 	else {
 		// Don't know where the abdomen is located: switch y and z axes
 		M_body = {{M_body(0,0),M_body(0,2),M_body(0,1),M_body(0,3)},
 				{M_body(1,0),M_body(1,2),M_body(1,1),M_body(1,3)},
 				{M_body(2,0),M_body(2,2),M_body(2,1),M_body(2,3)},
 				{M_body(3,0),M_body(3,2),M_body(3,1),M_body(3,3)}};
 	}

 	if (Iy_1 > Iy_2) {
 		// c.g. is located at the bottom, z-axis is oriented correctly
 	}
 	else if (Iy_1 < Iy_2) {
 		// c.g is located at the top, rotate 180 degrees around x-axis
 		M_body = {{M_body(0,0),M_body(0,1),-M_body(0,2),M_body(0,3)},
 				{M_body(1,0),M_body(1,1),-M_body(1,2),M_body(1,3)},
 				{M_body(2,0),M_body(2,1),-M_body(2,2),M_body(2,3)},
 				{M_body(3,0),M_body(3,1),-M_body(3,2),M_body(3,3)}};
 	}
 	else {
 		// Don't know where the c.g. is located: keep current M_body
 	}

 	arma::Col<double> x_axis = {M_body(0,0),M_body(1,0),M_body(2,0)};
 	arma::Col<double> z_axis = {M_body(0,2),M_body(1,2),M_body(2,2)};

 	arma::Col<double> y_axis_M = {M_body(0,1),M_body(1,1),M_body(2,1)};
 	arma::Col<double> y_axis_cross = arma::cross(x_axis,z_axis);

 	double y_dot = arma::dot(y_axis_M,y_axis_cross)/(arma::norm(y_axis_M,2)*arma::norm(y_axis_cross,2));

 	if (y_dot > 0.5) {
 		// Switch sign y-axis:
 		M_body = {{M_body(0,0),-M_body(0,1),M_body(0,2),M_body(0,3)},
 				{M_body(1,0),-M_body(1,1),M_body(1,2),M_body(1,3)},
 				{M_body(2,0),-M_body(2,1),M_body(2,2),M_body(2,3)},
 				{M_body(3,0),-M_body(3,1),M_body(3,2),M_body(3,3)}};
 	}

	return M_body;
}

arma::Mat<double> InitState::FindRootTip(arma::Mat<double> M_body, struct bbox &seg_box) {

	arma::Mat<double> rot_body_T = {{M_body(0,0),M_body(1,0),M_body(2,0),0.0},
			{M_body(0,1),M_body(1,1),M_body(2,1),0.0},
			{M_body(0,2),M_body(1,2),M_body(2,2),0.0}};

	arma::Mat<double> rot_mat = {{seg_box.R_box[0][0], seg_box.R_box[0][1], seg_box.R_box[0][2], seg_box.box_center[0]},
			{seg_box.R_box[1][0], seg_box.R_box[1][1], seg_box.R_box[1][2], seg_box.box_center[1]},
			{seg_box.R_box[2][0], seg_box.R_box[2][1], seg_box.R_box[2][2], seg_box.box_center[2]},
			{0.0, 0.0, 0.0, 1.0}};

	arma::Col<double> body_center = {M_body(0,3), M_body(1,3), M_body(2,3), 1.0};

	arma::Mat<double> corners_seg(4,8);
	arma::Row<double> corner_dist(8);

	for (int i=0; i<8; i++) {
		arma::Col<double> temp_corner = {seg_box.box_corners[0][i], seg_box.box_corners[1][i], seg_box.box_corners[2][i], 1.0};
		corners_seg.col(i) = rot_mat*temp_corner;
		corner_dist(i) = sqrt(pow(corners_seg(0,i)-body_center(0),2.0)+pow(corners_seg(1,i)-body_center(1),2.0)+pow(corners_seg(2,i)-body_center(2),2.0));
	}

	// Select the 4 corners which are closest to the body center and the 4 corners which are the furthest away:

	arma::uvec c_sorted = arma::sort_index(corner_dist);

	arma::Mat<double> c_points(4,12);

	// Sorted corners + distances

	c_points(0,0) = corners_seg(0,c_sorted(0));
	c_points(1,0) = corners_seg(1,c_sorted(0));
	c_points(2,0) = corners_seg(2,c_sorted(0));
	c_points(3,0) = corner_dist(c_sorted(0));

	c_points(0,1) = corners_seg(0,c_sorted(1));
	c_points(1,1) = corners_seg(1,c_sorted(1));
	c_points(2,1) = corners_seg(2,c_sorted(1));
	c_points(3,1) = corner_dist(c_sorted(1));

	c_points(0,2) = corners_seg(0,c_sorted(2));
	c_points(1,2) = corners_seg(1,c_sorted(2));
	c_points(2,2) = corners_seg(2,c_sorted(2));
	c_points(3,2) = corner_dist(c_sorted(2));

	c_points(0,3) = corners_seg(0,c_sorted(3));
	c_points(1,3) = corners_seg(1,c_sorted(3));
	c_points(2,3) = corners_seg(2,c_sorted(3));
	c_points(3,3) = corner_dist(c_sorted(3));

	c_points(0,4) = corners_seg(0,c_sorted(4));
	c_points(1,4) = corners_seg(1,c_sorted(4));
	c_points(2,4) = corners_seg(2,c_sorted(4));
	c_points(3,4) = corner_dist(c_sorted(4));

	c_points(0,5) = corners_seg(0,c_sorted(5));
	c_points(1,5) = corners_seg(1,c_sorted(5));
	c_points(2,5) = corners_seg(2,c_sorted(5));
	c_points(3,5) = corner_dist(c_sorted(5));

	c_points(0,6) = corners_seg(0,c_sorted(6));
	c_points(1,6) = corners_seg(1,c_sorted(6));
	c_points(2,6) = corners_seg(2,c_sorted(6));
	c_points(3,6) = corner_dist(c_sorted(6));

	c_points(0,7) = corners_seg(0,c_sorted(7));
	c_points(1,7) = corners_seg(1,c_sorted(7));
	c_points(2,7) = corners_seg(2,c_sorted(7));
	c_points(3,7) = corner_dist(c_sorted(7));

	// root point

	c_points(0,8) = 0.25*(corners_seg(0,c_sorted(0))+corners_seg(0,c_sorted(1))+corners_seg(0,c_sorted(2))+corners_seg(0,c_sorted(3)))-body_center(0);
	c_points(1,8) = 0.25*(corners_seg(1,c_sorted(0))+corners_seg(1,c_sorted(1))+corners_seg(1,c_sorted(2))+corners_seg(1,c_sorted(3)))-body_center(1);
	c_points(2,8) = 0.25*(corners_seg(2,c_sorted(0))+corners_seg(2,c_sorted(1))+corners_seg(2,c_sorted(2))+corners_seg(2,c_sorted(3)))-body_center(2);
	c_points(3,8) = sqrt(pow(c_points(0,8),2.0)+pow(c_points(1,8),2.0)+pow(c_points(2,8),2.0));

	// tip point

	c_points(0,9) = 0.25*(corners_seg(0,c_sorted(4))+corners_seg(0,c_sorted(5))+corners_seg(0,c_sorted(6))+corners_seg(0,c_sorted(7)))-body_center(0);
	c_points(1,9) = 0.25*(corners_seg(1,c_sorted(4))+corners_seg(1,c_sorted(5))+corners_seg(1,c_sorted(6))+corners_seg(1,c_sorted(7)))-body_center(1);
	c_points(2,9) = 0.25*(corners_seg(2,c_sorted(4))+corners_seg(2,c_sorted(5))+corners_seg(2,c_sorted(6))+corners_seg(2,c_sorted(7)))-body_center(2);
	c_points(3,9) = sqrt(pow(c_points(0,9),2.0)+pow(c_points(1,9),2.0)+pow(c_points(2,9),2.0));

	// angular orientation w.r.t. body ref frame and volume
	arma::Col<double> tip_body_ref = rot_body_T*c_points.col(9);
	double xsi = atan2(tip_body_ref(1),tip_body_ref(2));
	double theta = atan2(tip_body_ref(0),sqrt(pow(tip_body_ref(1),2)+pow(tip_body_ref(2),2)));
	double seg_L = sqrt(pow((c_points(0,9)-c_points(0,8)),2)+pow((c_points(1,9)-c_points(1,8)),2)+pow((c_points(2,9)-c_points(2,8)),2));
	c_points(0,10) = xsi;
	c_points(1,10) = theta;
	c_points(2,10) = seg_L;
	c_points(3,10) = seg_box.volume;

	// center point
	c_points(0,11) = seg_box.box_center[0]-body_center(0);
	c_points(1,11) = seg_box.box_center[1]-body_center(1);
	c_points(2,11) = seg_box.box_center[2]-body_center(2);
	c_points(3,11) = sqrt(pow(c_points(0,11),2.0)+pow(c_points(1,11),2.0)+pow(c_points(2,11),2.0));

	// Project a point one wing length away from the wing tip along the y-axis of the 

	return c_points;
}

vector<tuple<struct bbox, arma::Mat<double>>> InitState::FindBoundingBoxes(arma::Mat<double> &pcl_in) {

	// Find the bounding boxes for all the bounding boxes in the image:
	arma::Row<double> pcl_intensity = pcl_in.row(0);
	arma::Row<double> segment_id = arma::unique(pcl_intensity);

	double body_id = segment_id.min();

	int N_segments = segment_id.n_cols;

	arma::Mat<double> body_pcl;
	struct bbox body_box;
	vector<arma::Mat<double>> segment_pcl;
	arma::Mat<double> seg_pcl;
	vector<struct bbox> segment_box;

	for (int i=0; i<N_segments; i++) {
		if (segment_id(i)==body_id) {
			arma::uvec seg_indices = arma::find(pcl_intensity == segment_id(i));
			body_pcl = pcl_in.cols(seg_indices);
			body_box = InitState::BoundingBox(body_pcl);
		}
		else {
			arma::uvec seg_indices = arma::find(pcl_intensity == segment_id(i));
			if (seg_indices.n_rows > 10) {
				seg_pcl = pcl_in.cols(seg_indices);
				segment_pcl.push_back(seg_pcl);
				segment_box.push_back(InitState::BoundingBox(seg_pcl));
			}
		}
	}

	// Return bounding boxes with body bounding box as the first item:
	vector<tuple<struct bbox, arma::Mat<double>>> bbox_list;
	for (int j=0; j<(segment_box.size()+1); j++) {
		if (j==0) {
			bbox_list.push_back(make_tuple(body_box,body_pcl));
		}
		else {
			bbox_list.push_back(make_tuple(segment_box[j-1],segment_pcl[j-1]));
		}
	}

	return bbox_list;
}

bbox InitState::BoundingBox(arma::Mat<double> &pcl_in) {

	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in (new pcl::PointCloud<pcl::PointXYZ>());
	*cloud_in = InitState::Convert_Mat_2_PCL_XYZ(pcl_in);

	pcl::MomentOfInertiaEstimation <pcl::PointXYZ> feature_extractor;
  	feature_extractor.setInputCloud (cloud_in);
  	feature_extractor.compute ();

  	// Oriented Bounding Box properties

  	vector<float> moment_of_inertia;
  	vector<float> eccentricity;
	pcl::PointXYZ min_point_OBB;
	pcl::PointXYZ max_point_OBB;
	pcl::PointXYZ position_OBB;
	Eigen::Matrix3f rotational_matrix_OBB;
  	float major_value, middle_value, minor_value;
  	Eigen::Vector3f major_vector, middle_vector, minor_vector;
  	Eigen::Vector3f mass_center;

  	feature_extractor.getMomentOfInertia(moment_of_inertia);
	feature_extractor.getEccentricity(eccentricity);
	feature_extractor.getOBB(min_point_OBB, max_point_OBB, position_OBB, rotational_matrix_OBB);
	feature_extractor.getEigenValues(major_value, middle_value, minor_value);
	feature_extractor.getEigenVectors(major_vector, middle_vector, minor_vector);
	feature_extractor.getMassCenter(mass_center);

	Eigen::Vector3f position(position_OBB.x, position_OBB.y, position_OBB.z);
  	Eigen::Quaternionf quat(rotational_matrix_OBB);

  	// Return bounding box structure

  	struct bbox bounding_box;

  	bounding_box.box_center[0] = (double) position_OBB.x;
  	bounding_box.box_center[1] = (double) position_OBB.y;
  	bounding_box.box_center[2] = (double) position_OBB.z;
  	bounding_box.box_corners[0][0] = (double) min_point_OBB.x;
  	bounding_box.box_corners[1][0] = (double) min_point_OBB.y;
  	bounding_box.box_corners[2][0] = (double) min_point_OBB.z;
	bounding_box.box_corners[0][1] = (double) min_point_OBB.x;
  	bounding_box.box_corners[1][1] = (double) min_point_OBB.y;
  	bounding_box.box_corners[2][1] = (double) max_point_OBB.z;
  	bounding_box.box_corners[0][2] = (double) max_point_OBB.x;
  	bounding_box.box_corners[1][2] = (double) min_point_OBB.y;
  	bounding_box.box_corners[2][2] = (double) max_point_OBB.z;
  	bounding_box.box_corners[0][3] = (double) max_point_OBB.x;
  	bounding_box.box_corners[1][3] = (double) min_point_OBB.y;
  	bounding_box.box_corners[2][3] = (double) min_point_OBB.z;
  	bounding_box.box_corners[0][4] = (double) min_point_OBB.x;
  	bounding_box.box_corners[1][4] = (double) max_point_OBB.y;
  	bounding_box.box_corners[2][4] = (double) min_point_OBB.z;
  	bounding_box.box_corners[0][5] = (double) min_point_OBB.x;
  	bounding_box.box_corners[1][5] = (double) max_point_OBB.y;
  	bounding_box.box_corners[2][5] = (double) max_point_OBB.z;
  	bounding_box.box_corners[0][6] = (double) max_point_OBB.x;
  	bounding_box.box_corners[1][6] = (double) max_point_OBB.y;
  	bounding_box.box_corners[2][6] = (double) max_point_OBB.z;
  	bounding_box.box_corners[0][7] = (double) max_point_OBB.x;
  	bounding_box.box_corners[1][7] = (double) max_point_OBB.y;
  	bounding_box.box_corners[2][7] = (double) min_point_OBB.z;
	bounding_box.q_box[0] = (double) quat.w();
	bounding_box.q_box[1] = (double) quat.x();
	bounding_box.q_box[2] = (double) quat.y();
	bounding_box.q_box[3] = (double) quat.z();
	bounding_box.R_box[0][0] = (double) rotational_matrix_OBB(0,0);
	bounding_box.R_box[0][1] = (double) rotational_matrix_OBB(0,1);
	bounding_box.R_box[0][2] = (double) rotational_matrix_OBB(0,2);
	bounding_box.R_box[1][0] = (double) rotational_matrix_OBB(1,0);
	bounding_box.R_box[1][1] = (double) rotational_matrix_OBB(1,1);
	bounding_box.R_box[1][2] = (double) rotational_matrix_OBB(1,2);
	bounding_box.R_box[2][0] = (double) rotational_matrix_OBB(2,0);
	bounding_box.R_box[2][1] = (double) rotational_matrix_OBB(2,1);
	bounding_box.R_box[2][2] = (double) rotational_matrix_OBB(2,2);
	bounding_box.mass_center[0] = (double)  mass_center(0);
	bounding_box.mass_center[1] = (double)  mass_center(1);
	bounding_box.mass_center[2] = (double)  mass_center(2);
	bounding_box.eigen_values[0] = (double) major_value;
	bounding_box.eigen_values[1] = (double) middle_value;
	bounding_box.eigen_values[2] = (double) minor_value;
	bounding_box.eigen_vectors[0][0] = (double) major_vector(0);
	bounding_box.eigen_vectors[0][1] = (double) major_vector(1);
	bounding_box.eigen_vectors[0][2] = (double) major_vector(2);
	bounding_box.eigen_vectors[1][0] = (double) middle_vector(0);
	bounding_box.eigen_vectors[1][1] = (double) middle_vector(1);
	bounding_box.eigen_vectors[1][2] = (double) middle_vector(2);
	bounding_box.eigen_vectors[2][0] = (double) minor_vector(0);
	bounding_box.eigen_vectors[2][1] = (double) minor_vector(1);
	bounding_box.eigen_vectors[2][2] = (double) minor_vector(2);
	bounding_box.moment_of_inertia[0] = (double) moment_of_inertia[0];
	bounding_box.moment_of_inertia[1] = (double) moment_of_inertia[1];
	bounding_box.moment_of_inertia[2] = (double) moment_of_inertia[2];
	bounding_box.eccentricity[0] = (double) eccentricity[0];
	bounding_box.eccentricity[1] = (double) eccentricity[1];
	bounding_box.eccentricity[2] = (double) eccentricity[2];
	bounding_box.volume = (max_point_OBB.x-min_point_OBB.x)*(max_point_OBB.y-min_point_OBB.y)*(max_point_OBB.z-min_point_OBB.z);

	return bounding_box;
}

np::ndarray InitState::ReturnBBoxSinglePCL(frames &frame_in) {

	arma::Mat<double> pcl_in = InitState::ConvertVectorPCLArma(frame_in.single_frame_pcl);

	vector<tuple<struct bbox, arma::Mat<double>>> bbox_list = InitState::FindBoundingBoxes(pcl_in);

	int N_boxes = bbox_list.size();

	p::tuple shape = p::make_tuple(24,N_boxes);
	np::dtype dtype = np::dtype::get_builtin<double>();
	np::ndarray array_out = np::zeros(shape,dtype);

	for (int i=0; i<N_boxes; i++) {
		arma::Mat<double> rot_mat = {{get<0>(bbox_list[i]).R_box[0][0], get<0>(bbox_list[i]).R_box[0][1], get<0>(bbox_list[i]).R_box[0][2], get<0>(bbox_list[i]).box_center[0]},
			{get<0>(bbox_list[i]).R_box[1][0], get<0>(bbox_list[i]).R_box[1][1], get<0>(bbox_list[i]).R_box[1][2], get<0>(bbox_list[i]).box_center[1]},
			{get<0>(bbox_list[i]).R_box[2][0], get<0>(bbox_list[i]).R_box[2][1], get<0>(bbox_list[i]).R_box[2][2], get<0>(bbox_list[i]).box_center[2]},
			{0.0, 0.0, 0.0, 1.0}};
		for (int j=0; j<8; j++) {
			arma::Col<double> corner_point = {get<0>(bbox_list[i]).box_corners[0][j], get<0>(bbox_list[i]).box_corners[1][j],
				get<0>(bbox_list[i]).box_corners[2][j], 1.0};
			arma::Col<double> new_corner_point = rot_mat*corner_point;

			array_out[j*3][i] = new_corner_point(0);
			array_out[j*3+1][i] = new_corner_point(1);
			array_out[j*3+2][i] = new_corner_point(2);
		}		
	}

	return array_out;
}

np::ndarray InitState::ReturnSinglePCL(frames &frame_in) {

	vector<tuple<int,double,double,double,double,double,double>> pcl_out = frame_in.single_frame_pcl;

	int N_points = pcl_out.size();

	p::tuple shape = p::make_tuple(7,N_points);
	np::dtype dtype = np::dtype::get_builtin<double>();
	np::ndarray array_out = np::zeros(shape,dtype);

	for (int i=0; i<N_points; i++) {
		array_out[0][i] = get<0>(pcl_out[i]);
		array_out[1][i] = get<1>(pcl_out[i]);
		array_out[2][i] = get<2>(pcl_out[i]);
		array_out[3][i] = get<3>(pcl_out[i]);
		array_out[4][i] = get<4>(pcl_out[i]);
		array_out[5][i] = get<5>(pcl_out[i]);
		array_out[6][i] = get<6>(pcl_out[i]);
	}

	return array_out;
}

np::ndarray InitState::ReturnBBoxBLR(frames &frame_in) {

	p::tuple shape = p::make_tuple(24,3);
	np::dtype dtype = np::dtype::get_builtin<double>();
	np::ndarray array_out = np::zeros(shape,dtype);

	for (int i=0; i<3; i++) {
		arma::Mat<double> pcl_now = frame_in.pcl_init_single_frame[i];

		if (pcl_now.n_cols>1) {

			struct bbox bbox_now = InitState::BoundingBox(pcl_now);

			arma::Mat<double> rot_mat = {{bbox_now.R_box[0][0], bbox_now.R_box[0][1],bbox_now.R_box[0][2], bbox_now.box_center[0]},
				{bbox_now.R_box[1][0], bbox_now.R_box[1][1], bbox_now.R_box[1][2], bbox_now.box_center[1]},
				{bbox_now.R_box[2][0], bbox_now.R_box[2][1], bbox_now.R_box[2][2], bbox_now.box_center[2]},
				{0.0, 0.0, 0.0, 1.0}};

			for (int j=0; j<8; j++) {
				arma::Col<double> corner_point = {bbox_now.box_corners[0][j],bbox_now.box_corners[1][j],bbox_now.box_corners[2][j], 1.0};
				arma::Col<double> new_corner_point = rot_mat*corner_point;

				array_out[j*3][i] = new_corner_point(0);
				array_out[j*3+1][i] = new_corner_point(1);
				array_out[j*3+2][i] = new_corner_point(2);
			}

		}
		else {
			for (int j=0; j<8; j++) {
				array_out[j*3][i] = 0.0;
				array_out[j*3+1][i] = 0.0;
				array_out[j*3+2][i] = 0.0;
			}
		}
	}

	return array_out;
}

np::ndarray InitState::ReturnBLRPCL(frames &frame_in) {

	arma::Mat<double> pcl_body = frame_in.pcl_init_single_frame[0];
	arma::Mat<double> pcl_wing_L = frame_in.pcl_init_single_frame[1];
	arma::Mat<double> pcl_wing_R = frame_in.pcl_init_single_frame[2];

	int N_points_body = pcl_body.n_cols;
	int N_points_wing_L = pcl_wing_L.n_cols;
	int N_points_wing_R = pcl_wing_R.n_cols;

	if (N_points_body < 2) {
		N_points_body = 0;
	}
	if (N_points_wing_L < 2) {
		N_points_wing_L = 0;
	}
	if (N_points_wing_R < 2) {
		N_points_wing_R = 0;
	}

	p::tuple shape = p::make_tuple(7,N_points_body+N_points_wing_L+N_points_wing_R);
	np::dtype dtype = np::dtype::get_builtin<double>();
	np::ndarray array_out = np::zeros(shape,dtype);

	for (int i=0; i<N_points_body; i++) {
		array_out[0][i] = pcl_body(0,i);
		array_out[1][i] = pcl_body(1,i);
		array_out[2][i] = pcl_body(2,i);
		array_out[3][i] = pcl_body(3,i);
		array_out[4][i] = pcl_body(4,i);
		array_out[5][i] = pcl_body(5,i);
		array_out[6][i] = pcl_body(6,i);
	}

	for (int j=0; j<N_points_wing_L; j++) {
		array_out[0][j+N_points_body] = pcl_wing_L(0,j);
		array_out[1][j+N_points_body] = pcl_wing_L(1,j);
		array_out[2][j+N_points_body] = pcl_wing_L(2,j);
		array_out[3][j+N_points_body] = pcl_wing_L(3,j);
		array_out[4][j+N_points_body] = pcl_wing_L(4,j);
		array_out[5][j+N_points_body] = pcl_wing_L(5,j);
		array_out[6][j+N_points_body] = pcl_wing_L(6,j);
	}

	for (int k=0; k<N_points_wing_R; k++) {
		array_out[0][k+N_points_body+N_points_wing_L] = pcl_wing_R(0,k);
		array_out[1][k+N_points_body+N_points_wing_L] = pcl_wing_R(1,k);
		array_out[2][k+N_points_body+N_points_wing_L] = pcl_wing_R(2,k);
		array_out[3][k+N_points_body+N_points_wing_L] = pcl_wing_R(3,k);
		array_out[4][k+N_points_body+N_points_wing_L] = pcl_wing_R(4,k);
		array_out[5][k+N_points_body+N_points_wing_L] = pcl_wing_R(5,k);
		array_out[6][k+N_points_body+N_points_wing_L] = pcl_wing_R(6,k);
	}

	return array_out;

}

np::ndarray InitState::ReturnMBody(frames &frame_in) {

	arma::Mat<double> M_body = frame_in.M_init_single_frame[0];

	p::tuple shape = p::make_tuple(4,4);
	np::dtype dtype = np::dtype::get_builtin<double>();
	np::ndarray array_out = np::zeros(shape,dtype);

	if (M_body(3,3)==1.0) {
		for (int i=0; i<4; i++) {
			for (int j=0; j<4; j++) {
				array_out[i][j] = M_body(i,j);
			}
		}
	}
	else {
		for (int i=0; i<4; i++) {
			for (int j=0; j<4; j++) {
				if (i==j) {
					array_out[i][j] = 1.0;
				}
				else {
					array_out[i][j] = 0.0;
				}
			}
		}

	}

	return array_out;
}

np::ndarray InitState::ReturnMWingL(frames &frame_in) {

	arma::Mat<double> M_wing_L = frame_in.M_init_single_frame[1];

	p::tuple shape = p::make_tuple(4,4);
	np::dtype dtype = np::dtype::get_builtin<double>();
	np::ndarray array_out = np::zeros(shape,dtype);

	if (M_wing_L(3,3)==1.0) {
		for (int i=0; i<4; i++) {
			for (int j=0; j<4; j++) {
				array_out[i][j] = M_wing_L(i,j);
			}
		}
	}

	return array_out;
}

np::ndarray InitState::ReturnMWingR(frames &frame_in) {

	arma::Mat<double> M_wing_R = frame_in.M_init_single_frame[2];

	p::tuple shape = p::make_tuple(4,4);
	np::dtype dtype = np::dtype::get_builtin<double>();
	np::ndarray array_out = np::zeros(shape,dtype);

	if (M_wing_R(3,3)==1.0) {
		for (int i=0; i<4; i++) {
			for (int j=0; j<4; j++) {
				array_out[i][j] = M_wing_R(i,j);
			}
		}
	}

	return array_out;
}

pcl::PointCloud<pcl::PointXYZ> InitState::Convert_Mat_2_PCL_XYZ(arma::Mat<double> &pcl_mat) {

	// Convert an armadillo matrix to a PointCloud vector

	pcl::PointCloud<pcl::PointXYZ> pcl_vec;

	int N_points = pcl_mat.n_cols;

	for (int i=0; i<N_points; i++) {
		pcl::PointXYZ point;
		point.x = pcl_mat(1,i);
		point.y = pcl_mat(2,i);
		point.z = pcl_mat(3,i);

		pcl_vec.push_back(point);
	}

	return pcl_vec;
}

arma::Mat<double> InitState::ConvertVectorPCLArma(vector<tuple<int,double,double,double,double,double,double>> &pcl_in) {

	int N_points = pcl_in.size();

	arma::Mat<double> pcl_out(7,N_points);

	for (int i=0; i<N_points; i++) {
		pcl_out(0,i) = (double) get<0>(pcl_in[i]);
		pcl_out(1,i) = get<1>(pcl_in[i]);
		pcl_out(2,i) = get<2>(pcl_in[i]);
		pcl_out(3,i) = get<3>(pcl_in[i]);
		pcl_out(4,i) = get<4>(pcl_in[i]);
		pcl_out(5,i) = get<5>(pcl_in[i]);
		pcl_out(6,i) = get<6>(pcl_in[i]);
	}

	return pcl_out;
}
#ifndef FIND_INITIAL_STATE_H
#define FIND_INITIAL_STATE_H

#include "session_param.h"
#include "frames.h"
#include "model.h"
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
#include <pcl/point_types.h>

#include "focal_grid.h"

using namespace std;

namespace p = boost::python;
namespace np = boost::python::numpy;

struct bbox {
	double box_center [3];
	double box_corners [3][8];
	double q_box [4];
	double R_box [3][3];
	double mass_center [3];
	double eigen_values [3];
	double eigen_vectors [3][3];
	double moment_of_inertia [3];
	double eccentricity [3];
	double volume;
};

class InitState
{
	
	public:

		// Class

		InitState();

		// Parameters
		double w_xsi;
		double w_theta;
		double w_length;
		double w_volume;
		double cone_angle;
		double cone_height;

		// Functions

		void SetWeightXsi(double WXsi);
		void SetWeightTheta(double WTheta);
		void SetWeightLength(double WLength);
		void SetWeightVolume(double WVolume);
		void SetConeAngle(double ConeAngle);
		void SetConeHeight(double ConeHeight);
		void ProjectSingleFrame(FocalGrid &fg, frames &frame_in, vox_grid &vox);
		void ProjectFrameBatch(FocalGrid &fg, frames &frame_in, vox_grid &vox);
		void FindInitialStateSingleFrame(frames &frame_in, model &mod);
		void FindInitialStateFrameBatch(frames &frame_in, model &mod);
		vector<tuple<arma::Mat<double>,arma::Mat<double>>> FindBodyandWing(vector<tuple<struct bbox, arma::Mat<double>>> &bbox_vector, double wing_length);
		vector<arma::Mat<double>> FindWingPCL(vector<arma::Mat<double>> wing_pcls, vector<arma::Mat<double>> wing_prop, double wing_length);
		arma::Mat<double> FindBodyRefFrame(struct bbox &body_box, arma::Mat<double> &body_pcl);
		//arma::Mat<double> FindWingOrientation(arma::Mat<double> M_vector, arma::Mat<double> &wing_pcl_in, int L_or_R);
		vector<arma::Mat<double>> FindWingRefFrame(arma::Mat<double> M_body, arma::Mat<double> wing_L_pcl, arma::Mat<double> wing_R_pcl);
		arma::Mat<double> FindRootTip(arma::Mat<double> M_body, struct bbox &seg_box);
		vector<tuple<struct bbox, arma::Mat<double>>> FindBoundingBoxes(arma::Mat<double> &pcl_in);
		bbox BoundingBox(arma::Mat<double> &pcl_in);
		np::ndarray ReturnBBoxSinglePCL(frames &frame_in);
		np::ndarray ReturnSinglePCL(frames &frame_in);
		np::ndarray ReturnBBoxBLR(frames &frame_in);
		np::ndarray ReturnBLRPCL(frames &frame_in);
		np::ndarray ReturnMBody(frames &frame_in);
		np::ndarray ReturnMWingL(frames &frame_in);
		np::ndarray ReturnMWingR(frames &frame_in);
		pcl::PointCloud<pcl::PointXYZ> Convert_Mat_2_PCL_XYZ(arma::Mat<double> &pcl_mat);
		arma::Mat<double> ConvertVectorPCLArma(vector<tuple<int,double,double,double,double,double,double>> &pcl_in);

};
#endif
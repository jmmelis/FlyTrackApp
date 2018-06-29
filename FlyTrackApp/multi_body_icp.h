#ifndef MULTI_BODY_ICP_H
#define MULTI_BODY_ICP_H

#include "vox_grid.h"
#include "model.h"
#include "focal_grid.h"

#include <boost/thread/thread.hpp>
#include <string>
#include <stdint.h>
#include <iostream>
#include <iomanip>
#include <math.h>
#include <chrono>
#include <vector>
#include <thread>
#include <algorithm>
#include <dirent.h>
#include <armadillo>
#include <pcl/console/parse.h>
#include <pcl/common/transforms.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/vtk_lib_io.h>
#include <pcl/point_types.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/features/normal_3d.h>
#include <pcl/surface/gp3.h>

using namespace std;

typedef pcl::PointNormal PointNT;
typedef pcl::PointCloud<PointNT> PointCloudT;

class MultiBodyICP{

	public:

		MultiBodyICP();

		arma::Mat<double> SingleFrameICP(frames &frame_in, model &mod, vox_grid &vox, FocalGrid &fcg);
		pcl::PointCloud<pcl::PointXYZ> GetModelSRCPointCloud(model &mod, arma::Mat<double> M_init, int seg_ind);
		tuple<arma::Mat<double>,double> ICP_on_single_segment(pcl::PointCloud<pcl::PointXYZ> &dest_pcl, pcl::PointCloud<pcl::PointXYZ> &src_pcl);
		tuple<arma::Mat<double>,double> RobustPoseEstimation(PointCloudT &dest_pcl, PointCloudT &src_pcl);
		arma::Mat<double> CalculateMeshTransform(model &mod, arma::Mat<double> M_init, int seg_ind);
		arma::Mat<double> ReturnSRCPCL(frames &frame_in, model &mod, vox_grid &vox, FocalGrid &fcg);
		vector<arma::Mat<int>> ReturnProjectedImages(frames &frame_in, model &mod, vox_grid &vox, FocalGrid &fcg);
		//tuple<arma::Mat<double>,double> PairAlign(arma::Mat<double> &dest_pcl, arma::Mat<double> &src_pcl);
		tuple<arma::Mat<double>,double> PairAlign(arma::Mat<double> &dest_pcl, model &mod, arma::Mat<double> M_init, int seg_ind);
		tuple<arma::Mat<double>,arma::Mat<double>,double> ICPAlgorithm(model &mod, vox_grid &vox, FocalGrid &fcg, arma::Mat<double> &dest_pcl, arma::Mat<double> M_init, int seg_ind, int int_val);
		double ProjectedModelScore(model &mod, vox_grid &vox, FocalGrid &fcg, arma::Mat<double> &dest_pcl, arma::Mat<double> M_init, int seg_ind, int int_val);
		arma::Mat<double> GetProjectedModelPCL(model &mod, vox_grid &vox, FocalGrid &fcg, arma::Mat<double> M_init, int seg_ind, int int_val);
		vector<arma::Col<int>> GetProjectedImages(model &mod, vox_grid &vox, FocalGrid &fcg, arma::Mat<double> M_init, int seg_ind, int int_val);
		//vector<tuple<double,double,double,int>> ProjectImage2PointCloud(vector<arma::Col<int>> &frame_in, vox_grid &vox, FocalGrid &fcg);
		//vector<arma::Col<int>> ProjectModel2Image(model &insect_model, vox_grid &vox, FocalGrid &fcg);
		//vector<tuple<double,double,double,int>> GetModelPointCloud(model &insect_model);
		pcl::PolygonMesh TransformMesh(pcl::PolygonMesh mesh_in, arma::Mat<double> transform_mat);
		pcl::PointCloud<pcl::PointXYZ> GetPointCloudFromMesh(pcl::PolygonMesh mesh_in);
		arma::Mat<double> MultiplyMat(arma::Mat<double> M_a, arma::Mat<double> M_b);
		arma::Mat<double> TransformMat(arma::Col<double> q_in, double scale);
		arma::Mat<double> Convert_PCL_Vec_2_Mat(vector<tuple<int,double,double,double,double,double,double>> pcl_in);
		vector<tuple<double,double,double,int>> Convert_PCL_XYZ_2_Vec(pcl::PointCloud<pcl::PointXYZ> pcl_in, int ind);
		vector<tuple<double,double,double,int>> Convert_Mat_2_PCL_Vec(arma::Mat<double> &pcl_in, int ind);
		pcl::PointCloud<pcl::PointXYZ> Convert_Mat_2_PCL_XYZ(arma::Mat<double> pcl_mat);
		arma::Mat<double> GetM(arma::Col<double> state_in);
		PointCloudT TransferPointXYZ2PointNT(pcl::PointCloud<pcl::PointXYZ> pcl_in);
		arma::Mat<double> TransferPointCloudT2Mat(PointCloudT pcl_in, int int_val);

};
#endif
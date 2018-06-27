#ifndef MODEL_H
#define MODEL_H

#include <string>
#include <stdint.h>
#include <iostream>
#include <iomanip>
#include <math.h>
#include <chrono>
#include <vector>
#include <algorithm>
#include <dirent.h>
#include <armadillo>
#include <pcl/io/pcd_io.h>
#include <pcl/io/vtk_lib_io.h>
#include <pcl/point_types.h>
#include <pcl/console/parse.h>
#include <pcl/common/transforms.h>
#include <pcl/PCLPointCloud2.h>

using namespace std;

struct model {
	int N_parts;
	vector<string> stl_list;
	vector<pcl::PolygonMesh> parts;
	vector<vector<int>> parents;
	vector<arma::Col<double>> joint_param_parent;
	vector<arma::Col<double>> joint_param_child;
	vector<arma::Col<double>> state;
	vector<arma::Mat<double>> M_vec;
	vector<arma::Col<double>> bounding_box_config;
	vector<double> scale;
	vector<double> alpha;
	arma::Col<double> origin_loc;
	double body_length;
	double wing_length;
};
#endif
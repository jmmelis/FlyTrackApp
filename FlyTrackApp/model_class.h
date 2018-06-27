#ifndef MODEL_CLASS_H
#define MODEL_CLASS_H

#include "session_param.h"
#include "model.h"
#include "frames.h"

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

class ModelClass
{
	
	public:

		// Class

		ModelClass();

		// Parameters

		// Functions
		void SetScale(model &mod, arma::Col<double> scale_vec);
		void SetOrigin(model &mod, arma::Col<double> origin_loc);
		void SetBodyLength(model &mod, double body_length);
		void SetWingLength(model &mod, double wing_length);
		void SetStartState(model &mod);
		bool SaveCurrentModel(session_param &ses_par, model &mod, string file_name);
		bool LoadModel(session_param &ses_par, model &mod, int model_ind);
		pcl::PolygonMesh LoadSTLFile(string FileName, string FileLoc);
		vector<string> ReturnSTLNames(model &mod);
		void SetState(model &mod, arma::Mat<double> state_mat);
		void SetInitState(model &mod, frames &frame);
		vector<arma::Mat<double>> ReturnTransformMatrices(model &mod);
		arma::Mat<double> GetM(arma::Col<double> state_in);
		arma::Col<double> GetStateVec(arma::Mat<double> M_in);
		arma::Col<double> MultiplyStates(arma::Col<double> state_A, arma::Col<double> state_B);
};
#endif
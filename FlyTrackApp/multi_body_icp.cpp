#include "multi_body_icp.h"

#include "vox_grid.h"
#include "frames.h"
#include "model.h"
#include "focal_grid.h"

#include <boost/thread/thread.hpp>
#include <boost/make_shared.hpp>
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
#include <nlopt.h>
#include <pcl/console/parse.h>
#include <pcl/common/transforms.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/vtk_lib_io.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/features/normal_3d.h>
#include <pcl/surface/gp3.h>
#include <pcl/registration/icp.h>
#include <Eigen/Core>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/point_representation.h>
#include <pcl/common/time.h>
#include <pcl/console/print.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/fpfh_omp.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/sample_consensus_prerejective.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/registration/icp_nl.h>
#include <pcl/registration/transforms.h>

using namespace std;

// Types
typedef pcl::PointXYZ PointT;
typedef pcl::PointNormal PointNT;
typedef pcl::PointCloud<PointT> PointCloud;
typedef pcl::PointCloud<PointNT> PointCloudT;
typedef pcl::FPFHSignature33 FeatureT;
typedef pcl::FPFHEstimationOMP<PointNT,PointNT,FeatureT> FeatureEstimationT;
typedef pcl::PointCloud<FeatureT> FeatureCloudT;

// Define a new point representation for < x, y, z, curvature >
class MyPointRepresentation : public pcl::PointRepresentation <PointNT>
{
  using pcl::PointRepresentation<PointNT>::nr_dimensions_;
public:
  MyPointRepresentation ()
  {
    // Define the number of dimensions
    nr_dimensions_ = 4;
  }

  // Override the copyToFloatArray method to define our feature vector
  virtual void copyToFloatArray (const PointNT &p, float * out) const
  {
    // < x, y, z, curvature >
    out[0] = p.x;
    out[1] = p.y;
    out[2] = p.z;
    out[3] = p.curvature;
  }
};

MultiBodyICP::MultiBodyICP() {
  // empty
}

/*
vector<arma::Mat<double>> MultiBodyICP::SingleFrameICP(frames &frame_in, model &mod) {

  vector<arma::Mat<double>> M_align_vec;

  arma::Mat<double> body_pcl = frame_in.pcl_init_single_frame[0];
  arma::Mat<double> wing_L_pcl = frame_in.pcl_init_single_frame[1];
  arma::Mat<double> wing_R_pcl = frame_in.pcl_init_single_frame[2];

  arma::Mat<double> M_body_init = frame_in.M_init_single_frame[0];
  arma::Mat<double> M_wing_L_init = frame_in.M_init_single_frame[1];
  arma::Mat<double> M_wing_R_init = frame_in.M_init_single_frame[2];

  // Calculate transformation matrices for all initial orientations:

  // body:

  arma::Mat<double> M_thorax_init = MultiBodyICP::CalculateMeshTransform(mod, M_body_init, 0);
  arma::Mat<double> M_head_init = MultiBodyICP::CalculateMeshTransform(mod, M_thorax_init, 1);
  arma::Mat<double> M_abdomen_init = MultiBodyICP::CalculateMeshTransform(mod, M_thorax_init, 2);

  pcl::PointCloud<pcl::PointXYZ> thorax_src_pcl = MultiBodyICP::GetModelSRCPointCloud(mod, M_thorax_init, 0);
  PointCloudT thorax_src = MultiBodyICP::TransferPointXYZ2PointNT(thorax_src_pcl);
  pcl::PointCloud<pcl::PointXYZ> head_src_pcl = MultiBodyICP::GetModelSRCPointCloud(mod, M_head_init, 1);
  PointCloudT head_src = MultiBodyICP::TransferPointXYZ2PointNT(head_src_pcl);
  pcl::PointCloud<pcl::PointXYZ> abdomen_src_pcl = MultiBodyICP::GetModelSRCPointCloud(mod, M_abdomen_init, 2);
  PointCloudT abdomen_src = MultiBodyICP::TransferPointXYZ2PointNT(abdomen_src_pcl);

  pcl::PointCloud<pcl::PointXYZ> body_dest_pcl = MultiBodyICP::Convert_Mat_2_PCL_XYZ(body_pcl);
  PointCloudT body_dest = MultiBodyICP::TransferPointXYZ2PointNT(body_dest_pcl);

  // Perform ICP on thorax
  //tuple<arma::Mat<double>,double> icp_result_thorax = MultiBodyICP::ICP_on_single_segment(body_dest_pcl, thorax_src_pcl);
  //tuple<arma::Mat<double>,double> icp_result_head = MultiBodyICP::ICP_on_single_segment(body_dest_pcl, head_src_pcl);
  //tuple<arma::Mat<double>,double> icp_result_abdomen = MultiBodyICP::ICP_on_single_segment(body_dest_pcl, abdomen_src_pcl);
  tuple<arma::Mat<double>,double> icp_result_thorax = MultiBodyICP::RobustPoseEstimation(body_dest, thorax_src);
  tuple<arma::Mat<double>,double> icp_result_head = MultiBodyICP::RobustPoseEstimation(body_dest, head_src);
  tuple<arma::Mat<double>,double> icp_result_abdomen = MultiBodyICP::RobustPoseEstimation(body_dest, abdomen_src);

  M_align_vec.push_back(get<0>(icp_result_thorax));
  M_align_vec.push_back(get<0>(icp_result_head));
  M_align_vec.push_back(get<0>(icp_result_abdomen));

  
  // left wing:

  arma::Mat<double> M_wing_L1_init = MultiBodyICP::CalculateMeshTransform(mod, M_wing_L_init.submat(0,0,3,3), 3);
  arma::Mat<double> M_wing_L2_init = MultiBodyICP::CalculateMeshTransform(mod, M_wing_L_init.submat(0,4,3,7), 3);
  arma::Mat<double> M_wing_L3_init = MultiBodyICP::CalculateMeshTransform(mod, M_wing_L_init.submat(0,8,3,11), 3);
  arma::Mat<double> M_wing_L4_init = MultiBodyICP::CalculateMeshTransform(mod, M_wing_L_init.submat(0,12,3,15), 3);

  pcl::PointCloud<pcl::PointXYZ> wing_L1_src_pcl = MultiBodyICP::GetModelSRCPointCloud(mod, M_wing_L1_init, 3);
  PointCloudT wing_L1_src = MultiBodyICP::TransferPointXYZ2PointNT(wing_L1_src_pcl);
  pcl::PointCloud<pcl::PointXYZ> wing_L2_src_pcl = MultiBodyICP::GetModelSRCPointCloud(mod, M_wing_L2_init, 3);
  PointCloudT wing_L2_src = MultiBodyICP::TransferPointXYZ2PointNT(wing_L2_src_pcl);
  pcl::PointCloud<pcl::PointXYZ> wing_L3_src_pcl = MultiBodyICP::GetModelSRCPointCloud(mod, M_wing_L3_init, 3);
  PointCloudT wing_L3_src = MultiBodyICP::TransferPointXYZ2PointNT(wing_L3_src_pcl);
  pcl::PointCloud<pcl::PointXYZ> wing_L4_src_pcl = MultiBodyICP::GetModelSRCPointCloud(mod, M_wing_L4_init, 3);
  PointCloudT wing_L4_src = MultiBodyICP::TransferPointXYZ2PointNT(wing_L4_src_pcl);

  pcl::PointCloud<pcl::PointXYZ> wing_L_dest_pcl = MultiBodyICP::Convert_Mat_2_PCL_XYZ(wing_L_pcl);
  PointCloudT wing_L_dest = MultiBodyICP::TransferPointXYZ2PointNT(wing_L_dest_pcl);

  // Perform ICP on the left wing:

  //tuple<arma::Mat<double>,double> icp_result_wing_L1 = MultiBodyICP::ICP_on_single_segment(wing_L_dest_pcl, wing_L1_src_pcl);
  //tuple<arma::Mat<double>,double> icp_result_wing_L2 = MultiBodyICP::ICP_on_single_segment(wing_L_dest_pcl, wing_L2_src_pcl);
  //tuple<arma::Mat<double>,double> icp_result_wing_L3 = MultiBodyICP::ICP_on_single_segment(wing_L_dest_pcl, wing_L3_src_pcl);
  //tuple<arma::Mat<double>,double> icp_result_wing_L4 = MultiBodyICP::ICP_on_single_segment(wing_L_dest_pcl, wing_L4_src_pcl);
  tuple<arma::Mat<double>,double> icp_result_wing_L1 = MultiBodyICP::RobustPoseEstimation(wing_L_dest, wing_L1_src);
  tuple<arma::Mat<double>,double> icp_result_wing_L2 = MultiBodyICP::RobustPoseEstimation(wing_L_dest, wing_L2_src);
  tuple<arma::Mat<double>,double> icp_result_wing_L3 = MultiBodyICP::RobustPoseEstimation(wing_L_dest, wing_L3_src);
  tuple<arma::Mat<double>,double> icp_result_wing_L4 = MultiBodyICP::RobustPoseEstimation(wing_L_dest, wing_L4_src);

  arma::Col<double> score_vec_L = {get<1>(icp_result_wing_L1),get<1>(icp_result_wing_L2),get<1>(icp_result_wing_L3),get<1>(icp_result_wing_L4)};

  if (score_vec_L.index_max()==0) {
    M_align_vec.push_back(get<0>(icp_result_wing_L1));
  }
  else if (score_vec_L.index_max()==1) {
    M_align_vec.push_back(get<0>(icp_result_wing_L2));
  }
  else if (score_vec_L.index_max()==2) {
    M_align_vec.push_back(get<0>(icp_result_wing_L3));
  }
  else if (score_vec_L.index_max()==3) {
    M_align_vec.push_back(get<0>(icp_result_wing_L4));
  }
  else {
    arma::Mat<double> zero_mat(4,4);
    zero_mat.zeros();
    M_align_vec.push_back(zero_mat);
  }

  
  // right wing:

  arma::Mat<double> M_wing_R1_init = MultiBodyICP::CalculateMeshTransform(mod, M_wing_R_init.submat(0,0,3,3), 4);
  arma::Mat<double> M_wing_R2_init = MultiBodyICP::CalculateMeshTransform(mod, M_wing_R_init.submat(0,4,3,7), 4);
  arma::Mat<double> M_wing_R3_init = MultiBodyICP::CalculateMeshTransform(mod, M_wing_R_init.submat(0,8,3,11), 4);
  arma::Mat<double> M_wing_R4_init = MultiBodyICP::CalculateMeshTransform(mod, M_wing_R_init.submat(0,12,3,15), 4);

  pcl::PointCloud<pcl::PointXYZ> wing_R1_src_pcl = MultiBodyICP::GetModelSRCPointCloud(mod, M_wing_R1_init, 4);
  PointCloudT wing_R1_src = MultiBodyICP::TransferPointXYZ2PointNT(wing_R1_src_pcl);
  pcl::PointCloud<pcl::PointXYZ> wing_R2_src_pcl = MultiBodyICP::GetModelSRCPointCloud(mod, M_wing_R2_init, 4);
  PointCloudT wing_R2_src = MultiBodyICP::TransferPointXYZ2PointNT(wing_R2_src_pcl);
  pcl::PointCloud<pcl::PointXYZ> wing_R3_src_pcl = MultiBodyICP::GetModelSRCPointCloud(mod, M_wing_R3_init, 4);
  PointCloudT wing_R3_src = MultiBodyICP::TransferPointXYZ2PointNT(wing_R3_src_pcl);
  pcl::PointCloud<pcl::PointXYZ> wing_R4_src_pcl = MultiBodyICP::GetModelSRCPointCloud(mod, M_wing_R4_init, 4);
  PointCloudT wing_R4_src = MultiBodyICP::TransferPointXYZ2PointNT(wing_R4_src_pcl);

  pcl::PointCloud<pcl::PointXYZ> wing_R_dest_pcl = MultiBodyICP::Convert_Mat_2_PCL_XYZ(wing_R_pcl);
  PointCloudT wing_R_dest = MultiBodyICP::TransferPointXYZ2PointNT(wing_R_dest_pcl);

  // Perform ICP on the right wing:

  //tuple<arma::Mat<double>,double> icp_result_wing_R1 = MultiBodyICP::ICP_on_single_segment(wing_R_dest_pcl, wing_R1_src_pcl);
  //tuple<arma::Mat<double>,double> icp_result_wing_R2 = MultiBodyICP::ICP_on_single_segment(wing_R_dest_pcl, wing_R2_src_pcl);
  //tuple<arma::Mat<double>,double> icp_result_wing_R3 = MultiBodyICP::ICP_on_single_segment(wing_R_dest_pcl, wing_R3_src_pcl);
  //tuple<arma::Mat<double>,double> icp_result_wing_R4 = MultiBodyICP::ICP_on_single_segment(wing_R_dest_pcl, wing_R4_src_pcl);
  tuple<arma::Mat<double>,double> icp_result_wing_R1 = MultiBodyICP::RobustPoseEstimation(wing_R_dest, wing_R1_src);
  tuple<arma::Mat<double>,double> icp_result_wing_R2 = MultiBodyICP::RobustPoseEstimation(wing_R_dest, wing_R2_src);
  tuple<arma::Mat<double>,double> icp_result_wing_R3 = MultiBodyICP::RobustPoseEstimation(wing_R_dest, wing_R3_src);
  tuple<arma::Mat<double>,double> icp_result_wing_R4 = MultiBodyICP::RobustPoseEstimation(wing_R_dest, wing_R4_src);

  arma::Col<double> score_vec_R = {get<1>(icp_result_wing_R1),get<1>(icp_result_wing_R2),get<1>(icp_result_wing_R3),get<1>(icp_result_wing_R4)};

  if (score_vec_R.index_max()==0) {
    M_align_vec.push_back(get<0>(icp_result_wing_R1));
  }
  else if (score_vec_R.index_max()==1) {
    M_align_vec.push_back(get<0>(icp_result_wing_R2));
  }
  else if (score_vec_R.index_max()==2) {
    M_align_vec.push_back(get<0>(icp_result_wing_R3));
  }
  else if (score_vec_R.index_max()==3) {
    M_align_vec.push_back(get<0>(icp_result_wing_R4));
  }
  else {
    arma::Mat<double> zero_mat(4,4);
    zero_mat.zeros();
    M_align_vec.push_back(zero_mat);
  }

  return M_align_vec;
}
*/

pcl::PointCloud<pcl::PointXYZ> MultiBodyICP::GetModelSRCPointCloud(model &mod, arma::Mat<double> M_init, int seg_ind) {

  pcl::PolygonMesh part_mesh = mod.parts[seg_ind];

  // Scale the part:

  double part_scale = mod.scale[seg_ind];

  arma::Mat<double> M_part = {{M_init(0,0)*part_scale, M_init(0,1)*part_scale, M_init(0,2)*part_scale, M_init(0,3)},
    {M_init(1,0)*part_scale, M_init(1,1)*part_scale, M_init(1,2)*part_scale, M_init(1,3)},
    {M_init(2,0)*part_scale, M_init(2,1)*part_scale, M_init(2,2)*part_scale, M_init(2,3)},
    {0.0, 0.0, 0.0, 1.0}};

  // Transform the mesh:

  pcl::PolygonMesh trans_mesh = MultiBodyICP::TransformMesh(part_mesh, M_part);

  // Get Pointcloud of the mesh:

  pcl::PointCloud<pcl::PointXYZ> pcl_out = MultiBodyICP::GetPointCloudFromMesh(trans_mesh);

  return pcl_out;
}

arma::Mat<double> MultiBodyICP::CalculateMeshTransform(model &mod, arma::Mat<double> M_init, int seg_ind) {

  arma::Mat<double> R_init = M_init.submat(0,0,2,2);
  arma::Col<double> T_init = {M_init(0,3), M_init(1,3), M_init(2,3)};

  double seg_scale = mod.scale[seg_ind];

  arma::Col<double> bbox_config = mod.bounding_box_config[seg_ind];
  arma::Mat<double> M_bbox = MultiBodyICP::GetM(bbox_config);
  arma::Mat<double> R_bbox = M_bbox.submat(0,0,2,2);
  arma::Col<double> T_bbox = {M_bbox(0,3)*seg_scale, M_bbox(1,3)*seg_scale, M_bbox(2,3)*seg_scale};

  arma::Mat<double> R_out = R_init*R_bbox;
  arma::Col<double> T_out = R_init*T_bbox+T_init;
  arma::Mat<double> M_out = {{R_out(0,0), R_out(0,1), R_out(0,2), T_out(0)},
    {R_out(1,0), R_out(1,1), R_out(1,2), T_out(1)},
    {R_out(2,0), R_out(2,1), R_out(2,2), T_out(2)},
    {0.0, 0.0, 0.0, 1.0}};
  mod.M_vec.push_back(M_out);

  return M_out;
}

tuple<arma::Mat<double>,double> MultiBodyICP::RobustPoseEstimation(PointCloudT &dest_pcl, PointCloudT &src_pcl) {

  int max_iter = 50000;
  double leaf_size = 0.10;
  double search_radius = 0.25;
  int N_samples = 3;
  int N_features = 5;

  // Pointclouds:
  PointCloudT::Ptr src (new PointCloudT);
  PointCloudT::Ptr src_aligned (new PointCloudT);
  PointCloudT::Ptr dest (new PointCloudT);
  FeatureCloudT::Ptr src_features (new FeatureCloudT);
  FeatureCloudT::Ptr dest_features (new FeatureCloudT);

  *src = src_pcl;
  *dest = dest_pcl;

  // Downsampling:
  pcl::VoxelGrid<PointNT> grid;
  grid.setLeafSize (leaf_size, leaf_size, leaf_size);
  grid.setInputCloud (src);
  grid.filter (*src);
  grid.setInputCloud (dest);
  grid.filter (*dest);

  // Estimate normals:
  pcl::NormalEstimationOMP<PointNT,PointNT> nest;
  nest.setRadiusSearch (search_radius);
  nest.setInputCloud (src);
  nest.compute (*src);
  nest.setInputCloud (dest);
  nest.compute (*dest);

  // Estimate features:
  FeatureEstimationT fest;
  fest.setRadiusSearch (search_radius);
  fest.setInputCloud (src);
  fest.setInputNormals (src);
  fest.compute (*src_features);
  fest.setInputCloud (dest);
  fest.setInputNormals (dest);
  fest.compute (*dest_features);

  // Perform alignment:
  pcl::SampleConsensusPrerejective<PointNT,PointNT,FeatureT> align;
  align.setInputSource (src);
  align.setSourceFeatures (src_features);
  align.setInputTarget (dest);
  align.setTargetFeatures (dest_features);
  align.setMaximumIterations (max_iter); // Number of RANSAC iterations
  align.setNumberOfSamples (N_samples); // Number of points to sample for generating/prejecting a pose
  align.setCorrespondenceRandomness (N_features); // Number of nearest features to use
  align.setSimilarityThreshold (0.5f); // Polygonal edge length similarity threshold
  align.setMaxCorrespondenceDistance (5.0f * leaf_size); // Inlier threshold
  align.setInlierFraction (0.1f); // Required inlier fraction ofr accepting a pose hypothesis

  {
    pcl::ScopeTime t("Alignment");
    align.align (*src_aligned);
  }

  arma::Mat<double> M_out(4,4);
  double score;

  if (align.hasConverged ()) {

    Eigen::Matrix4f transformation = align.getFinalTransformation ();

    M_out(0,0) = transformation(0,0);
    M_out(0,1) = transformation(0,1);
    M_out(0,2) = transformation(0,2);
    M_out(0,3) = transformation(0,3);
    M_out(1,0) = transformation(1,0);
    M_out(1,1) = transformation(1,1);
    M_out(1,2) = transformation(1,2);
    M_out(1,3) = transformation(1,3);
    M_out(2,0) = transformation(2,0);
    M_out(2,1) = transformation(2,1);
    M_out(2,2) = transformation(2,2);
    M_out(2,3) = transformation(2,3);
    M_out(3,0) = transformation(3,0);
    M_out(3,1) = transformation(3,1);
    M_out(3,2) = transformation(3,2);
    M_out(3,3) = transformation(3,3);

    score = 1.0;

    cout << "transformation matrix" << endl;
    cout << M_out << endl;
    cout << "inliers" << endl;
    cout << align.getInliers().size() << endl;
    cout << "" << endl;

  }
  else {
    cout << "aligned failed" << endl;
  }

  return make_tuple(M_out,score);
}

tuple<arma::Mat<double>,double> MultiBodyICP::ICP_on_single_segment(pcl::PointCloud<pcl::PointXYZ> &dest_pcl, pcl::PointCloud<pcl::PointXYZ> &src_pcl) {

  pcl::PointCloud<pcl::PointXYZ>::Ptr dest_cloud (new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr src_cloud (new pcl::PointCloud<pcl::PointXYZ>);

  *dest_cloud = dest_pcl;
  *src_cloud = src_pcl;

  pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
  icp.setInputCloud(src_cloud);
  icp.setInputTarget(dest_cloud);

  pcl::PointCloud<pcl::PointXYZ> Final;
  icp.align(Final);

  int icp_converged = icp.hasConverged();
  double icp_score = icp.getFitnessScore();
  Eigen::Matrix4f M_final = icp.getFinalTransformation();

  cout << "has converged:" << icp_converged << " score: " << icp_score << endl;
  cout << M_final << endl;

  arma::Mat<double> M_out(4,4);

  M_out(0,0) = M_final(0,0);
  M_out(0,1) = M_final(0,1);
  M_out(0,2) = M_final(0,2);
  M_out(0,3) = M_final(0,3);
  M_out(1,0) = M_final(1,0);
  M_out(1,1) = M_final(1,1);
  M_out(1,2) = M_final(1,2);
  M_out(1,3) = M_final(1,3);
  M_out(2,0) = M_final(2,0);
  M_out(2,1) = M_final(2,1);
  M_out(2,2) = M_final(2,2);
  M_out(2,3) = M_final(2,3);
  M_out(3,0) = M_final(3,0);
  M_out(3,1) = M_final(3,1);
  M_out(3,2) = M_final(3,2);
  M_out(3,3) = M_final(3,3);

  return make_tuple(M_out,icp_score);

}

vector<arma::Mat<int>> MultiBodyICP::ReturnProjectedImages(frames &frame_in, model &mod, vox_grid &vox, FocalGrid &fcg) {

  // Compute initial orientation matrices:

  arma::Mat<double> M_body_init = frame_in.M_init_single_frame[0];
  arma::Mat<double> M_wing_L_init = frame_in.M_init_single_frame[1];
  arma::Mat<double> M_wing_R_init = frame_in.M_init_single_frame[2];

  // body:
  arma::Mat<double> M_thorax_init  = MultiBodyICP::CalculateMeshTransform(mod, M_body_init, 0);
  arma::Mat<double> M_head_init    = MultiBodyICP::CalculateMeshTransform(mod, M_thorax_init, 1);
  arma::Mat<double> M_abdomen_init = MultiBodyICP::CalculateMeshTransform(mod, M_thorax_init, 2);

  // left wing:
  arma::Mat<double> M_wing_L1_init = MultiBodyICP::CalculateMeshTransform(mod, M_wing_L_init.submat(0,0,3,3), 3);

  // right wing:
  arma::Mat<double> M_wing_R1_init = MultiBodyICP::CalculateMeshTransform(mod, M_wing_R_init.submat(0,0,3,3), 4);

  // Project the model segments from 3d to 2d and back to 3d:

  vector<arma::Col<int>> mdl_pcl_thorax  = MultiBodyICP::GetProjectedImages(mod, vox, fcg, M_thorax_init, 0, 10);
  vector<arma::Col<int>> mdl_pcl_head    = MultiBodyICP::GetProjectedImages(mod, vox, fcg, M_head_init, 1, 10);
  vector<arma::Col<int>> mdl_pcl_abdomen = MultiBodyICP::GetProjectedImages(mod, vox, fcg, M_abdomen_init, 2, 10);
  vector<arma::Col<int>> mdl_pcl_wing_L1 = MultiBodyICP::GetProjectedImages(mod, vox, fcg, M_wing_L1_init, 3, 3);
  vector<arma::Col<int>> mdl_pcl_wing_R1 = MultiBodyICP::GetProjectedImages(mod, vox, fcg, M_wing_R1_init, 4, 3);

  vector<arma::Mat<int>> proj_frames;

  int N_frames = mdl_pcl_thorax.size();

  for (int i=0; i<N_frames; i++) {

    arma::Col<int> frame = mdl_pcl_thorax[i]+mdl_pcl_head[i]+mdl_pcl_abdomen[i]+mdl_pcl_wing_L1[i]+mdl_pcl_wing_R1[i];

    int N_row = get<0>(frame_in.image_size[i]);
    int N_col = get<1>(frame_in.image_size[i]);

    arma::Mat<int> frame_mat(N_row,N_col);

    for (int j=0; j<N_col; j++) {
      for (int k=0; k<N_row; k++) {
        if (frame(k*N_col+j)>10) {
          frame_mat(k,j) = 10;
        }
        else {
          frame_mat(k,j) = frame(k*N_col+j);
        }
      }
    }

    proj_frames.push_back(frame_mat);

  }

  return proj_frames;
}

vector<arma::Col<int>> MultiBodyICP::GetProjectedImages(model &mod, vox_grid &vox, FocalGrid &fcg, arma::Mat<double> M_init, int seg_ind, int int_val) {

  vector<arma::Mat<double>> proj_images;

  pcl::PointCloud<pcl::PointXYZ> seg_pcl = MultiBodyICP::GetModelSRCPointCloud(mod, M_init, seg_ind);

  vector<tuple<double,double,double,int>> seg_pcl_vec = MultiBodyICP::Convert_PCL_XYZ_2_Vec(seg_pcl, int_val);

  vector<arma::Col<int>> proj_frames = fcg.ProjectCloud2Image(seg_pcl_vec);

  return proj_frames;
}

arma::Mat<double> MultiBodyICP::ReturnSRCPCL(frames &frame_in, model &mod, vox_grid &vox, FocalGrid &fcg) {

  // Compute initial orientation matrices:

  arma::Mat<double> M_body_init   = frame_in.M_init_single_frame[0];
  arma::Mat<double> M_wing_L_init = frame_in.M_init_single_frame[1];
  arma::Mat<double> M_wing_R_init = frame_in.M_init_single_frame[2];

  // body:
  arma::Mat<double> M_thorax_init  = MultiBodyICP::CalculateMeshTransform(mod, M_body_init, 0);
  arma::Mat<double> M_head_init    = MultiBodyICP::CalculateMeshTransform(mod, M_thorax_init, 1);
  arma::Mat<double> M_abdomen_init = MultiBodyICP::CalculateMeshTransform(mod, M_thorax_init, 2);

  // left wing:
  arma::Mat<double> M_wing_L1_init = MultiBodyICP::CalculateMeshTransform(mod, M_wing_L_init.submat(0,0,3,3), 3);
  arma::Mat<double> M_wing_L2_init = MultiBodyICP::CalculateMeshTransform(mod, M_wing_L_init.submat(0,4,3,7), 3);
  arma::Mat<double> M_wing_L3_init = MultiBodyICP::CalculateMeshTransform(mod, M_wing_L_init.submat(0,8,3,11), 3);
  arma::Mat<double> M_wing_L4_init = MultiBodyICP::CalculateMeshTransform(mod, M_wing_L_init.submat(0,12,3,15), 3);

  // right wing:
  arma::Mat<double> M_wing_R1_init = MultiBodyICP::CalculateMeshTransform(mod, M_wing_R_init.submat(0,0,3,3), 4);
  arma::Mat<double> M_wing_R2_init = MultiBodyICP::CalculateMeshTransform(mod, M_wing_R_init.submat(0,4,3,7), 4);
  arma::Mat<double> M_wing_R3_init = MultiBodyICP::CalculateMeshTransform(mod, M_wing_R_init.submat(0,8,3,11), 4);
  arma::Mat<double> M_wing_R4_init = MultiBodyICP::CalculateMeshTransform(mod, M_wing_R_init.submat(0,12,3,15), 4);

  // Project the model segments from 3d to 2d and back to 3d:

  arma::Mat<double> mdl_pcl_thorax  = MultiBodyICP::GetProjectedModelPCL(mod, vox, fcg, M_thorax_init,  0, 1);
  arma::Mat<double> mdl_pcl_head    = MultiBodyICP::GetProjectedModelPCL(mod, vox, fcg, M_head_init,    1, 1);
  arma::Mat<double> mdl_pcl_abdomen = MultiBodyICP::GetProjectedModelPCL(mod, vox, fcg, M_abdomen_init, 2, 1);
  arma::Mat<double> mdl_pcl_wing_L1 = MultiBodyICP::GetProjectedModelPCL(mod, vox, fcg, M_wing_L1_init, 3, 1);
  arma::Mat<double> mdl_pcl_wing_L2 = MultiBodyICP::GetProjectedModelPCL(mod, vox, fcg, M_wing_L2_init, 3, 1);
  arma::Mat<double> mdl_pcl_wing_L3 = MultiBodyICP::GetProjectedModelPCL(mod, vox, fcg, M_wing_L3_init, 3, 1);
  arma::Mat<double> mdl_pcl_wing_L4 = MultiBodyICP::GetProjectedModelPCL(mod, vox, fcg, M_wing_L4_init, 3, 1);
  arma::Mat<double> mdl_pcl_wing_R1 = MultiBodyICP::GetProjectedModelPCL(mod, vox, fcg, M_wing_R1_init, 4, 1);
  arma::Mat<double> mdl_pcl_wing_R2 = MultiBodyICP::GetProjectedModelPCL(mod, vox, fcg, M_wing_R2_init, 4, 1);
  arma::Mat<double> mdl_pcl_wing_R3 = MultiBodyICP::GetProjectedModelPCL(mod, vox, fcg, M_wing_R3_init, 4, 1);
  arma::Mat<double> mdl_pcl_wing_R4 = MultiBodyICP::GetProjectedModelPCL(mod, vox, fcg, M_wing_R4_init, 4, 1);

  arma::Mat<double> pcl_mat_out = mdl_pcl_thorax;
  pcl_mat_out = arma::join_rows(pcl_mat_out,mdl_pcl_head);
  pcl_mat_out = arma::join_rows(pcl_mat_out,mdl_pcl_abdomen);
  pcl_mat_out = arma::join_rows(pcl_mat_out,mdl_pcl_wing_L1);
  pcl_mat_out = arma::join_rows(pcl_mat_out,mdl_pcl_wing_L2);
  pcl_mat_out = arma::join_rows(pcl_mat_out,mdl_pcl_wing_L3);
  pcl_mat_out = arma::join_rows(pcl_mat_out,mdl_pcl_wing_L4);
  pcl_mat_out = arma::join_rows(pcl_mat_out,mdl_pcl_wing_R1);
  pcl_mat_out = arma::join_rows(pcl_mat_out,mdl_pcl_wing_R2);
  pcl_mat_out = arma::join_rows(pcl_mat_out,mdl_pcl_wing_R3);
  pcl_mat_out = arma::join_rows(pcl_mat_out,mdl_pcl_wing_R4);

  return pcl_mat_out;
}

arma::Mat<double> MultiBodyICP::SingleFrameICP(frames &frame_in, model &mod, vox_grid &vox, FocalGrid &fcg) {

  clock_t start;
  double duration;

  start = clock();

  arma::Mat<double> body_pcl = frame_in.pcl_init_single_frame[0];
  arma::Mat<double> M_body_init   = frame_in.M_init_single_frame[0];

  arma::Mat<double> M_thorax_init  = MultiBodyICP::CalculateMeshTransform(mod, M_body_init, 0);
  arma::Mat<double> M_head_init    = MultiBodyICP::CalculateMeshTransform(mod, M_thorax_init, 1);
  arma::Mat<double> M_abdomen_init = MultiBodyICP::CalculateMeshTransform(mod, M_thorax_init, 2);

  tuple<arma::Mat<double>,arma::Mat<double>,double> thorax_icp_results  = MultiBodyICP::ICPAlgorithm(mod, vox, fcg, body_pcl, M_thorax_init,  0, 1);
  tuple<arma::Mat<double>,arma::Mat<double>,double> head_icp_results    = MultiBodyICP::ICPAlgorithm(mod, vox, fcg, body_pcl, M_head_init,  1, 3);
  tuple<arma::Mat<double>,arma::Mat<double>,double> abdomen_icp_results = MultiBodyICP::ICPAlgorithm(mod, vox, fcg, body_pcl, M_abdomen_init,  2, 5);

  arma::Mat<double> pcl_mat_out = get<0>(thorax_icp_results);
  pcl_mat_out = arma::join_rows(pcl_mat_out,get<0>(head_icp_results));
  pcl_mat_out = arma::join_rows(pcl_mat_out,get<0>(abdomen_icp_results));

  duration = (clock()-start)/ (double) CLOCKS_PER_SEC;

  cout << "ICP took" << endl;
  cout << duration << endl;

  return pcl_mat_out;

}


/*
arma::Mat<double> MultiBodyICP::SingleFrameICP(frames &frame_in, model &mod, vox_grid &vox, FocalGrid &fcg) {

  // Compute initial orientation matrices:

  arma::Mat<double> body_pcl = frame_in.pcl_init_single_frame[0];
  arma::Mat<double> wing_L_pcl = frame_in.pcl_init_single_frame[1];
  arma::Mat<double> wing_R_pcl = frame_in.pcl_init_single_frame[2];

  arma::Mat<double> M_body_init   = frame_in.M_init_single_frame[0];
  arma::Mat<double> M_wing_L_init = frame_in.M_init_single_frame[1];
  arma::Mat<double> M_wing_R_init = frame_in.M_init_single_frame[2];

  // body:
  arma::Mat<double> M_thorax_init  = MultiBodyICP::CalculateMeshTransform(mod, M_body_init, 0);
  arma::Mat<double> M_head_init    = MultiBodyICP::CalculateMeshTransform(mod, M_thorax_init, 1);
  arma::Mat<double> M_abdomen_init = MultiBodyICP::CalculateMeshTransform(mod, M_thorax_init, 2);

  // left wing:
  arma::Mat<double> M_wing_L1_init = MultiBodyICP::CalculateMeshTransform(mod, M_wing_L_init.submat(0,0,3,3), 3);
  arma::Mat<double> M_wing_L2_init = MultiBodyICP::CalculateMeshTransform(mod, M_wing_L_init.submat(0,4,3,7), 3);
  arma::Mat<double> M_wing_L3_init = MultiBodyICP::CalculateMeshTransform(mod, M_wing_L_init.submat(0,8,3,11), 3);
  arma::Mat<double> M_wing_L4_init = MultiBodyICP::CalculateMeshTransform(mod, M_wing_L_init.submat(0,12,3,15), 3);

  // right wing:
  arma::Mat<double> M_wing_R1_init = MultiBodyICP::CalculateMeshTransform(mod, M_wing_R_init.submat(0,0,3,3), 4);
  arma::Mat<double> M_wing_R2_init = MultiBodyICP::CalculateMeshTransform(mod, M_wing_R_init.submat(0,4,3,7), 4);
  arma::Mat<double> M_wing_R3_init = MultiBodyICP::CalculateMeshTransform(mod, M_wing_R_init.submat(0,8,3,11), 4);
  arma::Mat<double> M_wing_R4_init = MultiBodyICP::CalculateMeshTransform(mod, M_wing_R_init.submat(0,12,3,15), 4);

  // Project the model segments from 3d to 2d and back to 3d:

  arma::Mat<double> mdl_pcl_thorax  = MultiBodyICP::GetProjectedModelPCL(mod, vox, fcg, M_thorax_init,  0, 1);
  arma::Mat<double> mdl_pcl_head    = MultiBodyICP::GetProjectedModelPCL(mod, vox, fcg, M_head_init,    1, 1);
  arma::Mat<double> mdl_pcl_abdomen = MultiBodyICP::GetProjectedModelPCL(mod, vox, fcg, M_abdomen_init, 2, 1);
  arma::Mat<double> mdl_pcl_wing_L1 = MultiBodyICP::GetProjectedModelPCL(mod, vox, fcg, M_wing_L1_init, 3, 2);
  arma::Mat<double> mdl_pcl_wing_L2 = MultiBodyICP::GetProjectedModelPCL(mod, vox, fcg, M_wing_L2_init, 3, 2);
  arma::Mat<double> mdl_pcl_wing_L3 = MultiBodyICP::GetProjectedModelPCL(mod, vox, fcg, M_wing_L3_init, 3, 2);
  arma::Mat<double> mdl_pcl_wing_L4 = MultiBodyICP::GetProjectedModelPCL(mod, vox, fcg, M_wing_L4_init, 3, 2);
  arma::Mat<double> mdl_pcl_wing_R1 = MultiBodyICP::GetProjectedModelPCL(mod, vox, fcg, M_wing_R1_init, 4, 3);
  arma::Mat<double> mdl_pcl_wing_R2 = MultiBodyICP::GetProjectedModelPCL(mod, vox, fcg, M_wing_R2_init, 4, 3);
  arma::Mat<double> mdl_pcl_wing_R3 = MultiBodyICP::GetProjectedModelPCL(mod, vox, fcg, M_wing_R3_init, 4, 3);
  arma::Mat<double> mdl_pcl_wing_R4 = MultiBodyICP::GetProjectedModelPCL(mod, vox, fcg, M_wing_R4_init, 4, 3);


  // Perform ICP:
  tuple<arma::Mat<double>,arma::Mat<double>,double> thorax_icp_results  = MultiBodyICP::ICPAlgorithm(mod, vox, fcg, body_pcl, mdl_pcl_thorax,  M_thorax_init,  0, 1);
  tuple<arma::Mat<double>,arma::Mat<double>,double> head_icp_results    = MultiBodyICP::ICPAlgorithm(mod, vox, fcg, body_pcl, mdl_pcl_head,     M_head_init,    1, 2);
  tuple<arma::Mat<double>,arma::Mat<double>,double> abdomen_icp_results = MultiBodyICP::ICPAlgorithm(mod, vox, fcg, body_pcl, mdl_pcl_abdomen, M_abdomen_init, 2, 3);
  tuple<arma::Mat<double>,arma::Mat<double>,double> wing_L1_icp_results = MultiBodyICP::ICPAlgorithm(mod, vox, fcg, wing_L_pcl, mdl_pcl_wing_L1, M_wing_L1_init, 3, 5);
  tuple<arma::Mat<double>,arma::Mat<double>,double> wing_L2_icp_results = MultiBodyICP::ICPAlgorithm(mod, vox, fcg, wing_L_pcl, mdl_pcl_wing_L2, M_wing_L2_init, 3, 5);
  tuple<arma::Mat<double>,arma::Mat<double>,double> wing_L3_icp_results = MultiBodyICP::ICPAlgorithm(mod, vox, fcg, wing_L_pcl, mdl_pcl_wing_L3, M_wing_L1_init, 3, 5);
  tuple<arma::Mat<double>,arma::Mat<double>,double> wing_L4_icp_results = MultiBodyICP::ICPAlgorithm(mod, vox, fcg, wing_L_pcl, mdl_pcl_wing_L4, M_wing_L2_init, 3, 5);
  tuple<arma::Mat<double>,arma::Mat<double>,double> wing_R1_icp_results = MultiBodyICP::ICPAlgorithm(mod, vox, fcg, wing_R_pcl, mdl_pcl_wing_R1, M_wing_R1_init, 4, 10);
  tuple<arma::Mat<double>,arma::Mat<double>,double> wing_R2_icp_results = MultiBodyICP::ICPAlgorithm(mod, vox, fcg, wing_R_pcl, mdl_pcl_wing_R2, M_wing_R2_init, 4, 10);
  tuple<arma::Mat<double>,arma::Mat<double>,double> wing_R3_icp_results = MultiBodyICP::ICPAlgorithm(mod, vox, fcg, wing_R_pcl, mdl_pcl_wing_R3, M_wing_R1_init, 4, 10);
  tuple<arma::Mat<double>,arma::Mat<double>,double> wing_R4_icp_results = MultiBodyICP::ICPAlgorithm(mod, vox, fcg, wing_R_pcl, mdl_pcl_wing_R4, M_wing_R2_init, 4, 10);

  // Find best fit:
  arma::Mat<double> pcl_mat_out = get<0>(thorax_icp_results);
  pcl_mat_out = arma::join_rows(pcl_mat_out,get<0>(head_icp_results));
  pcl_mat_out = arma::join_rows(pcl_mat_out,get<0>(abdomen_icp_results));

  arma::Row<double> score_vec_L = {get<2>(wing_L1_icp_results), get<2>(wing_L2_icp_results),get<2>(wing_L3_icp_results), get<2>(wing_L4_icp_results)};
  arma::Row<double> score_vec_R = {get<2>(wing_R1_icp_results), get<2>(wing_R2_icp_results),get<2>(wing_R3_icp_results), get<2>(wing_R4_icp_results)};

  if (score_vec_L.index_min()==0) {
    pcl_mat_out = arma::join_rows(pcl_mat_out,get<0>(wing_L1_icp_results));
  }
  else if (score_vec_L.index_min()==1) {
    pcl_mat_out = arma::join_rows(pcl_mat_out,get<0>(wing_L2_icp_results));
  }
  else if (score_vec_L.index_min()==2) {
    pcl_mat_out = arma::join_rows(pcl_mat_out,get<0>(wing_L3_icp_results));
  }
  else if (score_vec_L.index_min()==3) {
    pcl_mat_out = arma::join_rows(pcl_mat_out,get<0>(wing_L4_icp_results));
  }

  if (score_vec_R.index_min()==0) {
    pcl_mat_out = arma::join_rows(pcl_mat_out,get<0>(wing_R1_icp_results));
  }
  else if (score_vec_R.index_min()==1) {
    pcl_mat_out = arma::join_rows(pcl_mat_out,get<0>(wing_R2_icp_results));
  }
  else if (score_vec_R.index_min()==2) {
    pcl_mat_out = arma::join_rows(pcl_mat_out,get<0>(wing_R3_icp_results));
  }
  else if (score_vec_R.index_min()==3) {
    pcl_mat_out = arma::join_rows(pcl_mat_out,get<0>(wing_R4_icp_results));
  }

  return pcl_mat_out;
}
*/

//tuple<arma::Mat<double>,double> MultiBodyICP::PairAlign(arma::Mat<double> &dest_pcl, arma::Mat<double> &src_pcl) {
tuple<arma::Mat<double>,double> MultiBodyICP::PairAlign(arma::Mat<double> &dest_pcl, model &mod, arma::Mat<double> M_init, int seg_ind) {

  //PointCloud cloud_src = MultiBodyICP::Convert_Mat_2_PCL_XYZ(src_pcl);
  //PointCloud cloud_tgt = MultiBodyICP::Convert_Mat_2_PCL_XYZ(dest_pcl);

  PointCloud cloud_src = MultiBodyICP::GetModelSRCPointCloud(mod, M_init, seg_ind);
  PointCloud cloud_tgt = MultiBodyICP::Convert_Mat_2_PCL_XYZ(dest_pcl);

  PointCloud::Ptr src (new PointCloud);
  PointCloud::Ptr tgt (new PointCloud);

  PointCloud::Ptr cloud_src2 (new PointCloud);
  PointCloud::Ptr cloud_tgt2 (new PointCloud);

  *cloud_src2 = cloud_src;
  *cloud_tgt2 = cloud_tgt;

  pcl::VoxelGrid<PointT> grid;

  grid.setLeafSize (0.06, 0.06, 0.06);
  grid.setInputCloud (cloud_src2);
  grid.filter (*src);

  grid.setInputCloud (cloud_tgt2);
  grid.filter (*tgt);

  pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
  //pcl::IterativeClosestPointNonLinear<pcl::PointXYZ, pcl::PointXYZ> icp;

  icp.setMaxCorrespondenceDistance(0.2);
  icp.setMaximumIterations(50);
  icp.setTransformationEpsilon(0.001);
  //icp.setEuclideanFitnessEpsilon(1);
  icp.setRANSACIterations(50);
  icp.setRANSACOutlierRejectionThreshold(0.06);

  icp.setInputCloud(src);
  icp.setInputTarget(tgt);

  pcl::PointCloud<pcl::PointXYZ> Final;

  icp.align(Final);

  double score = icp.getFitnessScore();

    // Get the transformation from target to source
  //targetToSource = Ti.inverse();

  Eigen::Matrix4f Ti = Eigen::Matrix4f::Identity ();

  Ti = icp.getFinalTransformation();

  arma::Mat<double> M_transform(4,4);

  M_transform(0,0) = Ti(0,0);
  M_transform(0,1) = Ti(0,1);
  M_transform(0,2) = Ti(0,2);
  M_transform(0,3) = Ti(0,3);
  M_transform(1,0) = Ti(1,0);
  M_transform(1,1) = Ti(1,1);
  M_transform(1,2) = Ti(1,2);
  M_transform(1,3) = Ti(1,3);
  M_transform(2,0) = Ti(2,0);
  M_transform(2,1) = Ti(2,1);
  M_transform(2,2) = Ti(2,2);
  M_transform(2,3) = Ti(2,3);
  M_transform(3,0) = 0.0;
  M_transform(3,1) = 0.0;
  M_transform(3,2) = 0.0;
  M_transform(3,3) = 1.0;

  return make_tuple(M_transform,score);

}

/*
//tuple<arma::Mat<double>,double> MultiBodyICP::PairAlign(arma::Mat<double> &dest_pcl, arma::Mat<double> &src_pcl) {
tuple<arma::Mat<double>,double> MultiBodyICP::PairAlign(arma::Mat<double> &dest_pcl, model &mod, arma::Mat<double> M_init, int seg_ind) {

  //PointCloud cloud_src = MultiBodyICP::Convert_Mat_2_PCL_XYZ(src_pcl);
  PointCloud cloud_src = MultiBodyICP::GetModelSRCPointCloud(mod, M_init, seg_ind);
  PointCloud cloud_tgt = MultiBodyICP::Convert_Mat_2_PCL_XYZ(dest_pcl);

  PointCloud::Ptr src (new PointCloud);
  PointCloud::Ptr tgt (new PointCloud);

  PointCloud::Ptr cloud_src2 (new PointCloud);
  PointCloud::Ptr cloud_tgt2 (new PointCloud);

  *cloud_src2 = cloud_src;
  *cloud_tgt2 = cloud_tgt;

  //*src = cloud_src;
  //*tgt = cloud_tgt;

  pcl::VoxelGrid<PointT> grid;

  grid.setLeafSize (0.06, 0.06, 0.06);
  grid.setInputCloud (cloud_src2);
  grid.filter (*src);

  grid.setInputCloud (cloud_tgt2);
  grid.filter (*tgt);

  // Compute surface normals and curvature
  PointCloudT::Ptr points_with_normals_src (new PointCloudT);
  PointCloudT::Ptr points_with_normals_tgt (new PointCloudT);

  //pcl::NormalEstimation<PointT, PointNT> norm_est;
  pcl::NormalEstimationOMP<PointT, PointNT> norm_est;
  pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ> ());
  norm_est.setSearchMethod (tree);
  norm_est.setKSearch (30);
    
  norm_est.setInputCloud (src);
  norm_est.compute (*points_with_normals_src);
  pcl::copyPointCloud (*src, *points_with_normals_src);

  norm_est.setInputCloud (tgt);
  norm_est.compute (*points_with_normals_tgt);
  pcl::copyPointCloud (*tgt, *points_with_normals_tgt);

  // Instantiate our custom point representation (defined above) ...
  MyPointRepresentation point_representation;
  // ... and weight the 'curvature' dimension so that it is balanced against x, y, and z
  float alpha[4] = {1.0, 1.0, 1.0, 1.0};
  point_representation.setRescaleValues (alpha);

  // Align
  pcl::IterativeClosestPointNonLinear<PointNT, PointNT> reg;
  reg.setTransformationEpsilon (0.001);
  //reg.setEuclideanFitnessEpsilon(1e-2);
  // Set the maximum distance between two correspondences (src<->tgt) to 10cm
  // Note: adjust this based on the size of your datasets
  reg.setMaxCorrespondenceDistance (0.2);
  // Set the point representation
  reg.setPointRepresentation (boost::make_shared<const MyPointRepresentation> (point_representation));

  reg.setInputSource (points_with_normals_src);
  reg.setInputTarget (points_with_normals_tgt);

  // Run the same optimization in a loop and visualize the results
  Eigen::Matrix4f Ti = Eigen::Matrix4f::Identity (), prev, targetToSource;
  PointCloudT::Ptr reg_result = points_with_normals_src;
  reg.setMaximumIterations (30);

  double score = 1.0;

  clock_t start;
  double duration;

  start = clock();

  for (int i = 0; i < 30; ++i) {
    // save cloud for visualization purpose
    points_with_normals_src = reg_result;

    // Estimate
    reg.setInputSource (points_with_normals_src);
    reg.align (*reg_result);

    //accumulate transformation between each Iteration
    Ti = reg.getFinalTransformation () * Ti;

    //if the difference between this transformation and the previous one
    //is smaller than the threshold, refine the process by reducing
    //the maximal correspondence distance
    if (fabs ((reg.getLastIncrementalTransformation () - prev).sum ()) < reg.getTransformationEpsilon ())
      reg.setMaxCorrespondenceDistance (reg.getMaxCorrespondenceDistance () - 0.001);
    
    prev = reg.getLastIncrementalTransformation ();

    score = reg.getFitnessScore();
  }

  duration = (clock()-start)/ (double) CLOCKS_PER_SEC;

  cout << "iteration time" << endl;
  cout << duration << endl;

  // Get the transformation from target to source
  //targetToSource = Ti.inverse();

  arma::Mat<double> M_transform(4,4);

  M_transform(0,0) = Ti(0,0);
  M_transform(0,1) = Ti(0,1);
  M_transform(0,2) = Ti(0,2);
  M_transform(0,3) = Ti(0,3);
  M_transform(1,0) = Ti(1,0);
  M_transform(1,1) = Ti(1,1);
  M_transform(1,2) = Ti(1,2);
  M_transform(1,3) = Ti(1,3);
  M_transform(2,0) = Ti(2,0);
  M_transform(2,1) = Ti(2,1);
  M_transform(2,2) = Ti(2,2);
  M_transform(2,3) = Ti(2,3);
  M_transform(3,0) = 0.0;
  M_transform(3,1) = 0.0;
  M_transform(3,2) = 0.0;
  M_transform(3,3) = 1.0;

  return make_tuple(M_transform,score);
}

/*
tuple<arma::Mat<double>,arma::Mat<double>,double> MultiBodyICP::ICPAlgorithm(model &mod, vox_grid &vox, FocalGrid &fcg, arma::Mat<double> &dest_pcl, arma::Mat<double> M_init, int seg_ind, int int_val) {

  arma::Mat<double> src_pcl_i;

  PointCloud cloud_src;
  PointCloud cloud_tgt;

  PointCloud::Ptr src (new PointCloud);
  PointCloud::Ptr tgt (new PointCloud);

  // Non-linear closest point:

  pcl::IterativeClosestPointNonLinear<pcl::PointXYZ, pcl::PointXYZ> icp;

  icp.setMaxCorrespondenceDistance(0.3);
  icp.setMaximumIterations(30);
  icp.setTransformationEpsilon(0.001);
  icp.setEuclideanFitnessEpsilon(1);
  icp.setRANSACIterations(30);
  icp.setRANSACOutlierRejectionThreshold(0.04);

  // Set target:
  cloud_tgt = MultiBodyICP::Convert_Mat_2_PCL_XYZ(dest_pcl);
  *tgt = cloud_tgt;
  icp.setInputTarget(tgt);

  // Iteration parameters:
  int max_iter = 10;
  double prev_score = 1e6;
  int i = 0;

  // Output parameters:
  arma::Mat<double> M_out;
  double score = 1.0;
  pcl::PointCloud<pcl::PointXYZ> Final;
  Eigen::Matrix4f Ti = Eigen::Matrix4f::Identity ();
  arma::Mat<double> M_delta(4,4);
  arma::Mat<double> M_update;
  arma::Mat<double> src_pcl_out;
  M_update = M_init;

  while ((i < max_iter) && ((score/prev_score)<1.02)) {

    // Set src:
    src_pcl_i = MultiBodyICP::GetProjectedModelPCL(mod,vox,fcg,M_update,seg_ind,int_val);
    cloud_src = MultiBodyICP::Convert_Mat_2_PCL_XYZ(src_pcl_i);
    *src = cloud_src;
    icp.setInputCloud(src);

    // Align:
    icp.align(Final);

    score = icp.getFitnessScore();

    Ti = icp.getFinalTransformation();

    M_delta(0,0) = Ti(0,0);
    M_delta(0,1) = Ti(0,1);
    M_delta(0,2) = Ti(0,2);
    M_delta(0,3) = Ti(0,3);
    M_delta(1,0) = Ti(1,0);
    M_delta(1,1) = Ti(1,1);
    M_delta(1,2) = Ti(1,2);
    M_delta(1,3) = Ti(1,3);
    M_delta(2,0) = Ti(2,0);
    M_delta(2,1) = Ti(2,1);
    M_delta(2,2) = Ti(2,2);
    M_delta(2,3) = Ti(2,3);
    M_delta(3,0) = 0.0;
    M_delta(3,1) = 0.0;
    M_delta(3,2) = 0.0;
    M_delta(3,3) = 1.0;

    M_update = M_delta*M_update;

    if ((score/prev_score)<1.02) {
      src_pcl_out = src_pcl_i;
      M_out = M_update;
      prev_score = score;
    }
    else {
      i = max_iter;
    }

    i++;
  }

  return make_tuple(src_pcl_out,M_out,score);
}
*/


tuple<arma::Mat<double>,arma::Mat<double>,double> MultiBodyICP::ICPAlgorithm(model &mod, vox_grid &vox, FocalGrid &fcg, arma::Mat<double> &dest_pcl, arma::Mat<double> M_init, int seg_ind, int int_val) {

  arma::Mat<double> M_update;
  M_update = M_init;

  tuple<arma::Mat<double>,double> pair_update = MultiBodyICP::PairAlign(dest_pcl, mod, M_init, seg_ind);

  double score = get<1>(pair_update);

  cout << "score" << endl;
  cout << score << endl;

  arma::Mat<double> M_delta = get<0>(pair_update);

  cout << M_delta << endl;

  M_update.submat(0,0,2,2) = M_delta.submat(0,0,2,2)*M_update.submat(0,0,2,2);
  M_update(0,3) = M_update(0,3)+M_delta(0,3);
  M_update(1,3) = M_update(1,3)+M_delta(1,3);
  M_update(2,3) = M_update(2,3)+M_delta(2,3);

  arma::Mat<double> src_pcl_out = MultiBodyICP::GetProjectedModelPCL(mod,vox,fcg,M_update,seg_ind,int_val);

  return make_tuple(src_pcl_out,M_update,score);
}

/*
tuple<arma::Mat<double>,arma::Mat<double>,double> MultiBodyICP::ICPAlgorithm(model &mod, vox_grid &vox, FocalGrid &fcg, arma::Mat<double> &dest_pcl, arma::Mat<double> M_init, int seg_ind, int int_val) {

  double score = 1.0;
  arma::Mat<double> src_pcl_i;
  arma::Mat<double> src_pcl_out;
  arma::Mat<double> M_update;
  M_update = M_init;
  arma::Mat<double> M_out;
  M_out.eye(4,4);

  int max_iter = 30;
  double prev_score = 1e6;
  int i = 0;

  //for (int i=0; i<20; i++) {
  while ((i < max_iter) && ((score/prev_score)<1.05)) {

    cout << "iteration" << endl;
    cout << i << endl;

    src_pcl_i = MultiBodyICP::GetProjectedModelPCL(mod,vox,fcg,M_update,seg_ind,int_val);

    tuple<arma::Mat<double>,double> pair_update = MultiBodyICP::PairAlign(dest_pcl, src_pcl_i);

    //score = get<1>(pair_update);
    score = get<1>(pair_update);
    cout << "score" << endl;
    cout << score << endl;

    arma::Mat<double> M_delta = get<0>(pair_update);

    M_update.submat(0,0,2,2) = M_update.submat(0,0,2,2)*M_delta.submat(0,0,2,2);
    M_update(0,3) = M_update(0,3)+M_delta(0,3);
    M_update(1,3) = M_update(1,3)+M_delta(1,3);
    M_update(2,3) = M_update(2,3)+M_delta(2,3);

    if ((score/prev_score)<1.05) {

      M_out = M_update;
      src_pcl_out = src_pcl_i;
      prev_score = score;

    }
    else {
      i = max_iter;
    }

    i++;
  }

  return make_tuple(src_pcl_out,M_out,score);
}
*/

double MultiBodyICP::ProjectedModelScore(model &mod, vox_grid &vox, FocalGrid &fcg, arma::Mat<double> &dest_pcl, arma::Mat<double> M_init, int seg_ind, int int_val) {

  pcl::PointCloud<pcl::PointXYZ> seg_pcl = MultiBodyICP::GetModelSRCPointCloud(mod, M_init, seg_ind);

  vector<tuple<double,double,double,int>> seg_pcl_vec = MultiBodyICP::Convert_PCL_XYZ_2_Vec(seg_pcl, int_val);

  vector<tuple<double,double,double,int>> dest_pcl_vec = MultiBodyICP::Convert_Mat_2_PCL_Vec(dest_pcl, int_val);

  vector<arma::Col<int>> proj_mdl_frames = fcg.ProjectCloud2Image(seg_pcl_vec);

  vector<arma::Col<int>> proj_dest_frames = fcg.ProjectCloud2Image(dest_pcl_vec);

  int N_views = proj_mdl_frames.size();

  double score = 0.0;

  for (int i=0; i<N_views; i++) {

    arma::Col<int> frame_diff = arma::abs(proj_dest_frames[i]-proj_mdl_frames[i]);

    score += (1.0*arma::sum(frame_diff))/(1.0*arma::sum(proj_dest_frames[i]));

  }

  return score;

}

arma::Mat<double> MultiBodyICP::GetProjectedModelPCL(model &mod, vox_grid &vox, FocalGrid &fcg, arma::Mat<double> M_init, int seg_ind, int int_val) {

  // Project model pointcloud to 2d images:

  pcl::PointCloud<pcl::PointXYZ> seg_pcl = MultiBodyICP::GetModelSRCPointCloud(mod, M_init, seg_ind);

  vector<tuple<double,double,double,int>> seg_pcl_vec = MultiBodyICP::Convert_PCL_XYZ_2_Vec(seg_pcl, int_val);

  vector<arma::Col<int>> proj_mdl_frames = fcg.ProjectCloud2Image(seg_pcl_vec);

  // Project 2d images to pointcloud:

  vector<tuple<int,double,double,double,double,double,double>> proj_pcl = fcg.ProjectImage2Cloud(proj_mdl_frames, vox);

  arma::Mat<double> pcl_out = MultiBodyICP::Convert_PCL_Vec_2_Mat(proj_pcl);

  return pcl_out;
}

pcl::PolygonMesh MultiBodyICP::TransformMesh(pcl::PolygonMesh mesh_in, arma::Mat<double> transform_mat) {

  pcl::PCLHeader header = mesh_in.header;

  pcl::PCLPointCloud2 cloud2_in = mesh_in.cloud;

  vector<pcl::Vertices> polygons = mesh_in.polygons;

  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in (new pcl::PointCloud<pcl::PointXYZ>);
  pcl::fromPCLPointCloud2(cloud2_in, *cloud_in);

  // Construct the transformation matrix

  Eigen::Matrix4f trans_mat = Eigen::Matrix4f::Identity();

  trans_mat(0,0) = transform_mat(0,0);
  trans_mat(0,1) = transform_mat(0,1);
  trans_mat(0,2) = transform_mat(0,2);
  trans_mat(0,3) = transform_mat(0,3);
  trans_mat(1,0) = transform_mat(1,0);
  trans_mat(1,1) = transform_mat(1,1);
  trans_mat(1,2) = transform_mat(1,2);
  trans_mat(1,3) = transform_mat(1,3);
  trans_mat(2,0) = transform_mat(2,0);
  trans_mat(2,1) = transform_mat(2,1);
  trans_mat(2,2) = transform_mat(2,2);
  trans_mat(2,3) = transform_mat(2,3);

  // Transform the pointcloud

  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_out (new pcl::PointCloud<pcl::PointXYZ>);

  pcl::transformPointCloud(*cloud_in, *cloud_out, trans_mat);

  pcl::PCLPointCloud2::Ptr cloud2_out (new pcl::PCLPointCloud2);

  pcl::toPCLPointCloud2(*cloud_out,*cloud2_out);

  pcl::PolygonMesh mesh_out;

  mesh_out.header = header;
  mesh_out.polygons = polygons;
  mesh_out.cloud = *cloud2_out;

  return mesh_out;

}

pcl::PointCloud<pcl::PointXYZ> MultiBodyICP::GetPointCloudFromMesh(pcl::PolygonMesh mesh_in) {

  pcl::PointCloud<pcl::PointXYZ> cloud_out;

  pcl::PCLPointCloud2 cloud2_in = mesh_in.cloud;

  pcl::fromPCLPointCloud2(cloud2_in, cloud_out);

  return cloud_out;
}

arma::Mat<double> MultiBodyICP::MultiplyMat(arma::Mat<double> M_a, arma::Mat<double> M_b) {

  arma::Mat<double> mult_mat;

  mult_mat.zeros(4,4);

  mult_mat.submat(0,0,2,2) = M_a.submat(0,0,2,2)*M_b.submat(0,0,2,2);

  mult_mat.submat(0,0,2,0) = M_a.submat(0,0,2,2)*M_b.submat(0,0,2,0)+M_a.submat(0,0,2,0);

  //mult_mat(0,3) = M_a(0,3)+M_b(0,3);
  //mult_mat(1,3) = M_a(1,3)+M_b(1,3);
  //mult_mat(2,3) = M_a(2,3)+M_b(2,3);

  mult_mat(3,3) = 1.0;

  return mult_mat;

}

arma::Mat<double> MultiBodyICP::TransformMat(arma::Col<double> q_in, double scale) {

  arma::Mat<double> start_mat;

  start_mat.zeros(4,4);

  start_mat(0,0) = (2.0*pow(q_in(0),2)-1.0+2.0*pow(q_in(1),2));
  start_mat(0,1) = (2.0*q_in(1)*q_in(2)+2.0*q_in(0)*q_in(3));
  start_mat(0,2) = (2.0*q_in(1)*q_in(3)-2.0*q_in(0)*q_in(2));
  start_mat(0,3) = scale*(q_in(4));
  start_mat(1,0) = (2.0*q_in(1)*q_in(2)-2.0*q_in(0)*q_in(3));
  start_mat(1,1) = (2.0*pow(q_in(0),2)-1.0+2.0*pow(q_in(2),2));
  start_mat(1,2) = (2.0*q_in(2)*q_in(3)+2.0*q_in(0)*q_in(1));
  start_mat(1,3) = scale*(q_in(5));
  start_mat(2,0) = (2.0*q_in(1)*q_in(3)+2.0*q_in(0)*q_in(2));
  start_mat(2,1) = (2.0*q_in(2)*q_in(3)-2.0*q_in(0)*q_in(1));
  start_mat(2,2) = (2.0*pow(q_in(0),2)-1.0+2.0*pow(q_in(3),2));
  start_mat(2,3) = scale*(q_in(6));
  start_mat(3,3) = 1.0;

  return start_mat;

}

arma::Mat<double> MultiBodyICP::Convert_PCL_Vec_2_Mat(vector<tuple<int,double,double,double,double,double,double>> pcl_in) {

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

vector<tuple<double,double,double,int>> MultiBodyICP::Convert_PCL_XYZ_2_Vec(pcl::PointCloud<pcl::PointXYZ> pcl_in, int ind) {

  vector<tuple<double,double,double,int>> pcl_out;

  for (int i=0; i<pcl_in.points.size(); i++) {

    pcl_out.push_back(make_tuple(pcl_in.points[i].x,pcl_in.points[i].y,pcl_in.points[i].z,ind));

  }

  return pcl_out;
}

vector<tuple<double,double,double,int>> MultiBodyICP::Convert_Mat_2_PCL_Vec(arma::Mat<double> &pcl_in, int ind) {

  vector<tuple<double,double,double,int>> pcl_out;

  for (int i=0; i<pcl_in.n_cols; i++) {

    pcl_out.push_back(make_tuple(pcl_in(1,i),pcl_in(2,i),pcl_in(3,i),ind));

  }

  return pcl_out;
}

pcl::PointCloud<pcl::PointXYZ> MultiBodyICP::Convert_Mat_2_PCL_XYZ(arma::Mat<double> pcl_mat) {

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

arma::Mat<double> MultiBodyICP::GetM(arma::Col<double> state_in) {

  arma::Col<double> q_vec = {state_in(0),state_in(1),state_in(2),state_in(3)};
  q_vec = q_vec/sqrt(pow(q_vec(0),2)+pow(q_vec(1),2)+pow(q_vec(2),2)+pow(q_vec(3),2));

  double q0 = q_vec(0);
  double q1 = q_vec(1);
  double q2 = q_vec(2);
  double q3 = q_vec(3);
  double tx = state_in(4);
  double ty = state_in(5);
  double tz = state_in(6);

  arma::Mat<double> M = {{2.0*pow(q0,2.0)-1.0+2.0*pow(q1,2.0), 2.0*q1*q2-2.0*q0*q3, 2.0*q1*q3+2.0*q0*q2, tx},
    {2.0*q1*q2+2.0*q0*q3, 2.0*pow(q0,2.0)-1.0+2.0*pow(q2,2.0), 2.0*q2*q3-2.0*q0*q1, ty},
    {2.0*q1*q3-2.0*q0*q2, 2.0*q2*q3+2.0*q0*q1, 2.0*pow(q0,2.0)-1.0+2.0*pow(q3,2.0), tz},
    {0.0, 0.0, 0.0, 1.0}};

  return M;

}

PointCloudT MultiBodyICP::TransferPointXYZ2PointNT(pcl::PointCloud<pcl::PointXYZ> pcl_in) {

  PointCloudT pcl_out;

  for (int i=0; i<pcl_in.points.size(); i++) {

    PointNT point_out;

    point_out.x = pcl_in.points[i].x;
    point_out.y = pcl_in.points[i].y;
    point_out.z = pcl_in.points[i].z;
    point_out.normal_x = 1.0;
    point_out.normal_y = 0.0;
    point_out.normal_z = 0.0;

    pcl_out.push_back(point_out);

  }
  return pcl_out;
}

arma::Mat<double> MultiBodyICP::TransferPointCloudT2Mat(PointCloudT pcl_in, int int_val) {

  int N_points = pcl_in.points.size();

  arma::Mat<double> pcl_out(7,N_points);

  for (int i=0; i<N_points; i++) {
    pcl_out(0,i) = (double) int_val;
    pcl_out(1,i) = pcl_in.points[i].x;
    pcl_out(2,i) = pcl_in.points[i].y;
    pcl_out(3,i) = pcl_in.points[i].z;
    pcl_out(4,i) = 1.0;
    pcl_out(5,i) = 0.0;
    pcl_out(6,i) = 0.0;
  }

  return pcl_out;
}


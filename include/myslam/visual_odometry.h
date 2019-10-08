#ifndef VISUALODOMETRY_H
#define VISUALODOMETRY_H

#include "myslam/common_include.h"
#include "myslam/map.h"
#include "myslam/localbundleadjustment.h"
#include <pangolin/pangolin.h>
#include <opencv2/features2d/features2d.hpp>


namespace myslam {
    class VisualOdometry{
    public:
        //functions:
        VisualOdometry();
        ~VisualOdometry();
        void addFrames(const Frame::Ptr lframe, const Frame::Ptr rframe);
        // important data
        typedef shared_ptr<VisualOdometry> Ptr;
        enum VOState {
            INITIALIZING=-1,
            OK=0,
            LOST=1
        };
        cv::Ptr<cv::ORB> orb_;
        cv::Ptr<cv::cuda::ORB> orb_cuda_;
        cv::BFMatcher matcher_;
        cv::Ptr<cv::cuda::DescriptorMatcher> matcher_cuda_;
        vector<Eigen::Vector3d> p3d_;
        vector<Eigen::Vector2d> p2d_;
        vector<cv::KeyPoint> lkpoints_cur_, rkpoints_cur_, lkpoints_ref_, rkpoints_ref_;
        Mat ldescriptors_cur_, rdescriptors_cur_, ldescriptors_ref_, rdescriptors_ref_;
        cv::cuda::GpuMat ldescriptors_cur_cuda_, rdescriptors_cur_cuda_, ldescriptors_ref_cuda_, rdescriptors_ref_cuda_;
        unordered_map<int, int> matches_stereo_cur_, matches_stereo_ref_;
        vector<cv::DMatch> matches_seq_;
        Frame::Ptr lcur_, rcur_, lref_, rref_;
        vector<Eigen::Vector3d> translation_;
        vector<Eigen::Matrix3d> rotation_;

        double para_r_[3];
        double para_t_[3];

        Eigen::Matrix3d R_;
        Eigen::Map<Eigen::Vector3d> t_;

        float mViewpointX, mViewpointY, mViewpointZ, mViewpointF;
        void showPointCloud();
        void getOpenGLCameraMatrix(pangolin::OpenGlMatrix&) const;
    protected:
        void extractKeyPointsComputeDescriptors();
        void stereoFeatureMatching();
        void seqFeatureMatching();
        inline bool checkSeqMatching() const;
        void poseEstimationPnP();
        void localBundleAdjustment();
        Eigen::Matrix3d angleAxisToRotationMatrix(double rvec[3]) const;

        void fileAndConsoleIO() const;
        int num_of_features_;
        float scale_factor_; // this is used in ORB pyramid
        int level_pyramid_;
        double stereo_distance_threshold_;// the max distance to be regarded as inlier
        double seq_distance_threshold_;
        double reprojection_error_threshold_;
        int RANSAC_iterations_;
        double keyframe_min_rotation_;
        double keyframe_min_translation_;
        VOState state_;
    };
}

#endif // VISUALODOMETRY_H

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <algorithm>
#include <random>
#include <eigen3/Eigen/Core>
#include "myslam/visual_odometry.h"
#include "myslam/localbundleadjustment.h"
#include <pangolin/pangolin.h>
#include <ceres/rotation.h>
#include <ceres/ceres.h>
#include <fstream>
#include <iomanip>

namespace myslam{
    VisualOdometry::VisualOdometry() :
    t_(para_t_),
    matcher_(cv::NORM_HAMMING),
    matcher_cuda_ (cv::cuda::DescriptorMatcher::createBFMatcher(cv::NORM_HAMMING)){
        num_of_features_ = 1000;
        scale_factor_ = 1.2;
        level_pyramid_ = 4;
        stereo_distance_threshold_ = 50;
        seq_distance_threshold_ = 40;
        reprojection_error_threshold_ = 4;
        state_ = INITIALIZING;
        RANSAC_iterations_ = 20;
        keyframe_min_rotation_ = 0.01;
        keyframe_min_translation_ = 0.5;
        orb_ = cv::ORB::create(num_of_features_, scale_factor_, level_pyramid_);
        orb_cuda_ = cv::cuda::ORB::create(num_of_features_, scale_factor_, level_pyramid_);
        lcur_ = nullptr; rcur_ = nullptr; lref_ = nullptr; rref_ = nullptr;
        Eigen::Matrix3d R_tmp = Eigen::Matrix3d::Identity();
        Eigen::Vector3d t_tmp = Eigen::Vector3d::Zero();
        mViewpointX = 0.;
        mViewpointY = -100.;
        mViewpointZ = -0.1;
        mViewpointF = 2000;
        para_r_[0] = 0; para_r_[1] = 0; para_r_[2] = 0;
        para_t_[0] = 0; para_t_[1] = 0; para_t_[2] = 0;
    }

    VisualOdometry::~VisualOdometry(){}

    void VisualOdometry::extractKeyPointsComputeDescriptors(){
        if (cv::cuda::getCudaEnabledDeviceCount()) {
            orb_cuda_ -> setBlurForDescriptor(true);
            cv::cuda::GpuMat imgGray1(lcur_ -> color_);
            cv::cuda::GpuMat mask1;
            vector<cv::KeyPoint> keys1;
            orb_cuda_ -> detectAndCompute(imgGray1, mask1, lkpoints_cur_, ldescriptors_cur_cuda_);
            ldescriptors_cur_cuda_.download(ldescriptors_cur_);

            cv::cuda::GpuMat imgGray2(rcur_ -> color_);
            cv::cuda::GpuMat keys2;
            cv::cuda::GpuMat mask2;
            orb_cuda_ -> detectAndCompute(imgGray2, mask2, rkpoints_cur_, rdescriptors_cur_cuda_);
            rdescriptors_cur_cuda_.download(rdescriptors_cur_);
        } else {
            orb_->detect ( lcur_->color_, lkpoints_cur_);
            orb_->detect ( rcur_->color_, rkpoints_cur_);
            orb_-> compute(lcur_->color_, lkpoints_cur_, ldescriptors_cur_);
            orb_-> compute(rcur_->color_, rkpoints_cur_, rdescriptors_cur_);
        }

    }

    void VisualOdometry::stereoFeatureMatching(){
        vector<cv::DMatch> matches;
        if (cv::cuda::getCudaEnabledDeviceCount()) {
            matcher_cuda_ -> match(ldescriptors_cur_cuda_, rdescriptors_cur_cuda_, matches);
        } else {
            matcher_.match (ldescriptors_cur_, rdescriptors_cur_, matches);
        }
        // clear the matching result of last time
        sort(matches.begin(), matches.end(), [](const cv::DMatch& m1, const cv::DMatch& m2){
           return m1.distance < m2.distance;
        });
        for (const auto& m : matches) {
            if (m.distance > stereo_distance_threshold_) {
                return;
            }
            bool valid = true;
            for (const auto& nei : matches_stereo_cur_) {
                auto delta_pt = lkpoints_cur_[m.queryIdx].pt - lkpoints_cur_[nei.first].pt;
                if (delta_pt.x * delta_pt.x + delta_pt.y * delta_pt.y < 9) {
                    valid = false;
                    break;
                }
            }
            if (valid) {
                matches_stereo_cur_.insert({m.queryIdx, m.trainIdx});
            }
        }
    }

    void VisualOdometry::seqFeatureMatching(){
        vector<cv::DMatch> matches;
        if (cv::cuda::getCudaEnabledDeviceCount()) {
            matcher_cuda_ -> match(ldescriptors_cur_cuda_, ldescriptors_ref_cuda_, matches);
        } else {
            matcher_.match(ldescriptors_cur_, ldescriptors_ref_, matches);
        }
        // use matches_seq_ matches_cur_hash_, matches_ref_hash
        matches_seq_.reserve(matches.size());
        for ( cv::DMatch& m : matches ) {
            // if the seq matching of features exist in both current and reference frame
            if (m.distance < seq_distance_threshold_ &&
                 matches_stereo_cur_.count(m.queryIdx)){
                int lkindex_cur = m.queryIdx;
                int rkindex_cur = matches_stereo_cur_[lkindex_cur];
                int lkindex_ref = m.trainIdx;

                //simply check the disparity first
                float disparity_cur = lkpoints_cur_[lkindex_cur].pt.x
                                    - rkpoints_cur_[rkindex_cur].pt.x;

                float depth_cur = lcur_->camera_->f_ / disparity_cur * lcur_->camera_->baseLine_;
                Eigen::Vector2d p2d(lkpoints_ref_[lkindex_ref].pt.x, lkpoints_ref_[lkindex_ref].pt.y);
                Eigen::Vector3d p3d_cur = lcur_->camera_->pixel2camera(
                            Eigen::Vector2d(lkpoints_cur_[lkindex_cur].pt.x, lkpoints_cur_[lkindex_cur].pt.y), depth_cur);
                //for too big change in x, y or z coordiante, throw them away
                if (matches_stereo_ref_.count(m.trainIdx)) {
                    int rkindex_ref = matches_stereo_ref_[lkindex_ref];
                    float disparity_ref = lkpoints_ref_[lkindex_ref].pt.x
                                          - rkpoints_ref_[rkindex_ref].pt.x;
                    float depth_ref = lcur_->camera_->f_ / disparity_ref * lcur_->camera_->baseLine_;
                    Eigen::Vector3d p3d_ref = lcur_->camera_->pixel2camera(
                            p2d, depth_ref);
                    if (std::fabs(p3d_ref[1] - p3d_cur[1]) > 0.5 ||
                        std::fabs(p3d_ref[2] - p3d_cur[2]) > 5 ||
                        std::fabs(p3d_ref[0] - p3d_cur[0]) > 2){
                        continue;
                    }
                }
                // add match of sequence
                matches_seq_.push_back(m);
                // add current frame p3d
                p3d_.push_back(p3d_cur);
                // add current frame p2d
                p2d_.push_back(p2d);
            }
        }
    }

    void VisualOdometry::poseEstimationPnP(){
        vector<cv::Point3f> pts3d;
        vector<cv::Point2f> pts2d;
        int npoints = p3d_.size();
        for (int i = 0; i < npoints; ++i){
            Eigen::Vector3d p3d = p3d_[i];
            Eigen::Vector2d p2d = p2d_[i];
            pts3d.emplace_back(p3d[0], p3d[1], p3d[2]);
            pts2d.emplace_back(p2d[0], p2d[1]);
        }
        //camera intrinsics
        Mat K = (cv::Mat_<double>(3,3)<<
                lref_->camera_->f_, 0, lref_->camera_->cx_,
                0, lref_->camera_->f_, lref_->camera_->cy_,
                0,0,1);
        //Do p3p
        Mat rvec, tvec;
        vector<int> inliers;
        cv::solvePnPRansac(pts3d, pts2d, K, Mat(), rvec, tvec, false,
                RANSAC_iterations_, reprojection_error_threshold_, 0.99, inliers, CV_EPNP);
        Mat Rmat;
        cv::Rodrigues(rvec, Rmat);
        Eigen::Matrix3d R;
        R.row(0) << Rmat.at<double>(0, 0), Rmat.at<double>(0, 1), Rmat.at<double>(0, 2);
        R.row(1) << Rmat.at<double>(1, 0), Rmat.at<double>(1, 1), Rmat.at<double>(1, 2);
        R.row(2) << Rmat.at<double>(2, 0), Rmat.at<double>(2, 1), Rmat.at<double>(2, 2);
        Eigen::Vector3d t(tvec.at<double>(0,0), tvec.at<double>(1,0), tvec.at<double>(2,0));
        //---------------------------------calculate reprojection error, count inliers--------------------//
        vector<Eigen::Vector3d> p3d_tmp;
        vector<Eigen::Vector2d> p2d_tmp;
        for (int i = 0; i < inliers.size(); ++i){
            p3d_tmp.push_back(p3d_[inliers[i]]);
            p2d_tmp.push_back(p2d_[inliers[i]]);
        }
        swap(p3d_, p3d_tmp); swap(p2d_, p2d_tmp);
        //update postion
        translation_.push_back(rotation_.back() * t_ + translation_.back());
        rotation_.push_back(R_ * rotation_.back());
        //output
        cout << "all matches: " << npoints << endl;
        cout << "inliers: " << p3d_.size() << endl;
        cout << "rotation: " << endl << R_ << endl;
        cout << "translation: " << endl << t_ << endl;
    }


    void VisualOdometry::localBundleAdjustment(){
        ceres::Problem problem;
        ceres::HuberLoss* loss_fun(new ceres::HuberLoss(0.1));
        //construct variables
        double scales[500];
        for (int i = 0; i < p2d_.size(); ++i){
            //-----------------------------------------simple ba ----------------------------//
            auto* costFun = new ceres::AutoDiffCostFunction<SimpleBundleAdjustmentCost, 2, 3, 3>
                    (new SimpleBundleAdjustmentCost(p3d_[i], p2d_[i]));
            problem.AddResidualBlock(costFun, loss_fun, para_r_, para_t_);
        }
        ceres::Solver::Options options;
        options.minimizer_progress_to_stdout = false;
        options.num_threads = 8;
        options.max_num_iterations = 5;
        options.function_tolerance = 1e-4;
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);

        R_ = angleAxisToRotationMatrix(para_r_);
        translation_.pop_back();
        rotation_.pop_back();
        translation_.push_back(rotation_.back() * t_ + translation_.back());
        rotation_.push_back(R_ * rotation_.back());
    }

    Eigen::Matrix3d VisualOdometry::angleAxisToRotationMatrix(double rvec[3]) const {
        cv::Mat vmat(cv::Size(1, 3), CV_64FC1);
        vmat.at<double>(0, 0) = rvec[0];
        vmat.at<double>(0, 1) = rvec[1];
        vmat.at<double>(0, 2) = rvec[2];
        cv::Mat Rmat;
        cv::Rodrigues(vmat, Rmat);
        Eigen::Matrix3d R;
        R.row(0) << Rmat.at<double>(0, 0), Rmat.at<double>(0, 1), Rmat.at<double>(0, 2);
        R.row(1) << Rmat.at<double>(1, 0), Rmat.at<double>(1, 1), Rmat.at<double>(1, 2);
        R.row(2) << Rmat.at<double>(2, 0), Rmat.at<double>(2, 1), Rmat.at<double>(2, 2);
        return R;
    }

    void VisualOdometry::addFrames(const Frame::Ptr lframe, const Frame::Ptr rframe){
        switch (state_) {
            case INITIALIZING:{
                state_ = OK;
                lref_ = lframe;
                rref_ = rframe;
                lcur_ = lframe;
                rcur_ = rframe;
                translation_.push_back(Eigen::Vector3d::Zero());
                rotation_.push_back(Eigen::Matrix3d::Identity());
                extractKeyPointsComputeDescriptors();
                stereoFeatureMatching();
//                fileAndConsoleIO();
                break;
            }
            case OK:{
                lref_ = lcur_;
                rref_ = rcur_;
                lcur_ = lframe;
                rcur_ = rframe;
                lkpoints_ref_ = lkpoints_cur_;
                ldescriptors_ref_ = ldescriptors_cur_.clone();
                ldescriptors_ref_cuda_ = ldescriptors_cur_cuda_.clone();
                rkpoints_ref_ = rkpoints_cur_;
                rdescriptors_ref_ = rdescriptors_cur_.clone();
                rdescriptors_ref_cuda_ = rdescriptors_cur_cuda_.clone();
                matches_stereo_ref_ = matches_stereo_cur_;
                matches_stereo_cur_.clear();
                matches_seq_.clear();
                p3d_.clear();
                p2d_.clear();
                extractKeyPointsComputeDescriptors();
                stereoFeatureMatching();
                seqFeatureMatching();
                poseEstimationPnP();
                localBundleAdjustment();
//                fileAndConsoleIO();
                //-----------------------visualization-------------------------//
//                Mat pose;
//                drawMatches(lcur_->color_,lkpoints_cur_,lref_->color_,lkpoints_ref_,matches_seq_,pose);
//                cv::resize(pose, pose, cv::Size(1800, 400));
//                imshow("match seq", pose);
//                cv::imshow("image", lcur_ -> color_);
//                cv::waitKey(1);
                break;
            }
        }
    }

    void VisualOdometry::getOpenGLCameraMatrix(pangolin::OpenGlMatrix& M) const {
        if (translation_.empty()) {
            M.SetIdentity();
            return;
        }
        static Eigen::Matrix3d correction_matrix;
        correction_matrix << 1, 0, 0,
                           0, 1, 0,
                           0, 0, 1;
        Eigen::Matrix3d camera_matrix = correction_matrix * rotation_.back();
        M.m[0] = camera_matrix(0,0);
        M.m[1] = camera_matrix(1,0);
        M.m[2] = camera_matrix(2,0);
        M.m[3] = 0.;
        M.m[4] = camera_matrix(0,1);
        M.m[5] = camera_matrix(1,1);
        M.m[6] = camera_matrix(2,1);
        M.m[7] = 0.;
        M.m[8] = camera_matrix(0,2);
        M.m[9] = camera_matrix(1,2);
        M.m[10] = camera_matrix(2,2);
        M.m[11] = 0.;
        M.m[12] = translation_.back()[0];
        M.m[13] = translation_.back()[1];
        M.m[14] = translation_.back()[2];
        M.m[15] = 1.;
    }

    void VisualOdometry::showPointCloud(){
        if (translation_.empty()) {
            cout << "no tracking found" << endl;
            return;
        }
        //start drawing points
        glPointSize(4);
        glColor3f(255,0,0);
        glBegin(GL_POINTS);
        for (auto point : translation_){
            glVertex3f(point[0],point[1],point[2]);
        }
        glEnd();
        //start drawing lines
        glBegin(GL_LINES);
        glColor3f(0., 0., 0.);
        for (int i = 0; i < translation_.size() - 1; ++i){
            auto point = translation_[i];
            glVertex3f(point[0], point[1], point[2]);
            point = translation_[i+1];
            glVertex3f(point[0], point[1], point[2]);
        }
        glEnd();
    }

    void VisualOdometry::fileAndConsoleIO() const{
        //save tracking data to txt file
        ofstream of;
        of.open("/home/ubuntu/Desktop/project_v1/evaluation/02/newres.txt", ios_base::app);
        if (!of){
            cout << "wrong trajectory file path" << endl;
            of.close();
        }
        else {
            Eigen::Matrix3d tmp_rot = rotation_.back();
            of << fixed << setprecision(9)
               << tmp_rot(0,0) << ' ' << tmp_rot(0,1) << ' ' << tmp_rot(0,2)<< ' ' << translation_.back()(0) << ' '
               << tmp_rot(1,0) << ' ' << tmp_rot(1,1) << ' ' << tmp_rot(1,2)<< ' ' << translation_.back()(1) << ' '
               << tmp_rot(2,0) << ' ' << tmp_rot(2,1) << ' ' << tmp_rot(2,2)<< ' ' << translation_.back()(2) << '\n';
            of.close();
        }    
    }
}



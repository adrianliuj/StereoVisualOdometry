#include<iostream>
#include<algorithm>
#include<fstream>
#include<iomanip>
#include<chrono>
#include<vector>
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<unistd.h>
#include "myslam/visual_odometry.h"
#include <pangolin/pangolin.h>
using namespace std;
using namespace cv;
void LoadImages(const string &strPathToSequence, vector<string> &vstrImageLeft,
                vector<string> &vstrImageRight);

int main(int argc, char ** argv){
    if(argc != 2){
        cerr << "wrong sequence path" << endl;
        return 1;
    }
    // Retrieve paths to images
    vector<string> vstrImageLeft;
    vector<string> vstrImageRight;
    //read left path, right path, timestamps
    LoadImages(string(argv[1]), vstrImageLeft, vstrImageRight);
    const int nImages = vstrImageRight.size();
    vector<float> vTimesTrack;
    //ceres initialization
    google::InitGoogleLogging(argv[0]);
    // some initialization
    cv::Mat imLeft, imRight;
    myslam::VisualOdometry vo;
    myslam::Camera::Ptr camera(new myslam::Camera());
    // create window for visualization
    pangolin::CreateWindowAndBind("Trajectory Viewer", 900, 628);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // Define Camera Render Object (for view / scene browsing)
    pangolin::OpenGlRenderState s_cam(
                pangolin::ProjectionMatrix(900,628,vo.mViewpointF,vo.mViewpointF,450,314,0.1,1000),
                pangolin::ModelViewLookAt(vo.mViewpointX,vo.mViewpointY,vo.mViewpointZ, 0,0,0,0.0,-1.0, 0.0)
                );

    pangolin::View &d_cam = pangolin::CreateDisplay()
            .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -900.0f / 628.0f)
            .SetHandler(new pangolin::Handler3D(s_cam));
    pangolin::OpenGlMatrix Twc;
    Twc.SetIdentity();
    // Main loop
    for(int ni=0; ni<nImages; ni++){
        // Read left and right images from file
        imLeft = cv::imread(vstrImageLeft[ni],CV_LOAD_IMAGE_UNCHANGED);
        imRight = cv::imread(vstrImageRight[ni],CV_LOAD_IMAGE_UNCHANGED);
        if (imLeft.empty()) {
            break;
        }
        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
        myslam::Frame::Ptr lframe = myslam::Frame::createFrame();
        myslam::Frame::Ptr rframe = myslam::Frame::createFrame();
        lframe->camera_ = camera;
        lframe->color_ = imLeft;
        rframe->color_ = imRight;
        vo.addFrames(lframe, rframe);
        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
        double ttrack= std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
        cout << "Frame ID: " << ni << endl;
        cout << "tracking time: " << ttrack * 1000 << " ms" << endl;
        vTimesTrack.push_back(ttrack);
        //start visualization
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        vo.getOpenGLCameraMatrix(Twc);
        s_cam.Follow(Twc);
        d_cam.Activate(s_cam);
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
        vo.showPointCloud();
        pangolin::FinishFrame();
        usleep(1);
    }
    // Tracking time statistics
    sort(vTimesTrack.begin(),vTimesTrack.end());
    float totaltime = 0;
    for(int ni=0; ni<nImages; ni++){
        totaltime+=vTimesTrack[ni];
    }
    return 0;
}

void LoadImages(const string &strPathToSequence, vector<string> &vstrImageLeft,
                vector<string> &vstrImageRight){

    string strPrefixLeft = strPathToSequence + "/image_0/";
    string strPrefixRight = strPathToSequence + "/image_1/";

    const int nTimes = 10000;
    vstrImageLeft.resize(nTimes);
    vstrImageRight.resize(nTimes);

    for(int i=0; i<nTimes; i++){
        stringstream ss;
        ss << setfill('0') << setw(6) << i;
        vstrImageLeft[i] = strPrefixLeft + ss.str() + ".png";
        vstrImageRight[i] = strPrefixRight + ss.str() + ".png";
    }
    return;
}

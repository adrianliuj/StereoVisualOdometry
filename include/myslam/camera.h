#ifndef CAMERA_H
#define CAMERA_H

#include "myslam/common_include.h"

namespace myslam
{

// Pinhole RGBD camera model
class Camera
{
public:
    typedef std::shared_ptr<Camera> Ptr;
    float   f_, cx_, cy_;  // Camera intrinsics
    float  baseLine_; // baseline distance
    Eigen::Matrix3d intrinsics_, inv_intrinsics_;
    Camera(){
        f_ = 718.856;
        cx_ = 607.19;
        cy_ = 185.2;
        baseLine_ = 0.53716;
        intrinsics_ << f_, 0., cx_, 0., f_, cy_, 0., 0., 1.;
        inv_intrinsics_ = intrinsics_.inverse();
    };

    // coordinate transform: world, camera, pixel
    Vector3d world2camera( const Vector3d& p_w, const SE3& T_c_w );
    Vector3d camera2world( const Vector3d& p_c, const SE3& T_c_w );
    Vector2d camera2pixel( const Vector3d& p_c );
    Vector3d pixel2camera( const Vector2d& p_p, double depth=1 ); 
    Vector3d pixel2world ( const Vector2d& p_p, const SE3& T_c_w, double depth=1 );
    Vector2d world2pixel ( const Vector3d& p_w, const SE3& T_c_w );

};

}
#endif // CAMERA_H

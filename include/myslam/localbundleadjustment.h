#ifndef LOCALBUNDLEADJUSTMENT_H
#define LOCALBUNDLEADJUSTMENT_H
#include "myslam/common_include.h"
namespace myslam{
    class BundleAdjustmentCost{
    public:
        BundleAdjustmentCost(Vector3d& p3d, Vector2d& p2d):
            p3d_(p3d), p2d_(p2d){}
        bool operator()(const double* const camera,
                        const double* const scale,
                        double* residuals) const{
            double f_ = 718.856;
            double cx_ = 607.19;
            double cy_ = 185.2;
            Sophus::SO3 so3(camera[3], camera[4], camera[5]);
            Vector3d p3d_tmp = *scale * p3d_;
            p3d_tmp = so3.matrix() * p3d_tmp;
            p3d_tmp[0] += camera[0]; p3d_tmp[1] += camera[1]; p3d_tmp[2] += camera[2];
            p3d_tmp /= p3d_tmp[2];
            residuals[0] = p3d_tmp[0] * f_ + cx_ - p2d_[0];
            residuals[1] = p3d_tmp[1] * f_ + cy_ - p2d_[1];
            return true;
        }
    private:
        Vector3d p3d_;
        Vector2d p2d_;
    };
    class SimpleBundleAdjustmentCost{
    public:
        SimpleBundleAdjustmentCost(Vector3d& p3d, Vector2d& p2d):
            p3d_(p3d), p2d_(p2d){}
        template <typename T>
        bool operator()(const T* const para_r,
                        const T* const para_t,
                        T* residuals) const{
            T f_(718.856);
            T cx_(607.19);
            T cy_ (185.2);

            T p3d_tmp[3];
            p3d_tmp[0] = T(p3d_[0]); p3d_tmp[1] = T(p3d_[1]); p3d_tmp[2] = T(p3d_[2]);
            T p[3];
            ceres::AngleAxisRotatePoint(para_r, p3d_tmp, p);
            p[0] += para_t[0]; p[1] += para_t[1]; p[2] += para_t[2];
            p[0] /= p[2];
            p[1] /= p[2];

            residuals[0] = p[0] * f_ + cx_ - T(p2d_[0]);
            residuals[1] = p[1] * f_ + cy_ - T(p2d_[1]);
            return true;
        }
    private:
        Vector3d p3d_;
        Vector2d p2d_;
    };
}
#endif // LOCALBUNDLEADJUSTMENT_H

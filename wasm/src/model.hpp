#pragma once
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#include <opencv2/core/mat.hpp>
#include <string>
#include <vector>

struct BoundingBox
{
    std::string cname;
    float confidence;
    int x1, y1, x2, y2;
};

class Model
{
    public:
        Model(std::string model_path);
        std::vector<BoundingBox> run(cv::Mat input);
    private:
        Ort::Session m_session;
        float* pre_process(cv::Mat input);
        std::vector<BoundingBox> post_process(Ort::Value output);
};

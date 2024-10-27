#include "model.hpp"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

Model::Model(std::string model_path):
{
    // load model vaiables and initialize model
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");
    // create session options
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    m_session = Ort::Session(env, model_path.c_str(), session_options);
    m_run_options = Ort::RunOptions{nullptr};
}

void Model::run(cv::Mat input)
    // run inferenece on model input
    std::cout << "Running model" << std::endl;
    // create input tensor
    float* input_tensor = new float[m_model_channels * m_model_width * m_model_height];
    pre_process(input, input_tensor);
    m_session.Run(Ort::RunOptions{nullptr}, input_tensor, m_model_channels * m_model_width * m_model_height * sizeof(float), nullptr, 0);

    // create input tensor
    delete[] input_tensor;
}

void Model::pre_process(cv::Mat input, float* input_tensor)
{
    assert(sizeof(input_tensor) == m_model_channels * m_model_width * m_model_height * sizeof(float));
    // pre-process input image
    cv::Mat resized;
    cv::resize(input, resized, cv::Size(m_model_width, m_model_height));
    // normalize and convert to float
    resized.convertTo(resized, CV_32FC3, 1.0f / 255.0f);
    // convert to tensor
    int input_size = m_model_width * m_model_height;
    for (int i = 0; i < m_model_width; i++)
    {
        for (int j = 0; j < m_model_height; j++)
        {
            cv::Vec3f pixel = resized.at<cv::Vec3f>(i, j);
            input_tensor[0 * input_size + i * m_model_width + j] = pixel[0];
            input_tensor[1 * input_size + i * m_model_width + j] = pixel[1];
            input_tensor[2 * input_size + i * m_model_width + j] = pixel[2];
        }
    }

    return resized;
}

std::vector<BoundingBox> Model::post_process(Ort::Value output)
{
    // post-process model output
    std::vector<BoundingBox> boxes;

    //TODO: Implement post-processing

    return boxes;
}

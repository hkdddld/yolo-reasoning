#ifndef COMMON_H
#define COMMON_H

#include "NvInfer.h"
#include "cuda_runtime_api.h

#include <cassert>

#include <cuda_runtime_api.h>

#include <NvInferRuntime.h>

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <string>
#include <vector>
#include <cstring>
#include <fstream>
#include <sstream> 
using namespace std;
using namespace cv;
#define CHECK(status) \
    do\
    {\
        auto ret = (status);\
        if (ret != 0)\
        {\
            std::cerr << "Cuda failure: " << ret << std::endl;\
            abort();\
        }\
    } while (0)
struct Configuration
{
public:
    float confThreshold; // 类别置信度阈值
    float nmsThreshold;  // 非最大抑制阈值
    float objThreshold;  // 物体置信度阈值
    string modelpath;    // 模型路径
};

// 定义BoxInfo结构类型
typedef struct BoxInfo
{
    float x1;
    float y1;
    float x2;
    float y2;
    float score;
    int label;
} BoxInfo;

class YOLOv5
{
public:
    YOLOv5(Configuration config);
    void detect(Mat &frame, bool &draw);

private:
    float confThreshold;
    float nmsThreshold;
    float objThreshold;
    int inpWidth;
    int inpHeight;
    int nout;
    int num_proposal;
    int num_classes;
    string classes[80] = {"person", "bicycle", "car", "motorbike", "aeroplane", "bus",
                          "train", "truck", "boat", "traffic light", "fire hydrant",
                          "stop sign", "parking meter", "bench", "bird", "cat", "dog",
                          "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
                          "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
                          "skis", "snowboard", "sports ball", "kite", "baseball bat",
                          "baseball glove", "skateboard", "surfboard", "tennis racket",
                          "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
                          "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
                          "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant",
                          "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse",
                          "remote", "keyboard", "cell phone", "microwave", "oven", "toaster",
                          "sink", "refrigerator", "book", "clock", "vase", "scissors",
                          "teddy bear", "hair drier", "toothbrush"};

    const bool keep_ratio = true;
    vector<float> input_image_; // 输入图片
    void normalize_(Mat img);   // 归一化函数
    void nms(vector<BoxInfo> &input_boxes);
    Mat resize_image(Mat srcimg, int *newh, int *neww, int *top, int *left);


    // SessionOptions sessionOptions;
    const char* INPUT_BLOB_NAME = "images";
    const char* OUTPUT_BLOB_NAME = "output0";
    float* pdata;                               // 存储推理结果概率值的指针
    int output_size = 1;                       // 输出张量的大小
    IRuntime* runtime;                         // TensorRT 运行时对象指针
    ICudaEngine* engine;                       // TensorRT CUDA 引擎对象指针
    IExecutionContext* context;                // TensorRT 执行上下文对象指针
};
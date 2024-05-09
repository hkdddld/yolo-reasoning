#include "../include/YOLOV5.hpp"

YOLOv5::YOLOv5(Configuration config)
{//从配置中初始化参数：
    this->confThreshold = config.confThreshold;
    this->nmsThreshold = config.nmsThreshold;
    this->objThreshold = config.objThreshold;
    // 对于一个数组，其大小等于整个数组占用的字节数除以数组中第一个元素的字节数。
    this->num_classes = sizeof(this->classes) / sizeof(this->classes[0]); // 类别数量 
//获取模型路径
    string model_path = config.modelpath;
    //sessionOptions 是用于配置 Session 对象的选项对象    
    sessionOptions.SetGraphOptimizationLevel(ORT_ENABLE_BASIC); // 设置图优化类型为ORT_ENABLE_BASIC

    session_options.SetExecutionPrecision(ExecutionPrecision::ORT_FP16);
//(const char *)model_path.c_str()将模型路径转化为c风格 (const char *)将model_path.c_str()明确的标为c风格指针    new 运算符用于在堆上动态分配内存    
    ort_session = new Session(env, (const char *)model_path.c_str(), sessionOptions);//.c_str(返回了一个指向模型路径的c风格(const char *)指针
    //设置模型输入输出节点和维度信息：
    size_t numInputNodes = ort_session->GetInputCount(); // 输入输出节点数量ort_session 是一个 ONNX Runtime 中的 Session 的指针 GetInputCount()获取输入节点的函数
    size_t numOutputNodes = ort_session->GetOutputCount();  //输出节点数量
    //AllocatorWithDefaultOptions 是 ONNX Runtime 提供的默认选项的内存分配器。
    AllocatorWithDefaultOptions allocator; // 配置 ONNX Runtime 中的内存分配器。
    //获取了模型的每个输入节点的名称、类型和形状信息。
    for (int i = 0; i < numInputNodes; i++) 
    {//AllocatedStringPtr   字符串类  通过分配器 allocator 来分配内存的
        Ort::AllocatedStringPtr input_name = ort_session->GetInputNameAllocated(i, allocator);  // 获取第i个节点的名称并分配内存
        // input_names.push_back(input_name.get());
        Ort::TypeInfo input_type_info = ort_session->GetInputTypeInfo(i);//获取了第 i 个输入节点的类型信息。这包括了输入节点的数据类型、形状等信息
        auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();//输入节点的张量信息
        auto input_dims = input_tensor_info.GetShape();//形状信息，通常是一个数组或向量。
        input_node_dims.push_back(input_dims);//将信息添加到 input_node_dims 
    }
// 上面那个是对于输出节点 下输入
    for (int i = 0; i < numOutputNodes; i++)
    {
        Ort::AllocatedStringPtr output_name = ort_session->GetOutputNameAllocated(i, allocator);
        // output_names.push_back(output_name.get());
        Ort::TypeInfo output_type_info = ort_session->GetOutputTypeInfo(i);
        auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
        auto output_dims = output_tensor_info.GetShape();
        output_node_dims.push_back(output_dims);
    }
    //设置输入输出节点名称
    input_names.push_back("images");
    output_names.push_back("output0");
 
    //初始化模型的其他参数    这些参数都是从模型中获得的
    this->inpHeight = input_node_dims[0][2];    //高
    this->inpWidth = input_node_dims[0][3];//宽
    this->nout = output_node_dims[0][2];         // 5+classes 输出节点的通道数
    this->num_proposal = output_node_dims[0][1]; // pre_box 预测框的数量
}

Mat YOLOv5::resize_image(Mat srcimg, int *newh, int *neww, int *top, int *left)
{
    int srch = srcimg.rows, srcw = srcimg.cols;
    *newh = this->inpHeight; //获取期望宽高
    *neww = this->inpWidth;
    Mat dstimg;
    if (this->keep_ratio && srch != srcw)   //是否要缩放 是否是正方行图片
    {
        float hw_scale = (float)srch / srcw;
        if (hw_scale > 1)//两侧
        {
            *newh = this->inpHeight;
            *neww = int(this->inpWidth / hw_scale);
            resize(srcimg, dstimg, Size(*neww, *newh), INTER_AREA);//缩放
            *left = int((this->inpWidth - *neww) * 0.5);//左侧要填充的量
            copyMakeBorder(dstimg, dstimg, 0, 0, *left, this->inpWidth - *neww - *left, BORDER_CONSTANT, 114);// copyMakeBorder填充方法
        }
        else
        {
            *newh = (int)this->inpHeight * hw_scale;
            *neww = this->inpWidth;
            resize(srcimg, dstimg, Size(*neww, *newh), INTER_AREA);
            *top = (int)(this->inpHeight - *newh) * 0.5;
            copyMakeBorder(dstimg, dstimg, *top, this->inpHeight - *newh - *top, 0, 0, BORDER_CONSTANT, 114);
        }
    }
    else
    {
        resize(srcimg, dstimg, Size(*neww, *newh), INTER_AREA);
    }
    return dstimg;
}

void YOLOv5::normalize_(Mat img)//将输入图片转化为一维数组并且改成rgb并归一化
{
    int row = img.rows;//宽
    int col = img.cols;//高
    this->input_image_.resize(row * col * img.channels()); //调整容器的大小 vector大小 img.channels()返回图像的通道数
    for (int c = 0; c < 3; c++)                            // bgr
    {
        for (int i = 0; i < row; i++) // 行
        {
            for (int j = 0; j < col; j++) // 列
            {
                float pix = img.ptr<uchar>(i)[j * 3 + 2 - c]; // ptr（）访问任意一行像素的首地址,2-c:表示rgb 取出的是一个
                this->input_image_[c * row * col + i * col + j] = pix / 255.0;//转化为红，绿，蓝
            }//| R0 | G0 | B0 | R1 | G1 | B1 | R2 | G2 | B2 | R3 | G3 | B3 |图像在计算机内部的存储方式
        }
    }
}

// void  YOLOv5::normalize_(Mat img )
// {   
//     Mat x;
//     Mat channels1[3];

//     cvtColor(img, x, COLOR_BGR2RGB);
//     x.convertTo(x, CV_32FC3);//转化数据类型
//     x *= 1 / 255.0;//归一化

//     split(x, channels1);//split 函数将输入图像 src 按通道分离，分别存储在 channels[0]、channels[1]、channels[2] 中
//     int offset = x.rows * x.cols;
//     this->input_image_.resize(offset * x.channels());
//     auto start = this->input_image_.data();//获得起是起始地址（指针）
   
//     for (int i = 0; i < 3; ++i)
//     {
//         memcpy(start + i * offset, channels1[i].data, offset * sizeof(float));
//     }

// }
// void YOLOv5::normalize_(Mat img) {
//     Mat x;
//     cvtColor(img, x, COLOR_BGR2RGB);
//     x.convertTo(x, CV_32FC3); // 将图像类型转换为 CV_32FC3
//     x *= 1.0f / 255.0f;       // 归一化到 [0, 1] 范围内

//     std::vector<uint16_t> inputTensorValueFp16;
//     for (int i = 0; i < x.rows; ++i) { //行
//         for (int j = 0; j < x.cols; ++j) {//列
//             for (int c = 0; c < x.channels(); ++c) {
//                 float pixel_value = x.at<Vec3f>(i, j)[c]; // 获取归一化后的像素值
//                 uint16_t fp16_value = float32_to_float16(pixel_value); // 转换为 FP16
//                 inputTensorValueFp16.push_back(fp16_value);
//             }
//         }
//     }

//     Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator(), OrtMemType::OrtMemTypeDefault);
//     Ort::Value input_image = Ort::Value::CreateTensor(memoryInfo, inputTensorValueFp16.data(), inputTensorValueFp16.size() * sizeof(uint16_t), {1, x.rows, x.cols, x.channels()}, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16);
// }

void YOLOv5::nms(vector<BoxInfo> &input_boxes)
{//将输入框按照置信度（score）进行降序排列，以便优先保留置信度高的框。
    sort(input_boxes.begin(), input_boxes.end(), [](BoxInfo a, BoxInfo b)
         { return a.score > b.score; }); // 降序排列[begin, end)
    //计算每个框的面积并存储在 vArea 中。
    vector<float> vArea(input_boxes.size());
    for (int i = 0; i < input_boxes.size(); ++i)
    {
        vArea[i] = (input_boxes[i].x2 - input_boxes[i].x1 + 1) * (input_boxes[i].y2 - input_boxes[i].y1 + 1);
    }
    // 全初始化为false，用来作为记录是否保留相应索引下pre_box的标志vector
    vector<bool> isSuppressed(input_boxes.size(), false);
    for (int i = 0; i < input_boxes.size(); ++i)
    {
        if (isSuppressed[i])
        {
            continue;
        }
        for (int j = i + 1; j < input_boxes.size(); ++j)
        {
            if (isSuppressed[j])
            {
                continue;//如果 isSuppressed[i] 的值为真（即 true），即表示该索引对应的元素被抑制（suppressed）了，那么代码会执行 continue;
            }
            float xx1 = max(input_boxes[i].x1, input_boxes[j].x1);
            float yy1 = max(input_boxes[i].y1, input_boxes[j].y1);
            float xx2 = min(input_boxes[i].x2, input_boxes[j].x2);
            float yy2 = min(input_boxes[i].y2, input_boxes[j].y2);

            float w = max(0.0f, xx2 - xx1 + 1);
            float h = max(0.0f, yy2 - yy1 + 1);
            float inter = w * h; // 交集
            
            if (input_boxes[i].label == input_boxes[j].label)
            {
                //遍历每个框，对于每对框，计算它们的交集面积（inter）和它们的 IoU（交并比），如果 IoU 大于等于设定的阈值（this->nmsThreshold），则标记其中一个框应该被抑制。
                float ovr = inter / (vArea[i] + vArea[j] - inter); // 计算iou
                if (ovr >= this->nmsThreshold)
                {
                    isSuppressed[j] = true;
                }
            }
        }
    }
    // return post_nms;
    int idx_t = 0;
    // remove_if()函数 remove_if(beg, end, op)    //移除区间[beg,end)中每一个“令判断式:op(elem)获得true”的元素
    input_boxes.erase(remove_if(input_boxes.begin(), input_boxes.end(), [&idx_t, &isSuppressed](const BoxInfo &f)
                                { return isSuppressed[idx_t++]; }),
                      input_boxes.end());//remove_if将满足条件的文件移到末尾并返回指向末尾的迭代器
}

void YOLOv5::detect(Mat &frame, bool &draw)
{
    // 图像预处理
    int newh = 0, neww = 0, padh = 0, padw = 0;
    Mat dstimg = this->resize_image(frame, &newh, &neww, &padh, &padw);//缩放图像
    this->normalize_(dstimg);//归一化图像

    // 定义一个输入矩阵，int64_t是下面作为输入参数时的类型
    array<int64_t, 4> input_shape_{1, 3, this->inpHeight, this->inpWidth};
    // 创建内存分配器信息： 创建了一个cpu内存分配的对象OrtDeviceAllocator: 这可能是一个表示ONNX Runtime中设备分配器（device allocator）的类或常量。这里用于指定分配器是用于CPU。OrtMemTypeCPU: 这可能是一个指定内存类型的常量，表示在CPU上进行内存分配
    auto allocator_info = MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    // 创建输入tensor
    Value input_tensor_ = Value::CreateTensor<float>(allocator_info, input_image_.data(), input_image_.size(), input_shape_.data(), input_shape_.size());

    // 开始推理ort_session指向模型的指针
    vector<Value> ort_outputs = ort_session->Run(RunOptions{nullptr}, &input_names[0], &input_tensor_, 1, output_names.data(), output_names.size()); 
                                            /*
                                ort_session->Run: 这是ONNX Runtime库的Session类的Run方法，用于执行模型推理。它接受一系列参数，其中包括输入和输出的名称、输入和输出的数据等。

                                            RunOptions{nullptr}: 这是一个RunOptions对象的初始化。RunOptions通常用于设置推理选项，例如设置超时等。在这里，通过使用{nullptr}，表示没有特殊的运行选项。

                                            &input_names[0]: 这是输入节点的名称数组的首地址。input_names是一个存储输入节点名称的字符串数组，通过取其首地址作为参数传递给Run方法。

                                            &input_tensor_: 这是输入数据的Tensor。input_tensor_是一个由前面的代码创建的ONNX Runtime Tensor，它包含了要传递给模型的输入数据。

                                            1: 表示要运行一次推理。在这里，只运行一次推理，但在其他情况下，可以根据需要设置不同的运行次数。

                                            output_names.data(): 这是输出节点的名称数组的首地址。output_names是一个存储输出节点名称的字符串数组，通过取其首地址作为参数传递给Run方法。

                                            output_names.size(): 这是输出节点的数量。通过使用output_names.size()，指定要获取的输出节点的数量。
                                            */
    // generate proposals
    //生成检测框
    vector<BoxInfo> generate_boxes; //创建一个名称为generate_boxes格式BoxInfo格式的动态数组
    float ratioh = (float)frame.rows / newh, ratiow = (float)frame.cols / neww;//ratioh：表示图像高度 frame.rows 与 newh 的比率。ratiow：表示图像宽度 frame.cols 与 neww 的比率。
    float *pdata = ort_outputs[0].GetTensorMutableData<float>(); // GetTensorMutableData
// ort_outputs[0]：这是一个索引运算符，用于访问名为 ort_outputs 的数组或容器中的第一个元素。这里假设 ort_outputs 是一个包含了输出张量的容器或数组。
// .GetTensorMutableData<float>()：这是调用了一个函数或方法，它可能是一个成员函数或者一个全局函数。这个函数的作用是获取张量的可变数据指针，该指针指向张量的数据缓冲区，并且将数据的类型视为 float 类型。这表示张量中的数据被解释为 float 类型的数据。
// float *pdata：这是一个指针变量的声明，类型为 float，用于存储张量数据的可变指针。
    for (int i = 0; i < num_proposal; ++i)                       // 遍历所有的num_pre_boxes
    {
        int index = i * nout;              //指向第几个节点 
        float obj_conf = pdata[index + 4]; // 置信度分数
        if (obj_conf > this->objThreshold) // 大于阈值
        {
            int class_idx = 0;
            float max_class_socre = 0;//最类别概率的存储
            for (int k = 0; k < this->num_classes; ++k)//遍历找到最大的类别概率
            {
                if (pdata[k + index + 5] > max_class_socre)
                {
                    max_class_socre = pdata[k + index + 5];
                    class_idx = k;
                }
            }
            max_class_socre *= obj_conf;               // 最大的类别分数*置信度
            if (max_class_socre > this->confThreshold) // 再次筛选
            {
                // const int class_idx = classIdPoint.x;
                float cx = pdata[index];     // x
                float cy = pdata[index + 1]; // y
                float w = pdata[index + 2];  // w
                float h = pdata[index + 3];  // h

                float xmin = (cx - padw - 0.5 * w) * ratiow;//cx给定的是中心点的坐标减去填充值向左移动标准框的距离*比例是为了还原到原图上   一开始按输入要求压缩过
                float ymin = (cy - padh - 0.5 * h) * ratioh;
                float xmax = (cx - padw + 0.5 * w) * ratiow;
                float ymax = (cy - padh + 0.5 * h) * ratioh;

                generate_boxes.push_back(BoxInfo{xmin, ymin, xmax, ymax, max_class_socre, class_idx});//向动态数组添加东西
            }
        }
    }

    // Perform non maximum suppression to eliminate redundant overlapping boxes with
    // lower confidences
    nms(generate_boxes);
    if (draw)
    {
        for (size_t i = 0; i < generate_boxes.size(); ++i)
        {
            int xmin = int(generate_boxes[i].x1);
            int ymin = int(generate_boxes[i].y1);
    //画框Point(x, y) 表示一个二维平面上的点
            rectangle(frame, Point(xmin, ymin), Point(int(generate_boxes[i].x2), int(generate_boxes[i].y2)), Scalar(0, 0, 255), 2);
           //物体的置信度分数格式化为字符串，保留两位小数，存储在 label 变量中。
            string label = format("%.2f", generate_boxes[i].score);
            //将物体的类别信息与置信度分数拼接在一起
            label = this->classes[generate_boxes[i].label] + ":" + label;
            
            putText(frame, label, Point(xmin, ymin - 5), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 255, 0), 1);
        }
    }
}
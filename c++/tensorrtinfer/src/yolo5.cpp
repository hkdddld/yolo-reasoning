YOLOv5::YOLOv5(Configuration config)
{
    this->confThreshold = config.confThreshold;
    this->nmsThreshold = config.nmsThreshold;
    this->objThreshold = config.objThreshold;
    this->num_classes = sizeof(this->classes) / sizeof(this->classes[0]); // 类别数量
    this->inpHeight = 640;
    this->inpWidth = 640;
    string model_path = config.modelpath;

    size_t size{0};
    char *trtModelStream{nullptr};
    // 读取文件
    std::ifstream file(model_path, std::ios::binary);
    if (file.good()) {
        file.seekg(0, file.end);    
        size = file.tellg();
        file.seekg(0, file.beg);
        trtModelStream = new char[size];
        assert(trtModelStream);
        file.read(trtModelStream, size);
        file.close();
    }
    std::cout << "engine init finished" << std::endl;

    runtime = createInferRuntime(gLogger);  //  创造tensorrt 运行对象
    assert(runtime != nullptr);// 检查运行时对象是否创建成功
    engine = runtime->deserializeCudaEngine(trtModelStream, size);
    assert(engine != nullptr);
    context = engine->createExecutionContext();
    assert(context != nullptr);
    delete[] trtModelStream;
    auto out_dims = engine->getBindingDimensions(1);//计算输出张量大小
    for(int j=0;j<out_dims.nbDims;j++) {
        this->output_size *= out_dims.d[j];
    }
    this->pdata = new float[this->output_size];// 分配存储推理结果概率值的内存
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

void YOLOv5::doInference(IExecutionContext& context, float* input, float* output, const int output_size, cv::Size input_shape) {
    const ICudaEngine& engine = context.getEngine();

    // 确保引擎绑定的缓冲区数量为2
    assert(engine.getNbBindings() == 2);
    void* buffers[2]; // 输入和输出缓冲区

    // 获取输入和输出张量的绑定索引
    const int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME);
    const int outputIndex = engine.getBindingIndex(OUTPUT_BLOB_NAME);

    // 确保输入和输出张量的数据类型为 FLOAT
    assert(engine.getBindingDataType(inputIndex) == nvinfer1::DataType::kFLOAT);
    assert(engine.getBindingDataType(outputIndex) == nvinfer1::DataType::kFLOAT);

    // 获取引擎支持的最大批处理大小
    int mBatchSize = engine.getMaxBatchSize();

    // 在设备上为输入和输出创建 GPU 缓冲区
    CHECK(cudaMalloc(&buffers[inputIndex], 3 * input_shape.height * input_shape.width * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex], output_size * sizeof(float)));

    // 创建 CUDA 流
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // 将输入数据异步传输到设备上
    CHECK(cudaMemcpyAsync(buffers[inputIndex], input, 3 * input_shape.height * input_shape.width * sizeof(float), cudaMemcpyHostToDevice, stream));

    // 使用执行上下文异步执行推理
    context.enqueue(1, buffers, stream, nullptr);

    // 将推理结果从设备异步传输回主机
    CHECK(cudaMemcpyAsync(output, buffers[outputIndex], output_size * sizeof(float), cudaMemcpyDeviceToHost, stream));

    // 同步 CUDA 流，等待推理完成
    cudaStreamSynchronize(stream);

    // 释放 CUDA 流和缓冲区内存
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex]));
}
void YOLOv5::detect(Mat &frame, bool &draw)
{
    // 图像预处理
    int newh = 0, neww = 0, padh = 0, padw = 0;
    Mat dstimg = this->resize_image(frame, &newh, &neww, &padh, &padw);//缩放图像
    this->normalize_(dstimg);//归一化图像

    this->doInference(context,input_image_, this->pdata, output_size, dstimg.size())
    const int num_class = 80;
    auto dets = output_size / (num_class + 5);
    for (int i = 0; i < dets; ++i)                       // 遍历所有的num_pre_boxes
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
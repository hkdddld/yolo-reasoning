
#include "../include/yolo5.hpp"

int main(int argc, char *argv[])
{
   
    clock_t startTime, endTime; // 计算时间
    Configuration yolo_nets = {0.3f, 0.5f, 0.3f, "yolov5_2.trt"};
    YOLOv5 yolo_model(yolo_nets);//yolo_model 是一个名为 yolo_model 的对象，它使用 yolo_nets 进行初始化。
    Mat srcimg = imread("bus.jpg");
    double timeStart = (double)getTickCount();//这一行代码使用 getTickCount() 函数获取当前系统时钟的计数值，将其转换为 double 类型，并将结果赋值给 timeStart 变量
    startTime = clock(); // 计时开始
    bool draw = true;
    yolo_model.detect(srcimg, draw);
    
    endTime = clock();                                                               // 计时结束
    double nTime = ((double)getTickCount() - timeStart) / getTickFrequency() * 1000; // 转换为毫秒
    cout << "clock_running time is:" << (double)(endTime - startTime) / CLOCKS_PER_SEC * 1000 << "ms" << endl;
    cout << "The run time is:" << (double)clock() / CLOCKS_PER_SEC * 1000 << "ms" << endl;
    cout << "getTickCount_running time :" << nTime << "ms\n"<< endl;
    if (draw)
    {
        imshow("bus", srcimg);
        // imwrite("bus.jpg", srcimg);
        waitKey(0);
    }
    
    return 0;
}
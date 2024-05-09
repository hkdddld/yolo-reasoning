
#include "common.hpp"

// 配置结构体，包含模型和检测的一些参数
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
float confThreshold;      // 置信度阈值
    float nmsThreshold;       // 非最大抑制阈值
    float objThreshold;       // 物体置信度阈值
    int inpWidth;             // 输入图像的宽度
    int inpHeight;            // 输入图像的高度
    int nout;                 // 输出节点数
    int num_proposal;         // 建议框的数量
    int num_classes;          // 目标类别的数量
    string classes[80] = {"人", "自行车", "汽车", "摩托车", "飞机", "公共汽车",
                        "火车", "卡车", "船", "交通灯", "消防栓",
                        "停车标志", "停车计时器", "长椅", "鸟", "猫", "狗",
                        "马", "羊", "牛", "大象", "熊", "斑马", "长颈鹿",
                        "背包", "雨伞", "手提包", "领带", "手提箱", "飞盘",
                        "滑雪板", "滑雪板", "运动球", "风筝", "棒球棒",
                        "棒球手套", "滑板", "冲浪板", "网球拍",
                        "瓶子", "酒杯", "杯子", "叉子", "刀", "勺子", "碗",
                        "香蕉", "苹果", "三明治", "橙子", "西兰花", "胡萝卜",
                        "热狗", "披萨", "甜甜圈", "蛋糕", "椅子", "沙发", "盆栽植物",
                        "床", "餐桌", "厕所", "电视监视器", "笔记本电脑", "鼠标",
                        "遥控器", "键盘", "手机", "微波炉", "烤箱", "烤面包机",
                        "水槽", "冰箱", "书", "时钟", "花瓶", "剪刀",
                        "泰迪熊", "吹风机", "牙刷"};


    const bool keep_ratio = true;//是否保持宽高比
    vector<float> input_image_; // 输入图片
    void normalize_(Mat img);   // 归一化函数
    void nms(vector<BoxInfo> &input_boxes);//非最大值抑制
    Mat resize_image(Mat srcimg, int *newh, int *neww, int *top, int *left);//裁剪

    Env env = Env(ORT_LOGGING_LEVEL_ERROR, "yolov5-6.1"); // 初始化环境
    Session *ort_session = nullptr;                       // 初始化Session指针选项
    SessionOptions sessionOptions = SessionOptions();     // 初始化Session对象
    SessionOptions sessionOptions;
    vector<char *> input_names;               // 定义一个字符指针vector（动态数组里面存的是指针）
    vector<char *> output_names;              // 定义一个字符指针vector
    vector<vector<int64_t>> input_node_dims;  // >=1 outputs  ，二维vector
    vector<vector<int64_t>> output_node_dims; // >=1 outputs ,int64_t C/C++标准
};
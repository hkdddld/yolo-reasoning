#include "json.hpp"
#include <iostream>
using namespace std;
class Motion
{
public:
    struct Params//用于存储变量的原格式
    {
        float confidencethreshold; // 置信度阈值
        float nmsthreshold;        // 非极大值抑制 (NMS) 阈值
        float objthreshold;        // 目标检测阈值
        string modelpath;          // 模型文件路径
        string imgpath;            // 图像文件路径
        bool draw;                 // 是否绘制结果

        NLOHMANN_DEFINE_TYPE_INTRUSIVE(Params, confidencethreshold, nmsthreshold, objthreshold, modelpath, imgpath, draw); // 添加构造函数
    };
    Params params;
    Motion()
    {
        string jsonPath = "../config/config.json";//定义 JSON 配置文件的路径。
        std::ifstream config_is(jsonPath);// 创建一个输入文件流，以便从文件中读取 JSON 数据。

        // 检查文件是否成功打开
        if (!config_is.good()) {
            std::cout << "Error: Params file path:[" << jsonPath << "] not find .\n";
            exit(-1);
        }

        // 创建一个 JSON 对象
        nlohmann::json js_value;

        // 从文件中读取 JSON 数据
        config_is >> js_value;

        try {
            // 将 JSON 数据反序列化为 Params 结构体
            params = js_value.get<Motion::Params>();
        } catch (const nlohmann::detail::exception &e) {
            // 捕获可能的异常，例如 JSON 解析失败
            std::cerr << "Json Params Parse failed :" << e.what() << '\n';
            exit(-1);
        }

    }
};
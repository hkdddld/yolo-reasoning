import ctypes
import os
import random
import sys
import threading
import time

import cv2
import numpy as np
import pycuda.driver as cuda
import tensorrt as trt
input_width=640
input_height=640
class YOLOv5:
    def __init__(self,
                engine_file_path: str,  # 模型文件路径
                classes,
                original_size,  # 原始图像尺寸，默认为(1280, 720)
                score_threshold: float = 0.3,  # 分数阈值，用于过滤检测结果，默认为0.1
                conf_threshold: float = 0.3,  # 置信度阈值，用于过滤检测结果，默认为0.4
                iou_threshold: float = 0.45,  # IOU阈值，用于判断两个物体是否重叠，默认为0.4
                ) -> None:  # 模型推理使用的设备，默认为"CPU"
        self.engine_file_path=engine_file_path
        self.score_threshold=score_threshold

        self.conf_threshold=conf_threshold
        self.iou_threshold=iou_threshold
        self.image_width,self.image_height=original_size
        self.input_width=input_width
        self.input_height=input_height
        self.classes=classes

        self.color_palette=np.random.uniform(0,255,size=(len(self.classes),3))
        cuda.init()#初始化
        self.cfx=cuda.Device(0).make_context()#CUDA 上下文堆栈 确定工作的gpu
        stream=cuda.Stream()#创建一个CUDA流
        TRT_LOGGER=trt.Logger(trt.Logger.INFO)#日志记录器 输出级别为INFO以上的信息
        runtime =trt.Runtime(TRT_LOGGER)#运行时的主要接口之一 用于加速

        #获取模型序列
        with open(engine_file_path,'rb') as f:
            engine=runtime.deserialize_cuda_engine(f.read())#f.read()将读取的内容转化为字符串输入
        context=engine.create_execution_context()#用于执行推理的对象

        host_inputs=[]
        cuda_inputs=[]
        host_outputs=[]
        cuda_outputs=[]
        bindings=[]#模型推理过程中绑定的信息

        for binding in engine: #binding 将模型的输入和输出与gpu关联起来的概念
            #计算内存大小 engine.max_batch_size 获取批次数 trt.volume() 计算形状的体积 (总元素个数)
            size =trt.volume(engine.get_tensor_shape(binding))*engine.max_batch_size
            #确定数据类型
            dtype =trt.nptype(engine.get_tensor_dtype(binding))#获取原来的数据类型 再通过nptype转化为np类型
            #返回一个 页锁定主机内存缓冲区 用于cpu和gpu高速传输
            host_mem=cuda.pagelocked_empty(size,dtype)#cpu端的内存 用于存储输入和输出数据 
            cuda_mem=cuda.mem_alloc(host_mem.nbytes)#用于存储输入和输出数据 表示了gpu上的一段内存
            bindings.append(int(cuda_mem))#int(cuda_mem)获取了他的地址 将他存入中

            if engine.binding_is_input(binding):#检测是输入还是输出状态
                host_inputs.append(host_mem)#输入数据
                cuda_inputs.append(cuda_mem)#gpu上的设备内存对象
            else:
                host_outputs.append(host_mem)#储模型输出结果的主机端（CPU 端）内存对象 host_mem 
                cuda_outputs.append(cuda_mem)#输出数据
        
        self.stream=stream
        self.context=context#重用上下文 以便执行模型的前向操作
        self.engine=engine# 推理操作
        self.host_inputs=host_inputs
        self.cuda_inputs=cuda_inputs
        self.host_outputs=host_outputs
        self.cuda_outputs=cuda_outputs
        self.bindings=bindings
    def preprocess(self,img):
        self.input_width=input_width
        self.input_height=input_height
        image_rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        
        resized=cv2.resize(image_rgb,(self.input_width,self.input_height)) 

        input_image=resized/255.0
        input_image=input_image.transpose(2,0,1)
        input_tensor=input_image[np.newaxis,:,:,:].astype(np.float32)#增加一个维度，然后转成fp32
        return input_tensor

    def infer(self, input_image):
        input_image = self.preprocess(input_image)  # 预处理输入图像
        threading.Thread.__init__(self)
        # Make self the active context, pushing it on top of the context stack.
        self.cfx.push()
        # Restore
        stream = self.stream
        context = self.context
        engine = self.engine
        host_inputs = self.host_inputs
        cuda_inputs = self.cuda_inputs
        host_outputs = self.host_outputs
        cuda_outputs = self.cuda_outputs
        bindings = self.bindings
        i=0
        # print(input_image)
        # 拷贝预处理后的图像数据到输入列表中
        np.copyto(host_inputs[0], input_image.ravel())
        cuda.memcpy_htod_async(cuda_inputs[0], host_inputs[0], stream)  # 将数据传输到 GPU 上
        # print("----",cuda_inputs)
        start = time.time()  # 记录推理开始时间
        # 执行推理操作
        context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)


        # print("---------",cuda_outputs)
        cuda.memcpy_dtoh_async(host_outputs[0], cuda_outputs[0],stream)  # 异步获取输出数据
        stream.synchronize()  # 等待推理流完成

        self.cfx.pop()  # 弹出 CUDA 上下文堆栈

        output = host_outputs[0]
        end = time.time()  # 记录推理结束时间
        inference_time = (end - start) * 1000
        print("time", inference_time,i)
        fps = 1000 / inference_time
        print("fps", fps,i)
        i+=1
        detections = self.postprocess(output)  # 后处理输出结果

        return detections

    def xywh2xyxy(self,x):
        y=np.copy(x)
        y[...,0]=x[...,0]-x[...,2]/2
        y[...,1]=x[...,1]-x[...,3]/2
        y[...,2]=x[...,0]+x[...,2]/2
        y[...,3]=x[...,1]+x[...,3]/2
        return y
    def get_label_name(self,class_id:int)->str:
        return self.classes[class_id]
    def postprocess(self, outputs):
        num=int(len(outputs))
        leibei=int(num/85)
        # print(leibei)
        # predictions=np.squeeze(outputs)#8400行84列84=4坐标+80类别
        predictions = np.reshape(outputs[:], (leibei, -1))#将一维数组换成二维数组
        
        scores=np.max(predictions[:,5:],axis=1)*predictions[:,4]
        predictions=predictions[scores>self.conf_threshold,:]#scores如果小于阈值时吧整列都剔除掉吗
        scores=scores[scores>self.conf_threshold]
        class_ids=np.argmax(predictions[:,5:],axis=1)

        # print(class_ids)
        boxes=predictions[:,:4]
        input_shape=np.array([self.input_width,self.input_height,self.input_width, self.input_height])
        boxes=np.divide(boxes,input_shape,dtype=np.float32) #divide元素除法
        boxes *=np.array([self.image_width,self.image_height,self.image_width,self.image_height])#转化到原始图像上 
        indices=cv2.dnn.NMSBoxes(boxes,scores,score_threshold=self.score_threshold,nms_threshold=self.iou_threshold)
        detections=[]
        for bbox,score,label in zip(self.xywh2xyxy(boxes[indices]),scores[indices],class_ids[indices]):
            detections.append({
                "class_index":label,
                "confidence":score,
                "box":bbox,
                "class_name":self.get_label_name(label)
            })
        return detections    

    def draw_detections(self,img,detections )->None:
        for detection in detections:
            x1,y1,x2,y2=detection['box'].astype(int)
            class_id =detection['class_index']#类别号
            confidence=detection['confidence']#置信度

            color=self.color_palette[class_id]

            cv2.rectangle(img,(x1,y1),(x2,y2),color,2)
            label=f"{self.classes[class_id]}:{confidence:.2f}"
            (label_width,label_height),_=cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX,0.5,1)#计算出文本长度
            label_x=x1
            label_y=y1-10 if y1-10>label_height else y1+10#向上移动
            cv2.rectangle(
                img,(label_x,label_y-label_height),(label_x+label_width,label_y+label_height),color,cv2.FILLED
            )
            cv2.putText(img, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    def destroy(self):
            self.cfx.pop()

class mtThread(threading.Thread):
    def __init__(self,func,args,kwargs={}):
        threading.Thread.__init__(self)
        self.func=func
        self.args=args
        self.kwargs = kwargs
    def run(self):
        self.func(*self.args,**self.kwargs)#**kwargs 来接受任意数量的关键字参数
def list_image_paths(folder_path):#folder_path图片路径 读取指定文件夹下所有文件并保存
    image_paths = []
    # 检查指定文件夹是否存在
    if not os.path.exists(folder_path):
        print(f"Error: Folder '{folder_path}' does not exist.")
        return image_paths

    # 遍历指定文件夹下的所有文件和子文件夹
    for root, dirs, files in os.walk(folder_path):#root主文件夹 files文件路径
        # 只处理当前文件夹下的图片文件
        for file in files:
            if file.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
                # 构建图片文件的完整路径
                image_path = os.path.join(root, file)
                # 将图片路径添加到列表中
                image_paths.append(image_path)

    return image_paths
if __name__=="__main__":
    weight_path=r"yolov5s.trt"
    image=cv2.imread(r"bus.jpg")

    h,w=image.shape[:2]

    classe=  {  0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus',
                6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 
                13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear',
                22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 
                30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard',
                37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 
                45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza',
                54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv',
                63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink',
                72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}
    detector=YOLOv5(engine_file_path=f"{weight_path}",
                    classes=classe,
                    original_size=(w,h))
    threads = []
    warmup_iterations = 5
    for _ in range(warmup_iterations):
        detector.infer(image)
    print('----------')
    # # 创建多个线程处理图像推理和绘制
    for _ in range(10):
        thread = mtThread(detector.infer, args=(image,))  # 每个线程使用图像的副本
        threads.append(thread)
        thread.start()


    # 等待所有线程执行完毕
    for thread in threads:
        thread.join()
   

    detector.destroy()#释放 pycuda

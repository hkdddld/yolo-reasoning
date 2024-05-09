import onnxruntime as ort
import cv2
import numpy as np
import time 
import torch
import torchvision
# from typing import Dict
print(("使用的设备为:",ort.get_device()))
class YOLOv5:
    def __init__(self,
                model_path: str,  # 模型文件路径
                classes,
                original_size,  # 原始图像尺寸，默认为(1280, 720)
                score_threshold: float = 0.3,  # 分数阈值，用于过滤检测结果，默认为0.1
                conf_threshold: float = 0.3,  # 置信度阈值，用于过滤检测结果，默认为0.4
                iou_threshold: float = 0.45,  # IOU阈值，用于判断两个物体是否重叠，默认为0.4
                device: str = "GPU") -> None:  # 模型推理使用的设备，默认为"CPU"
        self.model_path=model_path


        self.device = device 
        self.score_threshold=score_threshold
        self.conf_threshold=conf_threshold
        self.iou_threshold=iou_threshold
        self.image_width,self.image_height=original_size
        self.create_session() 
        self.classes=classes

        self.color_palette=np.random.uniform(0,255,size=(len(self.classes),3))
    def create_session(self)->None:

        opt_sessiong=ort.SessionOptions()
        opt_sessiong.graph_optimization_level=ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        providers = [] 
        if self.device.casefold() != "cpu":  # 如果设备不是CPU，则添加CUDA执行提供者
            providers.append("CUDAExecutionProvider")
        session=ort.InferenceSession(self.model_path,providers=providers)
        self.session=session
        self.model_inputs=self.session.get_inputs()# 获取模型输入
        self.input_names = [self.model_inputs[i].name for i in range(len(self.model_inputs))]  # 记录输入名称
        self.input_shape = self.model_inputs[0].shape  # 获取第一个输入的形状
        self.model_output = self.session.get_outputs()  # 获取模型输出
        self.output_names = [self.model_output[i].name for i in range(len(self.model_output))]  # 记录输出名称
        self.input_height, self.input_width = self.input_shape[2:]  # 从输入形状中提取高度和宽度

    def preprocess(self,img: np.ndarray)->np.ndarray:
        image_rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        
        resized=cv2.resize(image_rgb,(self.input_width,self.input_height))

        input_image=resized/255.0
        input_image=input_image.transpose(2,0,1)
        # input_image=np.ascontiguousarray(input_image)
        input_tensor=input_image[np.newaxis,:,:,:].astype(np.float16)
        # print(input_tensor)
        return input_tensor
    def xywh2xyxy(self,x):
        y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
        y[...,0]=x[...,0]-x[...,2]/2
        y[...,1]=x[...,1]-x[...,3]/2
        y[...,2]=x[...,0]+x[...,2]/2
        y[...,3]=x[...,1]+x[...,3]/2
        return y

    def postprocess(self, outputs):#
        predictions=np.squeeze(outputs)#8400行84列84=4坐标+80类别
        # 将 predictions 转换为 PyTorch Tensor，并移动到 GPU
        predictions = torch.tensor(predictions).cuda()

        # 计算 scores（置信度）
        scores = torch.max(predictions[:, 5:], dim=1)[0] * predictions[:, 4]

        # 根据置信度阈值过滤 predictions
        mask = scores > self.conf_threshold
        predictions = predictions[mask]

        # 重新计算 scores 和 class_ids
        scores = scores[mask]
        class_ids = torch.argmax(predictions[:, 5:], dim=1)

        # 提取 boxes 坐标
        boxes = predictions[:, :4]

        # 计算输入形状和图像形状
        input_shape = torch.tensor([self.input_width, self.input_height, self.input_width, self.input_height], dtype=torch.float32).cuda()
        boxes = boxes / input_shape  # 按元素除法

        image_shape = torch.tensor([self.image_width, self.image_height, self.image_width, self.image_height], dtype=torch.float32).cuda()
        boxes *= image_shape  # 按元素乘法，转换到原始图像上

        # 将 boxes 转换为 float16 类型
        boxes = boxes.to(torch.float16)

        indices=torchvision.ops.nms(boxes,scores,iou_threshold=self.iou_threshold).cpu()
        detections=[]
        for bbox,score,label in zip(self.xywh2xyxy(boxes[indices]),scores[indices],class_ids[indices]):
            detections.append({
                "class_index":label.item(),
                "confidence":score.item(),
                "box":bbox.cpu().numpy(),
                "class_name":self.get_label_name(label.item())
            })
        return detections
    def get_label_name(self,class_id:int)->str:
        return self.classes[class_id]
    
    def detect(self,img:np.ndarray):
        start_time=time.time()
        input_tensor=self.preprocess(img)
        outputs=self.session.run(self.output_names,{self.input_names[0]:input_tensor})[0]
        end_time=time.time()
        inference_time=(end_time-start_time)*1000
        print("time",inference_time)
        fps=1000/inference_time
        print("fps",fps)
        return self.postprocess(outputs)

    def draw_detections(self,img,detections )->None:
        for detection in detections:
            x1,y1,x2,y2=detection['box'].astype(int)
            class_id =detection['class_index']
            confidence=detection['confidence']

            color=self.color_palette[class_id]

            cv2.rectangle(img,(x1,y1),(x2,y2),color,2)
            label=f"{self.classes[class_id]}:{confidence:.2f}"
            (label_width,label_height),_=cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX,0.5,1)#计算出文本长度
            label_x=x1
            label_y=y1-10 if y1-10>label_height else y1+10
            cv2.rectangle(
                img,(label_x,label_y-label_height),(label_x+label_width,label_y+label_height),color,cv2.FILLED
            )
            cv2.putText(img, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

if __name__=="__main__":
    weight_path=r"yolov5s_1.onnx"
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
    detector=YOLOv5(model_path=f"{weight_path}",
                    classes=classe,
                    original_size=(w,h))
    for i in range(10):
        detections = detector.detect(image)
        detector.draw_detections(image, detections=detections)
    # detections = detector.detect(image)
    # detector.draw_detections(image, detections=detections)
    # print(("使用的设备为:",ort.get_device()))
    # cv2.imshow("结果", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
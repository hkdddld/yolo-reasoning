## 在树莓派和jetson nano上通过ncnn ort tensorrt 部署的yolov5对128类检测模型 python版

如果要跑yolov9模型的话将 postprocess 下改成predictions=np.squeeze(outputs).T

scores== torch.max(predictions[:, 4:], dim=1)[0] 

模型地址也要该
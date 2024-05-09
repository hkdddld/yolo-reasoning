import onnx
from onnxconverter_common import float16
 
model = onnx.load(r"C:\Users\15511\Desktop\python\c++\onnxinfer-main\python-ort\yolov5s.onnx")
model_fp16 = float16.convert_float_to_float16(model)
onnx.save(model_fp16, r"C:\Users\15511\Desktop\python\c++\onnxinfer-main\python-ort\yolov5s_1.onnx")
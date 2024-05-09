import onnx
import tensorrt as trt
import os

EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
TRT_LOGGER = trt.Logger(trt.Logger.INFO)

def build_engine(onnx_model_path, engine_file_path):
    """Builds a TensorRT engine from an ONNX model and saves it to a file."""
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(EXPLICIT_BATCH) as network, trt.OnnxParser(network, TRT_LOGGER) as parser, trt.Runtime(TRT_LOGGER) as runtime:
        workspace_size = 1 << 30  # 1 GB
        builder.max_workspace_size = workspace_size

        # Load ONNX model
        if not os.path.exists(onnx_model_path):
            print(f"ONNX file '{onnx_model_path}' not found.")
            return None

        print(f"Loading ONNX file from path: {onnx_model_path}")
        with open(onnx_model_path, 'rb') as model:
            print("Beginning ONNX file parsing")
            if not parser.parse(model.read()):
                print("ERROR: Failed to parse the ONNX file.")
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                return None
            print("Completed parsing of ONNX file")

        # Build TensorRT engine
        print("Building TensorRT engine. This may take a while...")
        config = builder.create_builder_config()
        engine = builder.build_engine(network, config)

        if engine is None:
            print("Failed to build TensorRT engine from ONNX model.")
            return None

        # Serialize and save the engine
        print(f"Saving TensorRT engine to file: {engine_file_path}")
        with open(engine_file_path, 'wb') as f:
            f.write(engine.serialize())

        print("TensorRT engine creation completed successfully.")
        return engine

def main():
    # Paths to ONNX model and desired TensorRT engine file
    onnx_model_path = r"C:\Users\15511\Desktop\python\c++\onnxinfer-main\python-ort\yolov5s.onnx"
    engine_file_path = r"C:\Users\15511\Desktop\python\c++\onnxinfer-main\python-ort\yolov5s.engine"

    # Build TensorRT engine from ONNX model
    build_engine(onnx_model_path, engine_file_path)

if __name__ == "__main__":
    main()

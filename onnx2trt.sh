/usr/src/tensorrt/bin/trtexec --onnx=weights/yolov9-nms.onnx \
                                --saveEngine=weights/yolov9-nms.trt \
                                --explicitBatch \
                                --minShapes=input:1x3x640x640 \
                                --optShapes=input:1x3x640x640 \
                                --maxShapes=input:4x3x640x640 \
                                --verbose \
                                --device=1
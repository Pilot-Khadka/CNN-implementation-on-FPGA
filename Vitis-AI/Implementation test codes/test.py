from ctypes import *
from typing import List
import cv2
import numpy as np
import vart
import pathlib
import xir
import os
import math
import threading
import time
import sys

current_directory = os.getcwd()
image_path = current_directory + "/../images/frog.jpg"


_B_MEAN = 104.0
_G_MEAN = 107.0
_R_MEAN = 123.0
MEANS = [_B_MEAN, _G_MEAN, _R_MEAN]
SCALES = [1.0, 1.0, 1.0]

def preprocess_image(image_path, width=224, height=224,fix_scale=1):
    means = MEANS
    scales = SCALES
    print("Current directory:",current_directory)
    print("Image path:", image_path)
    image = cv2.imread(image_path)
    print("image size:",image.shape)

    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (width, height))
    print("Image after resizing:", image.shape)

    B, G, R = cv2.split(image)

    B = (B - means[0]) * scales[0] * fix_scale
    G = (G - means[1]) * scales[1] * fix_scale
    R = (R - means[2]) * scales[2] * fix_scale

    image = cv2.merge([B,G,R])
    image = image.astype('float32')
    image /= 225.0
    #print(image)

    #cv2.imshow('img', image)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    return image

def get_child_subgraph_dpu(graph: "Graph") -> List["Subgraph"]:
    assert graph is not None, "'graph' should not be None."
    root_subgraph = graph.get_root_subgraph()
    assert (root_subgraph
            is not None), "Failed to get root subgraph of input Graph object."
    if root_subgraph.is_leaf:
        return []
    child_subgraphs = root_subgraph.toposort_child_subgraph()
    assert child_subgraphs is not None and len(child_subgraphs) > 0
    return [
        cs for cs in child_subgraphs
        if cs.has_attr("device") and cs.get_attr("device").upper() == "DPU"
    ]

def runEfficientnetv2(dpu_runner, image):
    inputTensor = dpu_runner.get_input_tensors()
    outputTensor = dpu_runner.get_output_tensors()
    print("input Tensor:",inputTensor)
    print("output tensor:",outputTensor)
    #print("input[0]:",inputTensor[0])
    print("output Tensor:", outputTensor[0])
    print("inputTensor[0] dim:",inputTensor[0].dims)
    
    input_ndim = tuple(inputTensor[0].dims)
    print(input_ndim)
    print("inputTensor[0].dims[1]", inputTensor[0].dims[1])
    print("inputTensor[0].dims[2]", inputTensor[0].dims[2])
    print("inputTensor[0].dims[3]", inputTensor[0].dims[3])
    pre_output_size = int(outputTensor[0].get_data_size() / input_ndim[0])
    print(pre_output_size)
    #print("outputTensor[0].getdatasize:",int(outputTensor[0].get_data_size))

    output_ndim = tuple(outputTensor[0].dims)

    runsize=1
    shape_in = (runsize,) + tuple([inputTensor[0].dims[i] for i in range(inputTensor[0].ndim)][1:])
    print("What shapein:",shape_in)

    input_data = [np.empty(input_ndim, dtype=np.float32, order="C")]
    print("Shape of input data:", len(input_data))
   
    image_run = input_data[0]
    print("image_run:", image_run)
    print("image run shape:", image_run.shape)
    #image_run[0,...] = image.reshape(inputTensor[0].dims[1]
    image_run[0,...] = image.reshape(inputTensor[0].dims[1],inputTensor[0].dims[2],inputTensor[0].dims[3])
    print("image_run shape after:", image_run.shape)
    print("image_run data:", image_run)
    
    print("Actual image data:", image)
    output_data = [np.empty(output_ndim, dtype=np.float32, order="C")]
    print("Output data before:", output_data)
    output_data_test = [np.empty(output_ndim, dtype=np.int8, order="C")]
     
    print("Execute async")
    #job_id = dpu_runner.execute_async(input_data, output_data)
    #print("job id:", job_id)
    #print("Output data after:", output_data)
    #dpu_runner.wait(job_id)
    #print("Execution complete")


    print("Run 2 with different input")

    print("Execute async")
    job_id2 = dpu_runner.execute_async(image_run, output_data_test)
    print("job id test:", job_id2)
    print("Output data after:", output_data_test)
    dpu_runner.wait(job_id2)
    print("Execution complete")


def main(argv):
    g = xir.Graph.deserialize(argv[1])
    subgraphs = get_child_subgraph_dpu(g)
    assert len(subgraphs) == 1

    dpu_runners = vart.Runner.create_runner(subgraphs[0], "run")
    print("DPU runner created")
    img = preprocess_image(image_path)
    
    runEfficientnetv2(dpu_runners, img)
    del dpu_runners
    
if __name__=="__main__":
    if len(sys.argv) !=2:
        print("usage: python3 test.py <xmodel location>")
    else:
        main(sys.argv)

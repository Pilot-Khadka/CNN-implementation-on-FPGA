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


"""
Example usage:
python3 ./inception_v1.py 8 /usr/share/vitis_ai_library/models/inception_v1/inception_v1.xmodel

number of threads = 8

"""
def CPUCalcSoftmax(data, size, scale):
    sum = 0.0
    result = [0 for i in range(size)]
    for i in range(size):
        result[i] = math.exp(data[i] * scale)
        sum += result[i]
    for i in range(size):
        result[i] /= sum
    return result

def get_script_directory():
    path = os.getcwd()
    return path

def TopK(datain, size, filePath):

    cnt = [i for i in range(size)]
    pair = zip(datain, cnt)
    pair = sorted(pair, reverse=True)
    softmax_new, cnt_new = zip(*pair)
    fp = open(filePath, "r")
    data1 = fp.readlines()
    fp.close()
    for i in range(5):
        flag = 0
        for line in data1:
            if (flag+1) == cnt_new[i]:
                print("Top[%d] %d %s" % (i, flag, (line.strip)("\n")))
            flag = flag + 1

def pred(datain, size, filePath):

    cnt = [i for i in range(size)]
    pair = zip(datain, cnt)
    pair = sorted(pair, reverse=True)
    softmax_new, cnt_new = zip(*pair)
    with open(filePath, "r") as fp:
        data1 = fp.readlines()

    with open(outputFilePath, "a") as output_file:
        for i in range(1):
            flag = 0
            for line in data1:
                if (flag+1) == cnt_new[i]:
                    result_line = "Top[%d] %d %s\n" % (i, flag, line.strip())
                    print(result_line)
                    output_file.write(result_line)
                flag += 1


_B_MEAN = 104.0
_G_MEAN = 107.0
_R_MEAN = 123.0
MEANS = [_B_MEAN, _G_MEAN, _R_MEAN]
SCALES = [1.0, 1.0, 1.0]


def preprocess_one_image_fn(image_path, fix_scale, width=224, height=224):
    means = MEANS
    scales = SCALES
    image = cv2.imread(image_path)
    image = cv2.resize(image, (width, height))
    B, G, R = cv2.split(image)
    B = (B - means[0]) * scales[0] * fix_scale
    G = (G - means[1]) * scales[1] * fix_scale
    R = (R - means[2]) * scales[2] * fix_scale
    image = cv2.merge([B, G, R])
    image = image.astype(np.int8)
    return image



SCRIPT_DIR = get_script_directory()
calib_image_dir = SCRIPT_DIR + "/../images/"
global threadnum
threadnum = 0

def runEfficientnet_v2(runner: "Runner", img, cnt):
    """get tensor"""
    inputTensors = runner.get_input_tensors()
    outputTensors = runner.get_output_tensors()
    input_ndim = tuple(inputTensors[0].dims)
    pre_output_size = int(outputTensors[0].get_data_size() / input_ndim[0])

    output_ndim = tuple(outputTensors[0].dims)
    output_fixpos = outputTensors[0].get_attr("fix_point")
    output_scale = 1 / (2**output_fixpos)
    n_of_images = len(img)
    count = 0
    while count < cnt:
        runSize = input_ndim[0]
        """prepare batch input/output """
        inputData = [np.empty(input_ndim, dtype=np.int8, order="C")]
        outputData = [np.empty(output_ndim, dtype=np.int8, order="C")]
        """init input image to input buffer """
        for j in range(runSize):
            imageRun = inputData[0]
            imageRun[j, ...] = img[(count + j) % n_of_images].reshape(input_ndim[1:])
        """run with batch """
        job_id = runner.execute_async(inputData, outputData)
        runner.wait(job_id)
        """softmax&TopK calculate with batch """
        """Benchmark DPU FPS performance over Vitis AI APIs execute_async() and wait() """
        """Uncomment the following code snippet to include softmax calculation for model’s end-to-end FPS evaluation """
        for j in range(runSize):
            softmax = CPUCalcSoftmax(outputData[0][j], pre_output_size, output_scale)
            pred(softmax, pre_output_size, "./words.txt")

        count = count + runSize

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




def main(argv):
    global threadnum

    listimage = os.listdir(calib_image_dir)
    threadAll = []
    threadnum = int(argv[1])
    i = 0
    global runTotall
    runTotall = len(listimage)

    g = xir.Graph.deserialize(argv[2])
    subgraphs = get_child_subgraph_dpu(g)
    assert len(subgraphs) == 1  # only one DPU kernel
    all_dpu_runners = []
    for i in range(int(threadnum)):
        all_dpu_runners.append(vart.Runner.create_runner(subgraphs[0], "run"))

    input_fixpos = all_dpu_runners[0].get_input_tensors()[0].get_attr("fix_point")
    input_scale = 2**input_fixpos
    """image list to be run """
    img = []
    for i in range(runTotall):
        path = os.path.join(calib_image_dir, listimage[i])
        img.append(preprocess_one_image_fn(path, input_scale))

    cnt = 360
    """run with batch """
    time_start = time.time()
    for i in range(int(threadnum)):
        t1 = threading.Thread(target=runEfficientnet_v2, args=(all_dpu_runners[i], img, cnt))
        threadAll.append(t1)

    for x in threadAll:
        x.start()

    for x in threadAll:
        x.join()

    del all_dpu_runners

    time_end = time.time()
    timetotal = time_end - time_start
    total_frames = cnt * int(threadnum)
    fps = float(total_frames / timetotal)
    print(
        "FPS=%.2f, total frames = %.2f , time=%.6f seconds"
        % (fps, total_frames, timetotal)
    )


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage : python3 efficientnet_v2.py <thread_number> <efficientnet_v2_model_file>")
    else:
        main(sys.argv)

# Project Write-Up: People Counter App

This is an Edge application which can work on a range of intel devices (from CPU's, GPU's to VPU's etc.). Optimised to work on low latency and bandwidth, this project uses the Intel Distribution of Open VINO Toolkit.

### Video Link to the project
I've recorded a video of working of the project on my local machine. 
[[Video Link](https://github.com/zed1025/person-counter-open-vino/blob/master/images/yt.jpg)](https://www.youtube.com/watch?v=ynGGSE9zy9c&feature=youtu.be)
- https://www.youtube.com/watch?v=ynGGSE9zy9c&feature=youtu.be

<hr>

## What are Custom Layers?
- The Intel Distribution of OpenVINO toolkit supports neural network model layers in multiple frameworks including TensorFlow, Caffe, MXNet, Kaldi and ONYX. **Custom layers are layers that are not included in the list of known layers.** If your topology contains any layers that are not in the list of known layers, the Model Optimizer classifies them as custom.
- Terminiology
  - layer: activation function used in the Neural Network. There can be multiple activation functions used. E.g. RELU, Tanh, sigmoid, etc.
  - Intermediate Representation(IR): The model optimizer in Open Vino converts models in different frameworks(Tensorflow, Caffe etc.) into a common representation that can be understood by all intel devices. IR is basically 2 files, a .bin(for weights and biases) and a .xml(the model architecture) 
- The Model Optimizer searches the list of known layers for each layer contained in the input model topology before building the model's internal representation, optimizing the model, and producing the Intermediate Representation files.
- If some unknown layers are found, error will be reported at which point we have to do something about the unsupported layers. Possible Solutions
  - Ignore the layers
  - Use custom layers
  - Use HETERO plugin

### Custom Layer Implementation
- [Source](https://docs.openvinotoolkit.org/latest/_docs_HOWTO_Custom_Layers_Guide.html)
- The following figure shows the basic processing steps for the Model Optimizer highlighting the two necessary custom layer extensions, the Custom Layer Extractor and the Custom Layer Operation.
[](https://github.com/zed1025/person-counter-open-vino/blob/master/images/MO_extensions_flow.png)
- The Model Optimizer first extracts information from the input model which includes the topology of the model layers along with parameters, input and output format, etc., for each layer. The model is then optimized from the various known characteristics of the layers, interconnects, and data flow which partly comes from the layer operation providing details including the shape of the output for each layer. Finally, the optimized model is output to the model IR files needed by the Inference Engine to run the model.
- The custom layer extensions needed by the Model Optimizer are:
  - _Custom Layer Extractor_: Responsible for identifying the custom layer operation and extracting the parameters for each instance of the custom layer. The layer parameters are stored per instance and used by the layer operation before finally appearing in the output IR.
  - _Custom Layer Operation_: Responsible for specifying the attributes that are supported by the custom layer and computing the output shape for each instance of the custom layer from its parameters.The **--mo-op** command-line argument generates a custom layer operation for the Model Optimizer.

### Custom Layer Extensions for the Inference Engine
- The following figure shows the basic flow for the Inference Engine highlighting two custom layer extensions for the CPU and GPU Plugins, the Custom Layer CPU extension and the Custom Layer GPU Extension.
[](https://github.com/zed1025/person-counter-open-vino/blob/master/images/IE_extensions_flow.png)
- Each device plugin includes a library of optimized implementations to execute known layer operations which must be extended to execute a custom layer. The custom layer extension is implemented according to the target device:
  - _Custom Layer CPU Extension_: A compiled shared library (.so or .dll binary) needed by the CPU Plugin for executing the custom layer on the CPU.
  - _Custom Layer GPU Extension_: OpenCL source code (.cl) for the custom layer kernel that will be compiled to execute on the GPU along with a layer description file (.xml) needed by the GPU Plugin for the custom layer kernel.

### Model Extension Generator
- Using answers to interactive questions or a *.json* configuration file, the Model Extension Generator tool generates template source code files for each of the extensions needed by the Model Optimizer and the Inference Engine. To complete the implementation of each extension, the template functions may need to be edited to fill-in details specific to the custom layer or the actual custom layer functionality itself.
- The Model Extension Generator is included in the Intel® Distribution of OpenVINO™ toolkit installation and is run using the command (here with the "--help" option): `python3 /opt/intel/openvino/deployment_tools/tools/extension_generator/extgen.py new --help`, where the output will appear similar to:
```
usage: You can use any combination of the following arguments:
Arguments to configure extension generation in the interactive mode:
optional arguments:
  -h, --help            show this help message and exit
  --mo-caffe-ext        generate a Model Optimizer Caffe* extractor
  --mo-mxnet-ext        generate a Model Optimizer MXNet* extractor
  --mo-tf-ext           generate a Model Optimizer TensorFlow* extractor
  --mo-op               generate a Model Optimizer operation
  --ie-cpu-ext          generate an Inference Engine CPU extension
  --ie-gpu-ext          generate an Inference Engine GPU extension
  --output_dir OUTPUT_DIR
                        set an output directory. If not specified, the current
                        directory is used by default.
``` 

### Some of the potential reasons for handling custom layers
- The research in the field of AI is very fastpaced. Every few month a new alogrithm is published. To meet this increasing demand, Custom Layers provide flexibility to the Open Vino Toolkit
- Some layers are very important and we cannot ignore them, during those times Custom Layers are very helpful. For e.g. in industry where apps are make for end users, every single layers, which can even add very slight performance improvement can be helpful

<hr>

## Comparing Model Performance

- I have used the pre-trained model from Open Vino Model Zoo, because the models that I converted were very poor at both accuracy and performance(inference times)
For comparing models I have used the following two metric
  - model size
  - inference time
- Model Size
| |ssd-mobilenet-v1-coco|ssd-mobilenet-v2-coco|faster-rcnn-inception-v2-coco|
|Before Conversion| 29.1 MB | 69.7 MB| 57.2 MB |
|After Conversion| 27.5 MB | 67.6 MB | 53.6 MB |
- Model Inference Time
| |ssd-mobilenet-v1-coco|ssd-mobilenet-v2-coco|faster-rcnn-inception-v2-coco|
|Before Conversion| 55 ms | 50 ms | 60 ms |
|After Conversion| 70 ms | 60 ms | 75 ms |
- Conclusion
  - The results for Model Size convey to us that, the model size do not vary by a lot before and after conversion. Open Vino does a very good job at handling that. Also all the three models selected by me have around the same size, so choosing any of them will be okay.
  - The inference times however have some variance. Like if the application is critical then that small performance increase can help a lot. I can draw out the following conclusions
    - If inference time is not important, but accuracy is go with faster_rcnn
    - If inference time matters then mobile net are the way to go, because, ssd_mobile nets may not have the best accuracy but they are fast.
    - In the people counter app both speed and accuracy are important, so I had difficults choosing one over the other. So if I could not get them to be more optimised I will go with the pre-trained models from Intel.


## Assess Model Use Cases

The following are some of the use cases of the people-counter-app
1. Attendence Counting
- In seminars, events, etc it can be used to count the number of people in attendence in various different events. 
- E.g. it could be used at Google IO to count people that attend different workshops, which can give an insight into what people are more interested in
2. Supermarkets
- Stores could count the number of people that visit a perticular section of the store, and make effors to increase their sales
- It can also help in setting up billing counters and staff in that particular area to prevent theft/loss, while at the same time making the shopping experience more convineant
3. During Elections
- To automate voter turnout
- With a good face detection app, we could use the combination to automatically get the names of people who came to vote


## Assess Effects on End User Needs

Lighting, model accuracy, and camera focal length/image size have different effects on a
deployed edge model. The potential effects of each of these are as follows...
- Poor lighting can decrease model accuracy significantly. However we can mitigate this using 
  - Better hardware
  - identify the areas that are working poorly and take steps to improve lighting in those areas
- Failed camera/focal length/image size can badly impact the model accuracy and predicting. One of the things that we can do to mitigate this is use a voting system
  - rather than having a single camera, we could deploy 3(or more) cameras, and 3(or more) edge applications at the same location. 
  - We let all of them perform inference and then we vote on the outputs of all the apps
  - While this will increase cost, it will greatly reduce the effects of hardware failure
- Decrease in model accuracy
  - If model accuracy decreases over time, it maybe because the data that was used to train the model was somewhat biased
  - Potential solutions that I can think of
    - Retrain the model with more data that is new
    - Adjust the parameters

## Model Research


In investigating potential people counter models, I tried each of the following three models:

- Model 1: [SSD Mobilenet V1 Coco]
  - [Model Source](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2018_01_28.tar.gz)
  - Downloading the model
    - `wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2018_01_28.tar.gz`
    - unzipping the model
      - `tar -xvf ssd_mobilenet_v1_coco_2018_01_28.tar.gz`
  - I converted the model to an Intermediate Representation with the following arguments
  `python3 /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --transformations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json`
  - The model was insufficient for the app because
    - The model does not detect people when they are stationary in the frame
    - The **average inference time** is about 10-25ms more than the intel model, about **~60ms**
    - The model does very poorly at detecting people when they are facing backward
    - The model sometimes counts the same person more than once
    [](images/ssd_v1.png)
  - I tried to improve the model for the app by
    - tried using a differnt precision than the default but it didnt make much difference
  
- Model 2: [SSD Mobilenet V2 Coco]
  - [Model Source](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz)
    - Downloading the model
    - `wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz`
    - unzipping the model
      - `tar -xvf ssd_mobilenet_v2_coco_2018_03_29.tar.gz`
  - I converted the model to an Intermediate Representation with the following arguments
  `python3 /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --transformations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json`
  - The model was insufficient for the app because
    - average inference time ~70ms
    - it performs poorly at inference. On some instance it works perfectly but most times it stops detecting a person in the frame
    [](images/ssd-v2-not-detecting.png)
    [](images/ssd-v2-wrong-infer.png)
  - I tried to improve the model for the app by
    - tried using a differnt precision than the default but it didnt make much difference

- Model 3: [Faster RCNN Inception V2 Coco]
  - [Model Source](http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz)
    - Downloading the model
    - `wget http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz`
    - unzipping the model
      - `tar -xvf faster_rcnn_inception_v2_coco_2018_01_28.tar.gz`
  - I converted the model to an Intermediate Representation with the following arguments
    - For faster_rcnn_inception_v2_coco, i've used the `faster_rcnn_support.json` file
    `python3 /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --transformations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/faster_rcnn_support.json`
  - The model was insufficient for the app because
    - While this model has pretty accurate inference but it takes a lot of time for inference which makes it not suitable for use at Edge. 
    - Moreover this model had very large size.
  - I tried to improve the model for the app by
    - tried using a differnt precision than the default but it didnt make much difference


## Model Used: person-detection-retail-0013

- Downloading the model
  `python3 /opt/intel/openvino/deployment_tools/open_model_zoo/tools/downloader/downloader.py --name person-detection-retail-0013 -o ~/dev/ov_workspace/project1/model`
  - No need to unzip the model, as you directly get the .xml and .bin file
- Running the App
    - In the first terminal window
        - `cd` into `webservice/server/node-server`
        - run `node ./server.js`
    - In the second terminal window
        - `cd` into `webservice/ui`
        - run `npm run dev`
    - In the third terminal window
        - run `sudo ffserver -f ./ffmpeg/server.conf`
    - In the fourth window 
        - run `python main.py -i resources/Pedestrian_Detect_2_1_1.mp4 -m /home/workspace/model/person-detection-retail-0013.xml -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU -pt 0.6 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm`
    - Then use the **Open App** button on the Guide page

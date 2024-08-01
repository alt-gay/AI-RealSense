# AI-Realsense
This GitHub repository contains a ROS2 package written in Python for the integration of AI and the Intel Realsense Camera. Please see below for more information on each of the Python files found within the camera_pkg.

## LLM Request
The purpose of the **llm_request** file is to extract data from user input and publish it to the relevant topics with the help of the **Llama3-Instruct** model. The 8 billion parameter version of the model is used, and it has been quantised and loaded into 4 bits. This file takes in the user's prompt from the command-line interface, and leverages on the LLM to extract the object to find, and the location to find it at. The object and room requests are published as strings to the /obj_req and /room_req topics respectively.

**Please complete the HuggingFace CLI login using your API key after gaining access to the [Model](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) through Meta to run this file!**

## Intel Publisher
The **intelpub** file essentially provides access to the Intel Realsense Camera feed and publishes the RGB frames to the /rgb_camera, for the VLM to access. The interval between each frame being published is 2 seconds. Furthermore, it also subscribes to the /depth_req topic (which is a float array containing the coordinates of the pixel in the centre of the detected object), and finds the depth of that pixel through the camera's depth frame. The depth found is then published to the /obj_dist topic, and is the distance between the Intel Realsense Camera & the detected object.

## Intel Subscriber (With Bounding Box)
The Vision Language Model (VLM) is contained within this intelsubb file. The VLM used is 

### Further Details
This [document](https://docs.google.com/document/d/1mzZNagqpMO3bk69fnhcjVuhvVGLbYOjWB5l5n0_Rkvo/edit?usp=sharing) contains more information on the usage of the camera_pkg. Do take a look for more clarity ✮ツ

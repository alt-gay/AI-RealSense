# AI-RealSense

Welcome to the AI-RealSense GitHub repository! This project features a ROS2 package, developed in Python, designed to integrate AI with the Intel RealSense Camera. Below, you'll find detailed information about each Python file within the `camera_pkg`.

## LLM Request

The **llm_request** file extracts user input data and publishes it to the appropriate topics using the **Llama3-Instruct** model. We utilise the 8 billion parameter version of this model, which has been quantised to 4 bits. This file processes prompts from the command-line interface to identify the object and its location. These requests are published as strings to the `/obj_req` and `/room_req` topics, respectively.

**Ensure that you complete the HuggingFace CLI login with your API key after accessing the [Model](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) via Meta to run this file!**

## Intel Publisher

The **intelpub** file manages access to the Intel RealSense Camera feed, publishing RGB frames to the `/rgb_camera` topic every 2 seconds for the VLM's use. It also subscribes to the `/depth_req` topic, which contains float array coordinates for the pixel at the center of a detected object. The file calculates the depth of this pixel using the camera's depth frame and publishes this distance to the `/obj_dist` topic, representing the space between the Intel RealSense Camera and the detected object.

## Intel Subscriber (With Bounding Box)

The **intelsubb** file houses the Vision Language Model (VLM) component, specifically [**OWL-ViT**](https://huggingface.co/docs/transformers/en/model_doc/owlvit), accessible via HuggingFace. This script monitors the `/obj_req` topic and the RGB camera feed (`/rgb_frame` topic), checking each new image (published every 2 seconds) for the specified object. The detection status is sent as a Boolean to the `/det_status` topic. If an object is detected, OWL-ViT determines its location in the image, and a bounding box is drawn around it. This annotated image is then published to `/bb_img`, and the midpoint coordinates of the bounding box are sent to the `/depth_req` topic.

Note that OWL-ViT does not handle bounding box drawing, so there may be a short delay before the bounding box image is published.

## Intel Subscriber (Without Bounding Box)

For testing purposes, the **intelsub** file is provided without bounding box capabilities. It includes all features of the `intelsubb` file except for bounding box drawing and distance publishing, allowing for faster testing without delays.

### Additional Information

For more details on how to use the `camera_pkg`, please refer to this [document](https://docs.google.com/document/d/1mzZNagqpMO3bk69fnhcjVuhvVGLbYOjWB5l5n0_Rkvo/edit?usp=sharing). Happy exploring! ✮ツ

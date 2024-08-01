import rclpy
from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import Image
from std_msgs.msg import String, Bool
from cv_bridge import CvBridge
from PIL import Image as PILImage
from PIL import ImageDraw
import cv2
import numpy as np
import torch
from transformers import AutoProcessor, Owlv2ForObjectDetection
from transformers.utils.constants import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD
from std_msgs.msg import Int32MultiArray

#################### PUBLISHES BOUNDING BOX IMAGE ####################

class IntelSubscriber(Node):
    def __init__(self):
        super().__init__("intel_subscriber")
        # Subscribe to RGB Camera Feed & Object Request Topics
        self.subscription1 = self.create_subscription(
            Image, "rgb_frame", self.rgb_frame_callback, 1, callback_group=MutuallyExclusiveCallbackGroup())
        self.subscription2 = self.create_subscription(String, 'obj_req', self.get_object, 1, callback_group=MutuallyExclusiveCallbackGroup())
        # Create Publisher for Detection Status, Bounding Box Image, and Location of Object
        self.pub_ = self.create_publisher(Bool, "det_status", 1)
        self.pub_2 = self.create_publisher(Image, "bb_img", 1)
        self.pub_3 = self.create_publisher(Int32MultiArray, "depth_req", 1)
        # Set up OWL-ViT Model
        self.processor = AutoProcessor.from_pretrained("google/owlv2-base-patch16-ensemble")
        self.model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble")
        self.item_find = None # Placeholder to avoid initialisation error
        self.br_rgb = CvBridge() # Create CV Bridge Object
        self.frame_queue = []  # Buffer for batch processing
        self.processing = False

    def get_object(self, message):
        self.item_find = message.data
        self.get_logger().info(f'Object to Find: {self.item_find}') # Publish Object to Find for Confirmation

    def get_preprocessed_image(self, pixel_values): # Speeds up image processing
        pixel_values = pixel_values.squeeze().numpy()
        unnormalized_image = (pixel_values * np.array(OPENAI_CLIP_STD)[:, None, None]) + np.array(OPENAI_CLIP_MEAN)[:, None, None]
        unnormalized_image = (unnormalized_image * 255).astype(np.uint8)
        unnormalized_image = np.moveaxis(unnormalized_image, 0, -1)
        return PILImage.fromarray(unnormalized_image)

    def rgb_frame_callback(self, data):
        self.get_logger().warning("Receiving RGB frame...")
        image1 = self.br_rgb.imgmsg_to_cv2(data) # Capture Image Frame
        h, w, _ = image1.shape  # Get dimensions (height, width)
        # Convert OpenCV Image to Regular Image for Processing
        color_coverted = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
        pil_image = PILImage.fromarray(color_coverted)
        self.frame_queue.append((pil_image, h, w))  # Queue frames for batch processing
        if not self.processing:
            self.process_next_frame()

    def process_next_frame(self):
        if self.frame_queue:
            self.processing = True
            pil_image, h, w = self.frame_queue.pop(0)  # Get next frame from queue
            
            if self.item_find is not None:
                msg = Bool() # For Object Detection Status
                msg.data = False
                msg2 = Int32MultiArray()
                texts = [[self.item_find]]
                # Input to OWL-ViT
                inputs = self.processor(text=texts, images=pil_image, return_tensors="pt")
                outputs = self.model(**inputs)
                unnormalized_image = self.get_preprocessed_image(inputs.pixel_values)
                target_sizes = torch.Tensor([unnormalized_image.size[::-1]])
                results = self.processor.post_process_object_detection(outputs=outputs, threshold=0.2, target_sizes=target_sizes)
                texts = texts[0]
                boxes, scores, labels = results[0]["boxes"], results[0]["scores"], results[0]["labels"]
                # Draw Bounding Box on Image
                draw = ImageDraw.Draw(unnormalized_image) 
                for box, score, label in zip(boxes, scores, labels):
                    xmin, ymin, xmax, ymax = box
                    midx = int((xmin+xmax)/2)
                    midy = int((ymin+ymax)/2)
                    midpt = [midx, midy]
                    msg2.data = midpt # Find and Publish Midpoint 
                    self.pub_3.publish(msg2)
                    draw.rectangle((xmin, ymin, xmax, ymax), outline="red", width=1)
                    draw.text((xmin, ymin), f"{texts[label.item()]}: {round(score.item(), 2)}", fill="white")
                    box = [round(i, 2) for i in box.tolist()]
                    self.get_logger().info(f"Detected {texts[label.item()]} with confidence {round(score.item(), 3)} at location {box}")
                    # If item is found, publish True status and exit
                    if self.item_find in texts[label.item()]:
                        self.get_logger().info(f"Found {self.item_find}")
                        msg.data = True
                        break
                
                # Crop and resize the image
                unnormalized_image = unnormalized_image.crop((0, 0, 960, h * 960 // w))
                unnormalized_image = unnormalized_image.resize((w, h))

                # Convert PIL Image to ROS Image
                ros_image = self.br_rgb.cv2_to_imgmsg(np.array(unnormalized_image), encoding="rgb8")
                self.pub_2.publish(ros_image)
                self.pub_.publish(msg)
                
                self.processing = False
                self.process_next_frame()  # Process next frame in queue
            else:
                self.get_logger().info("Waiting for user input...") # Nothing has been published to /obj_req topic yet

def main(args=None):
    rclpy.init(args=args)
    intel_subscriber = IntelSubscriber()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(intel_subscriber)
    try:
        intel_subscriber.get_logger().info('Beginning client, shut down with CTRL-C')
        executor.spin()
    except KeyboardInterrupt:
        intel_subscriber.get_logger().info('Keyboard interrupt, shutting down.\n')
        
    intel_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()

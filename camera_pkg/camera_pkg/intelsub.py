import rclpy
from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
from std_msgs.msg import Bool
import cv2
from transformers import OwlViTProcessor, OwlViTForObjectDetection
from PIL import Image as img
import torch

#################### NO BOUNDING BOX IMAGE ####################

class IntelSubscriber(Node):
    def __init__(self):
        super().__init__("intel_subscriber")
        # Subscribe to RGB Camera Feed & Object Request Topics
        self.subscription1 = self.create_subscription(
            Image, "rgb_frame", self.rgb_frame_callback, 10, callback_group=MutuallyExclusiveCallbackGroup())
        self.subscription2 = self.create_subscription(String, 'obj_req', self.get_object, 10, callback_group = MutuallyExclusiveCallbackGroup())
        # Create Publisher for Detection Status
        self.pub_ = self.create_publisher(Bool, "det_status", 10)
        # Set Up OWL-ViT Model
        self.processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
        self.model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")
        self.item_find = None # Placeholder to avoid initialisation error
        
    def get_object(self, message):
        self.item_find = message.data
        strg = 'Object to Find: ' + self.item_find 
        self.get_logger().info(strg) # Publish Object to Find for Confirmation
    
    def rgb_frame_callback(self, data):
        self.get_logger().warning("Receiving RGB frame...")
        # Create CV Bridge Object & Capture Frame
        self.br_rgb = CvBridge()
        image1 = self.br_rgb.imgmsg_to_cv2(data)
        # Convert OpenCV Image to Regular Image for Processing
        color_coverted = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB) 
        pil_image = img.fromarray(color_coverted) 
        if self.item_find != None:
            msg = Bool() # For Object Detection Status
            msg.data = False
            texts = [[self.item_find]]
            # Input to OWL-ViT
            inputs = self.processor(text=texts, images=pil_image, return_tensors="pt")
            outputs = self.model(**inputs)
            target_sizes = torch.Tensor([pil_image.size[::-1]])
            self.results = self.processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=0.1)
            self.text = texts[0]
            boxes, scores, labels = self.results[0]["boxes"], self.results[0]["scores"], self.results[0]["labels"]
            for box, score, label in zip(boxes, scores, labels):
                box = [round(i, 2) for i in box.tolist()]
                self.get_logger().info(f"Detected {self.text[label]} with confidence {round(score.item(), 3)} at location {box}")
                # If item is found, publish True status and exit
                if self.item_find in self.text[label]:
                    str = 'Found ' + self.item_find
                    self.get_logger().info(str)
                    msg.data = True
                    break
            self.pub_.publish(msg)
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

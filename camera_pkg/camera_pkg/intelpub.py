import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import pyrealsense2 as rs
import numpy as np
from std_msgs.msg import Int32MultiArray, Float32
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup


class IntelPublisher(Node):
    def __init__(self):
        super().__init__("intel_publisher")
        # Create Publishers for Detection Status and Depth Request
        self.intel_publisher_rgb = self.create_publisher(
            Image, "rgb_frame", 10)
        self.depth_pub = self.create_publisher(
            Float32, "obj_dist", 10)
        timer_period = 2 # Interval between Images Published (in seconds)
        self.br_rgb = CvBridge() # Create CV Bridge Object
        self.i = 0 # Initialise Counter for Frames
        # Subscribe to Object Location Topic
        self.subscription2 = self.create_subscription(Int32MultiArray, 'depth_req', self.get_distance, 1, callback_group=MutuallyExclusiveCallbackGroup())
        # Start RGB & Depth Camera Streams
        try:
            self.pipe = rs.pipeline()
            self.cfg = rs.config()
            self.cfg.enable_stream(rs.stream.color, 848,
                                   480, rs.format.bgr8, 30)
            self.cfg.enable_stream(rs.stream.depth, 848,
                                   480, rs.format.z16, 30)
            self.pipe.start(self.cfg)
            self.timer = self.create_timer(timer_period, self.timer_callback, callback_group=MutuallyExclusiveCallbackGroup())
        except Exception as e:
            print(e)
            self.get_logger().error("INTEL REALSENSE IS NOT CONNECTED")

    def timer_callback(self):
        # Wait for frames to become available
        frames = self.pipe.wait_for_frames()
        color_frame = frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())
        self.intel_publisher_rgb.publish(
            self.br_rgb.cv2_to_imgmsg(color_image)) # Convert OpenCV Image to Regular Image & Publish
        self.get_logger().info('Publishing RGB Frame %d' % self.i)
        self.i += 1
    
    def get_distance(self, message):
        align_to = rs.stream.depth # Alignment to RGB Frame to avoid errors
        align = rs.align(align_to)
        msg = Float32()
        midpt = message.data # Receive Coordinate of Object Midpoint
        x = midpt[0]
        y = midpt[1]
        self.get_logger().info(f"Received coordinates: x={x}, y={y}")
        frames = self.pipe.wait_for_frames()
        aligned_frames = align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        zDepth = depth_frame.get_distance(int(x),int(y)) # Get depth of midpoint pixel
        msg.data = zDepth
        self.depth_pub.publish(msg) # Publish distance between object & camera


def main(args=None):
    rclpy.init(args=None)
    intel_publisher = IntelPublisher()
    rclpy.spin(intel_publisher)
    intel_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

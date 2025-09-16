#!/usr/bin/env python3
import rclpy
import rclpy

from sensor_msgs.msg import Image

class ObjectDetector(Node):
    def __init__(self):
        super().__init__('object_detector')

        self.subscription = self.create_subscription(
            Image,
            '/camera/image_rect',
            self.image_callback,
            10)
        self.subscription

    def subscription_callback(self, msg):
        pass

def main():
    rclpy.init()
    node = ObjectDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()

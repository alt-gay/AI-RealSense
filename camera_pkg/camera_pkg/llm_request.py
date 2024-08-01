import rclpy
from rclpy.node import Node
import torch
import torch.nn as nn
from transformers import AutoTokenizer,  BitsAndBytesConfig, AutoModelForCausalLM
import re
import os
from std_msgs.msg import String

#################### SET UP HUGGINGFACE CLI LOGIN BEFORE USE ####################

class LLMPublisher(Node):

    def __init__(self):
        super().__init__('llm_publisher')
        torch.cuda.empty_cache() # Clear cache
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        # Quantise model & load in 4-bits
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_t=True,
            llm_int8_enable_fp32_cpu_offload=True
        )
        # Set Up Llama3-Instruct Model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Meta-Llama-3-8B-Instruct",
            device_map = 'auto',
            quantization_config=bnb_config,
            torch_dtype = torch.bfloat16
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Meta-Llama-3-8B-Instruct")
        # Create Publisher for Room & Object Requests
        self.publisher1_ = self.create_publisher(String, 'room_req', 10)
        self.publisher2_ = self.create_publisher(String, 'obj_req', 10)
        timer_period = 0.5  # Interval for publishing data (in seconds)
        self.get_data() # Run LLM & get user input
        self.extrapolate_data() # Extract relevant information
        self.timer1 = self.create_timer(timer_period, self.room_callback)
        self.timer2 = self.create_timer(timer_period, self.obj_callback)
    
    def get_data(self):
        self.command = input('Please Enter your Command: ')
        # Adjust the prompt below for grounding to specific locations!
        self.prompt = '<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a means to extract information to be fed to a robot, so keep your answers short. Your task is to generate the place the robot is to go to and the object it is to find in that place based on the given prompt. The available rooms are: bedroom, kitchen, living room. If you are told to head to rooms other than those previously mentioned as available, just return "Error" and nothing else. <|eot_id|><|start_header_id|>user<|end_header_id|> Go to the kitchen and find a microwave. <|eot_id|><|start_header_id|>assistant<|end_header_id|>\nPlace: kitchen\nObject: microwave\n<|eot_id|><|start_header_id|>user<|end_header_id|>' + ' ' +self.command + ' r' + '<|eot_id|><|start_header_id|>assistant<|end_header_id|>'
        self.inputs = self.tokenizer(self.prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            self.outputs = self.model.generate(**self.inputs, max_new_tokens=50,
                                    temperature=0.00001, pad_token_id=self.tokenizer.eos_token_id)
        self.response = self.tokenizer.decode(self.outputs[0], skip_special_tokens=True)
    
    def extrapolate_data(self):
        parts = self.response.split('rassistant')

        # Check if there are parts after 'rassistant'
        if len(parts) > 1:
            # Get the last part after 'rassistant'
            last_part = parts[-1].strip()
            
            # Use regex to extract Place and Object
            place_match = re.search(r'Place:\s*([^\n]*)', last_part)
            object_match = re.search(r'Object:\s*([^\n]*)', last_part)

            if place_match and object_match:
                self.place1 = place_match.group(1).strip()
                self.object1 = object_match.group(1).strip()
            else:
                self.get_logger().info('Error')
        else:
            self.get_logger().info('Error')

    def room_callback(self):
        msg = String()
        msg.data = self.place1
        self.publisher1_.publish(msg)
        self.get_logger().info('Publishing: "%s"' % msg.data)
    
    def obj_callback(self):
        msg = String()
        msg.data = self.object1
        self.publisher2_.publish(msg)
        self.get_logger().info('Publishing: "%s"' % msg.data)

def main(args=None):
    rclpy.init(args=args)
    llm_publisher = LLMPublisher()
    rclpy.spin(llm_publisher)
    llm_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
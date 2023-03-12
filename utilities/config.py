import sys
sys.path.insert(0, '/home/pallav/yolo/object_detection_v1')

from utilities import argsParser

args = argsParser.parse_args()

class Config:
    def __init__(self, config_file_path):
        self.config_file_path = config_file_path
    
    def get_config(self):
        return argsParser.parse_config(self.config_file_path)

def get_objDetectConfig():
    config_obj = Config(args.object_detection)
    config_string = config_obj.get_config()
    return config_string



if __name__ == '__main__':
    config_string = get_objDetectConfig()
    print(config_string.blob_size)
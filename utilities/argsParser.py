import argparse
import logging
from utilities import utility

logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--object_detection',
                        type=str,
                        default='/home/pallav/yolo/object_detection_v1/config/objDetectConfig.yaml',
                        help='path to object detection config')

    args = parser.parse_args('')
    return args

def parse_config(cfg_file):
    import yaml
    with open(cfg_file, 'r') as fopen:
        yaml_config = utility.AttrDict(yaml.load(fopen, Loader=yaml.Loader))
    return yaml_config
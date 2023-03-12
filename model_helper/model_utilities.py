import sys
sys.path.insert(0, '/home/pallav/yolo/object_detection_v1')

import uuid
import string
import secrets
from datetime import datetime


class ModelUtilities():
    def __init__(self):
        pass

    def generate_child_ids(self, child_model_count, model_dict):
        print(" -- Generating Child Id -- ")
        # print(model_dict)
        for c_id in range(child_model_count):
            print(c_id, " Child Model -->> ", model_dict['child_models'][c_id])
            alphabet = string.ascii_letters + string.digits
            id = ''.join(secrets.choice(alphabet) for i in range(10))
            model_dict['child_models'][c_id]['child_id'] = id
        return model_dict

    def register_model(self, model_dict):
        try:
            unique_id = str(uuid.uuid4())
            model_dict['parent_model_id'] = unique_id
            model_dict['created_at'] = datetime.now()

            child_model_count = len(model_dict['child_models'])
            if child_model_count == 0:
                print('No child model found')
            else:
                print(f"Generating ids for {child_model_count} child models")
                model_dict = self.generate_child_ids(child_model_count, model_dict)

            return model_dict['parent_model_id']
        except Exception as e:
            print(e)
            return None


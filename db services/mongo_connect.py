import sys
sys.path.insert(0, '/home/pallav/yolo/object_detection_v1')

import pymongo
from pymongo import MongoClient

class mongo_connect():
    def __init__(self):
        pass

    def make_connection(self):
        try:
            client = MongoClient('mongodb://localhost:27017')
            return client
        except Exception as e:
            print(e)
            exit()

    




# if __name__ == '__main__':
#     mongo_obj = mongo_connect()
#     client = mongo_obj.make_connection()
#     print(client.list_database_names())
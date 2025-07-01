import os, sys
sys.path.append("./")
sys.path.append("../")


from config.logger import logger
from config.datasets.dataset_instance import mal_api

logger.is_debug(True)



if __name__ == '__main__':
    mal_api.load()
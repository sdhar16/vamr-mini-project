import os
import time
from pipeline import Pipeline
from helpers import helpers

CONFIG_PATH = "src/main/configs/Parking.yaml"
#CONFIG_PATH = "src/main/configs/malaga.yaml"
#CONFIG_PATH = "src/main/configs/KITTI.yaml" 

CONFIG_PATH = os.environ["CONFIG_PATH"]

h = helpers()
config = h.read_yaml(CONFIG_PATH)
config = config["config"]

image_path = config["images_path"]
K_path = config["K_path"]

def main():
    start_time = time.time()

    current_path = os.getcwd()

    img_dir = os.path.join(current_path, image_path)
    K_file = os.path.join(current_path, K_path)

    pipeline = Pipeline(img_dir, K_file, config)
    pipeline.run()

    print(f"runtime in seconds: {time.time() - start_time}")



if __name__ == "__main__":
    main()
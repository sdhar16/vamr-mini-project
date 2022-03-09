from helpers import helpers
from Initialization import Initialization
from Continuous import Continuous

class Pipeline:
    def __init__(self, img_dir, K_file, config):
        self.h = helpers()
        self.config = config
        should_resize = config["resize"]
        self.images = self.h.loadImages(img_dir, should_resize)
        self.K = self.h.load_poses(K_file)
    
    def run(self):
        baseline = self.config["basline_images"]
        
        initialise_vo = Initialization(self.images[baseline[0]], self.images[baseline[1]], self.K, self.config["Initialization"])
        keypoints, landmarks, T = initialise_vo.run()
        
        continuous_vo = Continuous(keypoints, landmarks, T, self.images, self.K, self.config["Continuous"], baseline)
        continuous_vo.run()

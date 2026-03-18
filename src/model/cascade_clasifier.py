import numpy as np

from src.data.filter import (
    get_integral_image, 
    get_integral_squared_image,
    get_integral_sum,
    get_std_from_integral_images
)
from src.data.crops import get_all_image_crops
from .haar_cascade_parser import HaarCascade


class CascadeClassifier:
    def __init__(self, CONFIG, cascade: HaarCascade):
        self.CONFIG = CONFIG
        self.cascade = cascade

    def predict(self, img=None, img_path=None):
        assert img is not None or img_path is not None, "Either img or img_path must be provided"

        print("Getting image crops...")
        crops = get_all_image_crops(
            self.CONFIG, 
            img=img,
            img_path=img_path
        )

        print("Predicting faces...")
        faces = [
            crop
            for crop in crops
            if self._predict_crop(crop["img"])
        ]

        return faces
    
    def _predict_crop(self, crop_img):
        # Calculate integral images
        integral_img = get_integral_image(crop_img)
        integral_img_2 = get_integral_squared_image(crop_img)

        # Calculate std 
        H, W = crop_img.shape
        r1 = np.array([[0]])
        r2 = np.array([[H - 1]])
        c1 = np.array([[0]])
        c2 = np.array([[W - 1]])
        
        _, std_dev = get_std_from_integral_images(integral_img, integral_img_2, r1, r2, c1, c2)
        std_dev = std_dev[0, 0] 
        
        if std_dev <= 0:
            std_dev = 1.0

        for stage in self.cascade.stages:
            if not self._predict_stage(integral_img, stage, std_dev):
                return False
        
        return True

    def _predict_stage(self, integral_img, stage, std_dev=1.0):
        total_sum = 0.0
        for clf in stage.weak_classifiers:
            feature_val = self._compute_feature(integral_img, clf.feature)

            norm_thrs = clf.threshold * std_dev
            total_sum += (
                clf.left_value 
                if feature_val >= norm_thrs 
                else clf.right_value
            )

        return total_sum >= stage.threshold
    
    def _compute_feature(self, integral_img, feature):
        feature_sum = 0.0
        for rec in feature.rectangles:
            rec_sum = get_integral_sum(
                integral_img, 
                rec.x, rec.y, 
                rec.x+rec.width-1, rec.y+rec.height-1
            )
            feature_sum += rec_sum * rec.weight
    
        return feature_sum
    
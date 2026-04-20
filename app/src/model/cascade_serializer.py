"""
Serialization utilities for Haar Cascade classifiers.
Handles saving to and loading from OpenCV-compatible XML format.
"""

import os
import pickle
import numpy as np
from pathlib import Path
from xml.dom import minidom
import xml.etree.ElementTree as ET

from .cascade_def import HaarCascade
from .cascade_parser import HaarCascadeParser, build_haar_cascade_from_stages

from maikol_utils.print_utils import print_color, print_warn

class CascadeStage:
    """Simple wrapper to convert (clf, threshold) tuples to Stage objects."""
    def __init__(self, clf, threshold):
        self.clf = clf
        self.threshold = threshold
        self.weak_classifiers = []  # Empty for intermediate saves
    
    def __repr__(self):
        return f"CascadeStage(threshold={self.threshold}, estimators={len(self.clf.estimators_)})"


class CascadeSerializer:
    """Handles serialization/deserialization of HaarCascade objects to/from XML."""
    
    @staticmethod
    def save(cascade, output_path: str) -> None:
        """
        Save a trained HaarCascade object to XML in OpenCV format.
        
        Args:
            cascade: HaarCascade object to save
            output_path: Path to save the XML file
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Create root element
        opencv_storage = ET.Element('opencv_storage')
        
        # Create cascade element
        cascade_elem = ET.SubElement(opencv_storage, 'haarcascade_frontalface_default')
        cascade_elem.set('type_id', 'opencv-haar-classifier')
        
        # Add cascade properties
        size_elem = ET.SubElement(cascade_elem, 'size')
        size_elem.text = f'{cascade.width} {cascade.height}'
        
        # Add stages
        stages_elem = ET.SubElement(cascade_elem, 'stages')
        
        for stage_idx, stage in enumerate(cascade.stages):
            stage_elem = ET.SubElement(stages_elem, '_')
            
            # Add weak classifiers as trees
            trees_elem = ET.SubElement(stage_elem, 'trees')
            
            for weak_clf in stage.weak_classifiers:
                tree_elem = ET.SubElement(trees_elem, '_')
                root_elem = ET.SubElement(tree_elem, '_')

                feature = weak_clf.feature
                if feature is None:
                    raise ValueError("Weak classifier is missing its feature definition")

                feature_elem = ET.SubElement(root_elem, 'feature')
                rects_elem = ET.SubElement(feature_elem, 'rects')
                for rect in feature.rectangles:
                    rect_text = (
                        f'{int(rect.x)} {int(rect.y)} '
                        f'{int(rect.width)} {int(rect.height)} {rect.weight}'
                    )
                    rect_node = ET.SubElement(rects_elem, '_')
                    rect_node.text = rect_text

                tilted_elem = ET.SubElement(feature_elem, 'tilted')
                tilted_elem.text = '0'

                threshold_elem = ET.SubElement(root_elem, 'threshold')
                threshold_elem.text = str(float(weak_clf.threshold))

                left_val_elem = ET.SubElement(root_elem, 'left_val')
                left_val_elem.text = str(float(weak_clf.left_value))

                right_val_elem = ET.SubElement(root_elem, 'right_val')
                right_val_elem.text = str(float(weak_clf.right_value))
            
            # Stage threshold
            threshold_elem = ET.SubElement(stage_elem, 'stage_threshold')
            threshold_elem.text = str(float(stage.threshold))
            
            # Parent and next (for cascade chain)
            parent_elem = ET.SubElement(stage_elem, 'parent')
            parent_elem.text = str(stage_idx - 1)
            
            next_elem = ET.SubElement(stage_elem, 'next')
            next_elem.text = '-1'
        
        # Pretty print and save
        tree_str = minidom.parseString(ET.tostring(opencv_storage)).toprettyxml(indent='  ')
        # Remove XML declaration and extra blank lines
        tree_str = '\n'.join([line for line in tree_str.split('\n') 
                             if line.strip() and not line.startswith('<?xml')])
        
        with open(output_path, 'w') as f:
            f.write('<?xml version="1.0"?>\n')
            f.write(tree_str)
        
        print(f" - Saved Haar cascade to: {output_path}")
    
    @staticmethod
    def load(xml_path: str):
        """
        Load a Haar cascade from XML file (compatible with OpenCV format).
        
        Args:
            xml_path: Path to XML file
            
        Returns:
            HaarCascade object
        """
        parser = HaarCascadeParser(xml_path)
        cascade = parser.parse()
        print(f"✓ Loaded Haar cascade from: {xml_path}")
        return cascade


def save_stages(CONFIG, stages, stage_num, fpr_macro, all_features):
    """
    Save the current stages to XML files in OpenCV format.
    
    Args:
        CONFIG: Configuration object
        stages: List of trained stages as tuples (clf, threshold)
        stage_num: Current stage number (for naming)
        fpr_macro: Current macro false positive rate (for naming)
        all_features: Full list of generated Haar features
    """
    filename = f'haar_cascade_stage_{stage_num}_fpr_{fpr_macro:.4f}.xml'
    output_path = os.path.join(CONFIG.computed_haar_cascades, filename)

    cascade = build_haar_cascade_from_stages(
        stages_output=stages,
        all_features=all_features,
        width=CONFIG.crop_size,
        height=CONFIG.crop_size,
        cascade_type="trained_adaboost_stages",
        feature_type="HAAR",
    )
    CascadeSerializer.save(cascade, output_path)


def _checkpoint_path(CONFIG):
    return os.path.join(CONFIG.computed_haar_cascades, "stages_checkpoint.pkl")


def save_stage_checkpoint(CONFIG, checkpoint):
    path = _checkpoint_path(CONFIG)
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "wb") as f:
        pickle.dump(checkpoint, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f" - Saved stage checkpoint to: {path}")
    return path


def load_stage_checkpoint(CONFIG):
    path = Path(_checkpoint_path(CONFIG))
    if not path.exists():
        return None
    return pickle.loads(path.read_bytes())

def resume_training_from_checkpoint(CONFIG, X_train_faces):
    stages = []
    n_features = X_train_faces.shape[1]
    prev_fp = np.empty((0, n_features), dtype=np.float32) # Start with no hard negatives
    prev_n_faces = len(X_train_faces)
    n_bg_pre = len(X_train_faces)
    fpr_macro_thr = 1.0
    start_stage = 0

    if CONFIG.resume_training:
        checkpoint = load_stage_checkpoint(CONFIG)
        if checkpoint:
            stages = checkpoint.get("stages", [])
            fpr_macro_thr = checkpoint.get("fpr_macro_thr", fpr_macro_thr)
            prev_n_faces = checkpoint.get("prev_n_faces", prev_n_faces)
            n_bg_pre = checkpoint.get("n_bg_pre", n_bg_pre)
            prev_fp = checkpoint.get("prev_fp", prev_fp)

            saved_n_features = checkpoint.get("n_features")
            if saved_n_features is not None and saved_n_features != n_features:
                print_warn("Checkpoint feature count mismatch. Starting training from scratch.")
                stages = []
                prev_fp = np.empty((0, n_features), dtype=np.float32)
                prev_n_faces = len(X_train_faces)
                n_bg_pre = len(X_train_faces)
                fpr_macro_thr = 1.0
            else:
                prev_fp = np.asarray(prev_fp, dtype=np.float32)
                if prev_fp.ndim != 2 or prev_fp.shape[1] != n_features:
                    print_warn("Checkpoint hard negatives shape mismatch. Resetting hard negatives.")
                    prev_fp = np.empty((0, n_features), dtype=np.float32)

                start_stage = len(stages)
                saved_stage_num = checkpoint.get("stage_num", start_stage)
                if saved_stage_num != start_stage:
                    print_warn(
                        f"Checkpoint stage_num ({saved_stage_num}) does not match stages ({start_stage})."
                    )

                if start_stage >= CONFIG.max_stages:
                    print_color("All stages already trained. Nothing to resume.", color="green")
                    return stages, fpr_macro_thr

                print_color(f"Resuming training from stage {start_stage + 1}.", color="yellow")
        else:
            print_color("No existing stages found. Starting training from scratch.", color="yellow")
    else:
        print_color("Resume disabled. Starting training from scratch.", color="yellow")

    return start_stage, stages, fpr_macro_thr, prev_n_faces, n_bg_pre, prev_fp, n_features

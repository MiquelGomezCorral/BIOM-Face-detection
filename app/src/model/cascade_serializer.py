"""
Serialization utilities for Haar Cascade classifiers.
Handles saving to and loading from OpenCV-compatible XML format.
"""

import os
from xml.dom import minidom
import xml.etree.ElementTree as ET

from .cascade_def import HaarCascade
from .cascade_parser import HaarCascadeParser, build_haar_cascade_from_stages


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
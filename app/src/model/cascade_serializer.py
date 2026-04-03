"""
Serialization utilities for Haar Cascade classifiers.
Handles saving to and loading from OpenCV-compatible XML format.
"""

import os
from xml.dom import minidom
import xml.etree.ElementTree as ET

from .cascade_def import HaarCascade
from .cascade_parser import HaarCascadeParser


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
        cascade_elem = ET.SubElement(opencv_storage, 'haarcascade_frontalface_computed')
        cascade_elem.set('type_id', 'opencv-haar-classifier')
        
        # Add cascade properties
        size_elem = ET.SubElement(cascade_elem, 'size')
        size_elem.text = f'{cascade.width} {cascade.height}'
        
        # Store feature type
        feature_type_elem = ET.SubElement(cascade_elem, 'featureType')
        feature_type_elem.text = cascade.feature_type
        
        # Store stage count
        stage_num_elem = ET.SubElement(cascade_elem, 'stageNum')
        stage_num_elem.text = str(len(cascade.stages))
        
        # Add features (store all used features)
        features_elem = ET.SubElement(cascade_elem, 'features')
        for feat_id in sorted(cascade.features.keys()):
            feature = cascade.features[feat_id]
            feature_elem = ET.SubElement(features_elem, '_')
            
            # Store rectangles for this feature
            for rect in feature.rectangles:
                # Format: y x h w weight
                rect_text = f'{int(rect.y)} {int(rect.x)} {int(rect.height)} {int(rect.width)} {rect.weight}'
                rect_elem = ET.SubElement(feature_elem, 'rects')
                
                # Store as text with count prefix (mimics OpenCV format)
                text_content = ET.SubElement(rect_elem, '_')
                text_content.text = rect_text
        
        # Add stages
        stages_elem = ET.SubElement(cascade_elem, 'stages')
        
        for stage_idx, stage in enumerate(cascade.stages):
            stage_elem = ET.SubElement(stages_elem, '_')
            
            # Add weak classifiers as trees
            trees_elem = ET.SubElement(stage_elem, 'trees')
            
            for clf_idx, weak_clf in enumerate(stage.weak_classifiers):
                tree_elem = ET.SubElement(trees_elem, '_')
                
                # Store classifier info (feature_id, threshold, left_val, right_val)
                clf_info = ET.SubElement(tree_elem, '_')
                clf_info.text = f'{weak_clf.feature_id}\n{weak_clf.threshold}\n{weak_clf.left_value}\n{weak_clf.right_value}'
            
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
        
        print(f"✓ Saved Haar cascade to: {output_path}")
    
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


def save_stages(CONFIG, stages, stage_num, fpr_macro):
    """
    Save the current stages to XML files in OpenCV format.
    
    Args:
        CONFIG: Configuration object
        stages: List of trained stages (each with weak classifiers and thresholds)
        stage_num: Current stage number (for naming)
        fpr_macro: Current macro false positive rate (for naming)
        output_dir: Directory to save the XML files
    """
    filename = f'haar_cascade_stage_{stage_num}_fpr_{fpr_macro:.4f}.xml'
    output_path = os.path.join(CONFIG.computed_haar_cascades, filename)
    
    # Create a HaarCascade object for serialization
    cascade = HaarCascade(
        stages=stages,
        features={feat.feature_id: feat for stage in stages for feat in stage.features},
        width=CONFIG.crop_size,
        height=CONFIG.crop_size,
        feature_type="HAAR"
    )
    CascadeSerializer.save(cascade, output_path)
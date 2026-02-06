"""
Automatic License Plate Recognition System
==========================================
A classical computer vision approach to detecting and recognizing
Bangladeshi license plates using morphological operations and Tesseract OCR.

Author: Abyead Zarif Hassan
Project: Undergraduate Computer Vision Course Project
"""

import cv2
import numpy as np
import pytesseract
from typing import Tuple, List, Optional

class LicensePlateDetector:
    """
    Detects and recognizes license plates using classical CV techniques.
    
    Features:
    - Edge-based segmentation with Canny detector
    - Morphological operations for region proposal
    - Aspect-ratio-based filtering
    - Tesseract OCR for text extraction
    
    Limitations:
    - Sensitive to lighting conditions
    - Requires relatively clear plate visibility
    - Aspect ratio heuristics may miss non-standard plates
    """
    
    def __init__(self, 
                 lang: str = 'ben', 
                 min_area: int = 4000,
                 aspect_range: Tuple[float, float] = (2.0, 6.0)):
        """
        Initialize detector with configurable parameters.
        
        Args:
            lang: Tesseract language code ('ben' for Bengali)
            min_area: Minimum contour area in pixels
            aspect_range: Valid width/height ratio range for plates
        """
        self.lang = lang
        self.min_area = min_area
        self.aspect_range = aspect_range
        self.kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 7))
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance image and extract edge features.
        
        Pipeline:
        1. Grayscale conversion
        2. Histogram equalization (lighting normalization)
        3. Gaussian blur (noise reduction)
        4. Canny edge detection
        5. Morphological closing (gap filling)
        
        Args:
            image: BGR input image
            
        Returns:
            Binary edge map with closed contours
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        return cv2.morphologyEx(edges, cv2.MORPH_CLOSE, self.kernel)
    
    def find_candidates(self, morph_img: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Extract and rank potential license plate regions.
        
        Uses geometric constraints based on Bangladeshi plate standards:
        - Aspect ratio typically 2:1 to 5:1
        - Minimum size threshold to filter noise
        
        Args:
            morph_img: Binary morphological edge map
            
        Returns:
            List of bounding boxes (x, y, w, h) sorted by area
        """
        contours, _ = cv2.findContours(
            morph_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        candidates = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = w / h
            area = w * h
            
            if (self.aspect_range[0] < aspect_ratio < self.aspect_range[1] 
                and area > self.min_area):
                candidates.append((x, y, w, h))
        
        # Prioritize larger regions (more likely to be plates)
        return sorted(candidates, key=lambda b: b[2] * b[3], reverse=True)
    
    def extract_text(self, image: np.ndarray, 
                     coords: Tuple[int, int, int, int]) -> Tuple[str, np.ndarray]:
        """
        Perform OCR on cropped region of interest.
        
        Preprocessing steps:
        1. ROI extraction
        2. Grayscale conversion
        3. Otsu's binarization (adaptive thresholding)
        
        Args:
            image: Original BGR image
            coords: Bounding box (x, y, width, height)
            
        Returns:
            Tuple of (extracted_text, preprocessed_roi)
        """
        x, y, w, h = coords
        roi = image[y:y+h, x:x+w]
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(
            roi_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        
        text = pytesseract.image_to_string(thresh, lang=self.lang).strip()
        return text, thresh
    
    def detect(self, image_path: str, 
               confidence_threshold: int = 3) -> Optional[Tuple[str, Tuple]]:
        """
        End-to-end detection pipeline.
        
        Args:
            image_path: Path to input image
            confidence_threshold: Minimum text length to consider valid
            
        Returns:
            (detected_text, bounding_box) or None if no plate found
        """
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Could not load image: {image_path}")
        
        processed = self.preprocess(img)
        candidates = self.find_candidates(processed)
        
        for coords in candidates:
            text, _ = self.extract_text(img, coords)
            if len(text) > confidence_threshold:
                return text, coords
        
        return None


def main():
    """Demonstration of the detector on a sample image."""
    detector = LicensePlateDetector(lang='ben')
    
    result = detector.detect("sample_images/test_plate.jpg")
    
    if result:
        text, bbox = result
        print(f" Detected Plate: {text}")
        print(f" Location: x={bbox[0]}, y={bbox[1]}, "
              f"width={bbox[2]}, height={bbox[3]}")
    else:
        print("No license plate detected")


if __name__ == "__main__":
    main()

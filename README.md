# Adaptive Image Registration and Object Masking in Video Streams 🎥🔒

## 📌 Project Overview  
This project aims to develop a system that dynamically detects and masks specified regions within a video stream or recorded video based on manually marked templates. Using **image registration**, **feature detection**, **homography computation**, and **real-time tracking**, the system ensures accurate masking across frames.  

---

## 🔧 Features  
### ✅ Template Marking and Field Selection  
- Users can **load template images** and manually mark specific regions for masking.  
- Marked regions are saved for reference to ensure consistent masking during video processing.  

### ✅ Feature Detection and Matching  
- Implements robust feature detection using:  
  - **SIFT (Scale-Invariant Feature Transform)**  
  - **ORB (Oriented FAST and Rotated BRIEF)**  
- Feature matching techniques used:  
  - **FLANN-based Matcher** for fast approximate matching.  
  - **BFMatcher** for brute-force matching with improved accuracy.  

### ✅ Homography and Image Registration  
- Computes **homography** to map marked regions from template images to corresponding regions in video frames.  
- Applies perspective transformations for precise alignment and registration.  
- Ensures accurate positioning of marked fields on video frames.  

### ✅ Real-Time Object Tracking and Masking  
- Uses the **Lucas-Kanade optical flow** method to track object movement across frames.  
- Dynamically applies masks to predefined regions, updating positions in real time.  
- Accurate and consistent masking using homography transformations.  

### ✅ Performance and Optimization  
- Designed to operate in **real-time** while maintaining high accuracy.  
- Measures and optimizes **FPS (Frames Per Second)** to balance performance and resource usage.  
- Adjustable frame processing speed for efficient resource utilization.  

---

## 🛠️ Technology Stack  
- **Python:** Core language for scripting and implementation.  
- **OpenCV:** For image processing, feature detection, and tracking.  
- **NumPy:** Efficient numerical computations and array operations.  
- **Streamlit:** (if applicable) for interactive UI and visualization.  
- **Matplotlib:** For visualization of tracking and results.  

---

### **Adaptive Image Registration and Object Masking in Video Streams**

The objective of this project was to develop a system that can dynamically detect and mask specified regions within a video stream or recorded video based on manually marked templates. This involves image registration using feature detection, homography computation, and real-time tracking to ensure that the masking is accurately applied across frames. 

**Features**  
Template Marking and Field Selection  
The system enables users to identify and mark specific regions in template images for video processing. By loading template images, users can manually mark regions intended for masking in video playback or live streaming. These marked regions are saved for later reference to ensure accurate masking during processing.

Feature Detection and Matching
To detect and match features between template images and video frames, the system implements algorithms such as **SIFT (Scale-Invariant Feature Transform)** and **ORB (Oriented FAST and Rotated BRIEF)**. These robust methods ensure accurate detection of key points, while matching is facilitated using **FLANN-based matcher** and **BFMatcher** for seamless alignment of features between templates and video content.  

Homography and Image Registration 
The system computes homography to map marked regions from template images to corresponding regions in video frames. This process involves identifying matched points and applying perspective transformations, enabling precise alignment and registration of marked fields to video frames.  

Real-Time Object Tracking and Masking 
To track object movement and dynamically apply masking, the system utilizes the **Lucas-Kanade optical flow** method. Detected points are tracked across video frames, and predefined masks are applied to obscure specific areas in real-time. The mask positions are updated dynamically using homography to ensure consistent and accurate coverage.  

Performance and Optimization
The system is designed to operate in real-time while maintaining high accuracy and performance. Optimizations include reducing processing time, ensuring acceptable frame rates, and calculating **FPS (Frames Per Second)**. Additionally, frame processing speed can be controlled to balance performance and resource utilization effectively.  

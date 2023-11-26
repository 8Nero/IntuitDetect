DeepFake Video Detection for Bank Applications
Project Logo/Image Here

This project aims to tackle the challenge of deepfake videos within bank applications. The system employs a multi-step verification process involving facial recognition, facial expression tracking, and lip reading to ensure the authenticity of user-submitted videos.

Overview
Problem Statement
In today's digital age, deepfake videos pose a significant threat, especially in sensitive areas like banking applications. It becomes crucial to ensure that videos submitted by users are authentic and not manipulated.

Solution
Our proposed solution comprises three fundamental checks:

Facial Similarity Check:

The user submits an ID card and records a video.
We perform a face similarity check between the ID card image and every 50th frame from the video.
Facial Expression Tracking:

Utilizing advanced face tracker models, we track facial expressions throughout the video.
The system detects unnatural movements or discrepancies in facial expressions.
Lip Reading for Audio Verification:

Implementing lip reading techniques to analyze and verify audio data synced with the video.
This step ensures that the spoken words match the lip movements in the video.
Features
ID Card vs. Video Frame Comparison: Face similarity comparison for ID card and video frames.
Facial Expression Monitoring: Real-time tracking of facial expressions for anomaly detection.
Lip Reading Integration: Audio-based verification by synchronizing lip movements.
Usage
Requirements
Python 3.x
Libraries: OpenCV, TensorFlow, etc. (Add specific library requirements)
Installation
bash
Copy code
git clone https://github.com/your_username/your_repository.git
cd your_repository
pip install -r requirements.txt
Running the Project
Facial Similarity Check:

Run python facial_similarity_check.py to perform the face similarity check.
Facial Expression Tracking:

Execute python facial_expression_tracking.py to monitor facial expressions.
Lip Reading for Audio Verification:

Run python lip_reading_verification.py to perform lip reading on audio data.
Results
Facial Similarity Check

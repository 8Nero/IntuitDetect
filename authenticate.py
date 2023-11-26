from pathlib import Path

import cv2
import face_alignment
import numpy as np
import torch

import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_tensor
from deepface import DeepFace

face_landmarks = {
    'Jaw': {'range': (0, 17), 'color': (0, 255, 255)},        # Cyan for Jaw
    'Eyebrows': {'range': (17, 27), 'color': (255, 255, 0)},  # Yellow for Eyebrows
    'Nose': {'range': (27, 36), 'color': (255, 0, 0)},        # Blue for Nose
    'Eyes': {'range': (36, 48), 'color': (0, 0, 255)},        # Red for Eyes
    'Outer Lips': {'range': (48, 60), 'color': (255, 0, 255)},# Magenta for Outer Lips
    'Inner Lips': {'range': (60, 68), 'color': (128, 0, 128)} # Purple for Inner Lips
}

eye_threshold = 5
jaw_threshold = 5
mouth_threshold = 8

def face_verify(image1, image2):
    obj = DeepFace.verify(image1, image2, 
                          model_name = 'ArcFace', 
                          detector_backend = 'retinaface')
    return obj["verified"]

def epsilon_threshold(array1, array2, epsilon):
    delta = array1 - array2
    return np.all(np.abs(delta) >= epsilon)

def authenticate(video, id_image):
    results = {
        "identity_match":True,
        "eye_motion":0,
        "jaw_motion":0,
        "mouth_motion":0,
        "trackers_video":None
    }

    # Capture the frames, resize each frame
    cap = cv2.VideoCapture(video)
    frames = []
    while True:
        success, frame = cap.read()
        if not success:
            break
    
        frame = cv2.resize(frame, None, fx=0.2, fy=0.2, interpolation=cv2.INTER_LINEAR)
        frames.append(frame)
    
    num_frames = len(frames)

    # Face verification
    # Performs face verification every 50 frames, returns False if at least one returns false.
    for i in range(0, num_frames, 50):
        if(not face_verify(frames[i], id_image)):
            results["identity_match"] = False
            return False
    
    # Face landmarks detection
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, device='cpu', face_detector='blazeface')
    
    batch_size = 32
    fps = 32

    # Initialize VideoWriter parameters
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_file = 'face_trackers_out.mp4'
    frame_width, frame_height = frames[0].shape[1], frames[0].shape[0]
    
    # Create VideoWriter object
    out = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))

    for i in range(0, num_frames, batch_size):
        batch_frames = frames[i:i+batch_size]

        # Convert frames to the required format (assuming frames are in BGR format)
        batch = np.stack(batch_frames)
        batch = torch.Tensor(batch.transpose(0, 3, 1, 2))
        preds = fa.get_landmarks_from_batch(batch)
        
        # Visualize results and write to video
        for j, pred in enumerate(preds):
            frame_index = i + j
            if frame_index < num_frames:
                frame = frames[frame_index]
            
            #Change facial features
            if(j > 1):
                #Eye threshold check
                if(epsilon_threshold(preds[j][36:48], preds[j-1][36:48], eye_threshold)):
                    results["eye_motion"] += 1
                  
                #Jaw threshold check
                if(epsilon_threshold(preds[j][:17], preds[j-1][:17], jaw_threshold)):
                    results["mouth_motion"] += 1
                
                #Mouth threshold check
                if(epsilon_threshold(preds[j][48:60], preds[j-1][48:60], mouth_threshold)):
                    results["mouth_motion"] += 1            
        
            x_values = pred[:, 0].astype(np.int64)
            y_values = pred[:, 1].astype(np.int64)

            # Plot all points using scatter plot with specific colors for facial parts
            for part, part_info in face_landmarks.items():
                start, end = part_info['range']
                color = part_info['color']
    
                # Draw circles for each facial part with different colors
                for i in range(start, end):
                    cv2.circle(frame, (int(x_values[i]), int(y_values[i])), 2, color, -1)
            
            # Write the frame with landmarks to the video
            out.write(frame)

    # Release the VideoWriter and close the output file
    out.release()

    results["trackers_video"] = output_file
    return results
    

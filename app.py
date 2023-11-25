from flask import Flask, request, jsonify
import os

from deepface import DeepFace
import cv2


def face_verify(image1, image2):
    obj = DeepFace.verify(image1, image2, 
                          model_name = 'ArcFace', 
                          detector_backend = 'retinaface')
    return obj["verified"]


def authenticate(video, id_image):

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
            return False
    
    return True



app = Flask(__name__)

@app.route('/upload', methods=['POST'])
def upload():
    video_file = request.files.get('video')
    image_file = request.files.get('image')
    
    if not video_file or not image_file:
        return jsonify({'error': 'Please provide both video and image files'}), 400

    video_path = os.path.join('uploads', video_file.filename)
    video_file.save(video_path)

    image_path = os.path.join('uploads', image_file.filename)
    image_file.save(image_path)

    results = authenticate(video_path, image_path)

    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)

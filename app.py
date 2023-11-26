from authenticate import *
import os
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/')
def base():
    return "yes"

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

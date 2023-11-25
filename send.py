import requests

# Replace 'http://localhost:5000/upload' with your server's endpoint
upload_url = 'http://localhost:5000/upload'

# Path to the local MP4 file you want to upload
file_path = '/home/miranjo/IntuitDetect/aka47.mp4'

# Make a POST request to upload the file
with open(file_path, 'rb') as file:
    files = {'file': (file_path, file, 'video/mp4')}
    response = requests.post(upload_url, files=files)

# Print the response from the server
print(response.text)


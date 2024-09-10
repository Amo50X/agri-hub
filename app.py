from flask import Flask, jsonify, request, send_file
from io import BytesIO
import os
import numpy as np
import cv2
import base64
from flask_cors import CORS
from ultralytics import YOLO
import supervision as sv


UPLOAD_FOLDER = 'static/uploads'
app = Flask(__name__)
CORS(app)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER 
app.config['SECRET_KEY'] = 'tut4ir@amogelangsibanda/50x%'

weed_model = YOLO('static/weight/weed.pt')
maize_model = YOLO('static/weight/weed.pt')
animal_model = YOLO('static/weight/animal.pt')
tomato_model = YOLO('static/weight/tomato.pt')

tracker = sv.ByteTrack()
box_annotation = sv.BoxAnnotator()
dot_annotation = sv.DotAnnotator()
label_annotation = sv.LabelAnnotator()
trace_annotator = sv.TraceAnnotator()
heat_map_annotator = sv.HeatMapAnnotator()

def ImageProcesing(image, model):
    results = model.track(image , verbose=False, conf=0.5)[0]
    annotated_image = results.plot()  
    detections = sv.Detections.from_ultralytics(results)
    
    image = box_annotation.annotate(scene=image.copy(), detections=detections)
    dot_image = dot_annotation.annotate(scene=image.copy(), detections=detections)

    boxes = detections.xyxy
    count = str(len(boxes))
    
    return annotated_image, dot_image, count

def VideoProcesing(filepath, filename, model):

    Class_Model_Dict = model.model.names
    id_count = set()
    heat_frame = ""
    video_info = sv.VideoInfo.from_video_path(video_path=filepath)
    frame_generator = sv.get_video_frames_generator(filepath)
    target_path=f"static/predicts/{filename}"
    with sv.VideoSink(target_path=target_path,video_info=video_info, codec='h264') as sink:
        for frame in frame_generator:
            results = model.track(frame, conf=0.5)[0]

            detections = sv.Detections.from_ultralytics(results)
            detections = tracker.update_with_detections(detections)
            
            if not len(detections.tracker_id) == 0:
                for track_id in detections.tracker_id:
                    id_count.add(track_id)


            labels = [
                f"{Class_Model_Dict[class_id]} {confidence:.2f} id:#{track_id}"
                for class_id, confidence, track_id
                in zip(detections.class_id, detections.confidence, detections.tracker_id)
            ]

            annotated_frame = box_annotation.annotate(
                scene=frame.copy(), detections=detections)
            annotated_frame = label_annotation.annotate(
                scene=annotated_frame, detections=detections, labels=labels)

            annotated_frame = trace_annotator.annotate(
                scene=annotated_frame.copy(),
                detections=detections)
            
            heat_frame = heat_map_annotator.annotate(
                scene=frame.copy(),
                detections=detections)
            
            sink.write_frame(frame=annotated_frame)
    return heat_frame, len(id_count), target_path

@app.route("/weed", methods=["POST"])
def weed():
    if 'file' not in request.files: 
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files["file"]
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file:
        if "image" in file.content_type:
            in_memory_file = BytesIO()
            file.save(in_memory_file)
            data = np.frombuffer(in_memory_file.getvalue(), dtype=np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
            annotated_image, dot_image, count = ImageProcesing(img , weed_model)
            _, img_encoded = cv2.imencode('.jpg', annotated_image)
            image_as_text = base64.b64encode(img_encoded).decode('utf-8')
            
            _, dot_encoded = cv2.imencode('.jpg', dot_image)
            dot = base64.b64encode(dot_encoded).decode('utf-8')

 
            return jsonify({
                "message" : "image",
                "image" : image_as_text,
                "dot" : dot,
                "count" : count,
            })
        else:
            print(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            heat_image, count, target = VideoProcesing(filepath, file.filename, weed_model )
            
            _, heat_encoded = cv2.imencode('.jpg', heat_image)
            heat = base64.b64encode(heat_encoded).decode('utf-8')

            # print(sv.VideoInfo.from_video_path(video_path=target))
            with open(target, 'rb') as video_file:
                encoded_string = base64.b64encode(video_file.read()).decode('utf-8')
            print(count)
            return jsonify({
                'video': encoded_string,
                "count" : count,
                "heat": heat
                })
        
    return jsonify({
        "message" : "failed",
    })

@app.route("/maize", methods=["POST"])
def maize():
    if 'file' not in request.files: 
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files["file"]
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file:
        if "image" in file.content_type:
            in_memory_file = BytesIO()
            file.save(in_memory_file)
            data = np.frombuffer(in_memory_file.getvalue(), dtype=np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
            annotated_image, dot_image, count = ImageProcesing(img , maize_model)
            _, img_encoded = cv2.imencode('.jpg', annotated_image)
            image_as_text = base64.b64encode(img_encoded).decode('utf-8')
            
            _, dot_encoded = cv2.imencode('.jpg', dot_image)
            dot = base64.b64encode(dot_encoded).decode('utf-8')

 
            return jsonify({
                "message" : "image",
                "image" : image_as_text,
                "dot" : dot,
                "count" : count,
            })
        else:
            print(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            heat_image, count, target = VideoProcesing(filepath, file.filename, maize_model )
            
            _, heat_encoded = cv2.imencode('.jpg', heat_image)
            heat = base64.b64encode(heat_encoded).decode('utf-8')

            # print(sv.VideoInfo.from_video_path(video_path=target))
            with open(target, 'rb') as video_file:
                encoded_string = base64.b64encode(video_file.read()).decode('utf-8')
            print(count)
            return jsonify({
                'video': encoded_string,
                "count" : count,
                "heat": heat
                })
            
@app.route("/animal", methods=["POST"])
def animal():
    if 'file' not in request.files: 
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files["file"]
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file:
        if "image" in file.content_type:
            in_memory_file = BytesIO()
            file.save(in_memory_file)
            data = np.frombuffer(in_memory_file.getvalue(), dtype=np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
            annotated_image, dot_image, count = ImageProcesing(img , animal_model)
            _, img_encoded = cv2.imencode('.jpg', annotated_image)
            image_as_text = base64.b64encode(img_encoded).decode('utf-8')
            
            _, dot_encoded = cv2.imencode('.jpg', dot_image)
            dot = base64.b64encode(dot_encoded).decode('utf-8')

 
            return jsonify({
                "message" : "image",
                "image" : image_as_text,
                "dot" : dot,
                "count" : count,
            })
        else:
            print(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            heat_image, count, target = VideoProcesing(filepath, file.filename, animal_model )
            
            _, heat_encoded = cv2.imencode('.jpg', heat_image)
            heat = base64.b64encode(heat_encoded).decode('utf-8')

            # print(sv.VideoInfo.from_video_path(video_path=target))
            with open(target, 'rb') as video_file:
                encoded_string = base64.b64encode(video_file.read()).decode('utf-8')
            print(count)
            return jsonify({
                'video': encoded_string,
                "count" : count,
                "heat": heat
                })
        
@app.route("/tomato", methods=["POST"])
def tomato():
    if 'file' not in request.files: 
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files["file"]
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file:
        if "image" in file.content_type:
            in_memory_file = BytesIO()
            file.save(in_memory_file)
            data = np.frombuffer(in_memory_file.getvalue(), dtype=np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
            annotated_image, dot_image, count = ImageProcesing(img , tomato_model)
            _, img_encoded = cv2.imencode('.jpg', annotated_image)
            image_as_text = base64.b64encode(img_encoded).decode('utf-8')
            
            _, dot_encoded = cv2.imencode('.jpg', dot_image)
            dot = base64.b64encode(dot_encoded).decode('utf-8')

 
            return jsonify({
                "message" : "image",
                "image" : image_as_text,
                "dot" : dot,
                "count" : count,
            })
        else:
            print(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            heat_image, count, target = VideoProcesing(filepath, file.filename, tomato_model )
            
            _, heat_encoded = cv2.imencode('.jpg', heat_image)
            heat = base64.b64encode(heat_encoded).decode('utf-8')

            # print(sv.VideoInfo.from_video_path(video_path=target))
            with open(target, 'rb') as video_file:
                encoded_string = base64.b64encode(video_file.read()).decode('utf-8')
            print(count)
            return jsonify({
                'video': encoded_string,
                "count" : count,
                "heat": heat
                })
    
       
    return jsonify({
        "message" : "failed",
    })

    
if __name__ == "__main__":
    app.run(debug=True)
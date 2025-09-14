# HawkEye 2.0 - Main Flask API Server
# Complete AI Detection System with Image, Video, and Audio Analysis

from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import cv2
import numpy as np
import librosa
import os
from werkzeug.utils import secure_filename
import tempfile
import base64
from PIL import Image, ImageChops, ImageEnhance
import io
import json
from datetime import datetime
import hashlib
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'mp4', 'avi', 'mov', 'wav', 'mp3'}
MAX_CONTENT_LENGTH = 100 * 1024 * 1024  # 100MB max file size

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Create upload directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

class HawkEyeDetector:
    """Main class for all detection methods"""
    
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    
    def detect_image_manipulation(self, image_path):
        """Image manipulation detection using multiple techniques"""
        try:
            img = cv2.imread(image_path)
            if img is None:
                return {'error': 'Could not load image'}
            
            results = {}
            
            # 1. Error Level Analysis (ELA)
            ela_score = self._error_level_analysis(image_path)
            results['ela_score'] = ela_score
            
            # 2. Face consistency check
            face_score = self._face_consistency_check(img)
            results['face_consistency'] = face_score
            
            # 3. Noise analysis
            noise_score = self._noise_analysis(img)
            results['noise_analysis'] = noise_score
            
            # 4. Compression artifacts
            compression_score = self._compression_analysis(img)
            results['compression_analysis'] = compression_score
            
            # 5. Metadata analysis
            metadata_score = self._metadata_analysis(image_path)
            results['metadata_analysis'] = metadata_score
            
            # Calculate overall confidence
            confidence = self._calculate_image_confidence(results)
            is_fake = confidence > 0.6
            
            return {
                'is_manipulated': is_fake,
                'confidence': round(confidence, 3),
                'analysis_details': results,
                'analysis_type': 'image',
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {'error': f'Image analysis failed: {str(e)}'}
    
    def detect_video_deepfake(self, video_path):
        """Video deepfake detection using frame analysis"""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return {'error': 'Could not open video file'}
            
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            
            # Sample frames (analyze every 30th frame)
            sample_interval = max(1, frame_count // 20)
            
            frame_scores = []
            temporal_scores = []
            face_tracking_scores = []
            
            prev_faces = None
            frame_idx = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_idx % sample_interval == 0:
                    # Frame-level analysis
                    frame_score = self._analyze_video_frame(frame)
                    frame_scores.append(frame_score)
                    
                    # Face tracking consistency
                    current_faces = self._detect_faces_in_frame(frame)
                    if prev_faces is not None:
                        tracking_score = self._face_tracking_consistency(prev_faces, current_faces)
                        face_tracking_scores.append(tracking_score)
                    prev_faces = current_faces
                
                frame_idx += 1
            
            cap.release()
            
            if not frame_scores:
                return {'error': 'No frames could be analyzed'}
            
            # Temporal consistency analysis
            if len(frame_scores) > 1:
                temporal_score = self._temporal_consistency_analysis(frame_scores)
                temporal_scores.append(temporal_score)
            
            # Calculate overall confidence
            avg_frame_score = np.mean(frame_scores)
            avg_temporal_score = np.mean(temporal_scores) if temporal_scores else 0.5
            avg_tracking_score = np.mean(face_tracking_scores) if face_tracking_scores else 0.5
            
            overall_confidence = (avg_frame_score * 0.5 + avg_temporal_score * 0.3 + avg_tracking_score * 0.2)
            is_deepfake = overall_confidence > 0.65
            
            return {
                'is_deepfake': is_deepfake,
                'confidence': round(overall_confidence, 3),
                'frame_analysis': {
                    'frames_analyzed': len(frame_scores),
                    'avg_frame_score': round(avg_frame_score, 3),
                    'temporal_consistency': round(avg_temporal_score, 3),
                    'face_tracking_score': round(avg_tracking_score, 3)
                },
                'video_info': {
                    'duration': round(frame_count / fps, 2),
                    'fps': fps,
                    'total_frames': frame_count
                },
                'analysis_type': 'video',
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {'error': f'Video analysis failed: {str(e)}'}
    
    def detect_audio_deepfake(self, audio_path):
        """Audio deepfake detection using spectral analysis"""
        try:
            y, sr = librosa.load(audio_path, sr=16000)
            if len(y) == 0:
                return {'error': 'Could not load audio file'}
            
            results = {}
            
            # 1. Spectral analysis
            spectral_score = self._spectral_analysis(y, sr)
            results['spectral_analysis'] = spectral_score
            
            # 2. Pitch consistency
            pitch_score = self._pitch_consistency_analysis(y, sr)
            results['pitch_consistency'] = pitch_score
            
            # 3. Formant analysis
            formant_score = self._formant_analysis(y, sr)
            results['formant_analysis'] = formant_score
            
            # 4. Temporal pattern analysis
            temporal_score = self._audio_temporal_analysis(y, sr)
            results['temporal_analysis'] = temporal_score
            
            # 5. Noise floor analysis
            noise_score = self._audio_noise_analysis(y, sr)
            results['noise_analysis'] = noise_score
            
            # Calculate overall confidence
            confidence = self._calculate_audio_confidence(results)
            is_synthetic = confidence > 0.7
            
            duration = len(y) / sr
            
            return {
                'is_synthetic': is_synthetic,
                'confidence': round(confidence, 3),
                'analysis_details': results,
                'audio_info': {
                    'duration': round(duration, 2),
                    'sample_rate': sr,
                    'samples': len(y)
                },
                'analysis_type': 'audio',
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {'error': f'Audio analysis failed: {str(e)}'}
    
    # Helper methods for image analysis
    def _error_level_analysis(self, image_path):
        """Error Level Analysis - detects JPEG compression inconsistencies"""
        try:
            img = Image.open(image_path)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            temp_path = tempfile.mktemp(suffix='.jpg')
            img.save(temp_path, 'JPEG', quality=90)
            
            original = Image.open(image_path).convert('RGB')
            compressed = Image.open(temp_path).convert('RGB')
            
            diff = ImageChops.difference(original, compressed)
            extrema = diff.getextrema()
            max_diff = max([ex[1] for ex in extrema])
            
            os.remove(temp_path)
            
            score = min(max_diff / 255.0, 1.0)
            return score
            
        except Exception:
            return 0.5
    
    def _face_consistency_check(self, img):
        """Check for face manipulation inconsistencies"""
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            
            if len(faces) == 0:
                return 0.3
            
            suspicious_score = 0
            for (x, y, w, h) in faces:
                face_region = img[y:y+h, x:x+w]
                
                # Check for edge artifacts
                edges = cv2.Canny(face_region, 50, 150)
                edge_density = np.sum(edges > 0) / (w * h)
                
                # Color consistency
                face_mean = np.mean(face_region, axis=(0, 1))
                surrounding_region = img[max(0, y-20):min(img.shape[0], y+h+20), 
                                       max(0, x-20):min(img.shape[1], x+w+20)]
                surrounding_mean = np.mean(surrounding_region, axis=(0, 1))
                
                color_diff = np.linalg.norm(face_mean - surrounding_mean) / 255.0
                
                suspicious_score += (edge_density + color_diff) / 2
            
            return min(suspicious_score / len(faces), 1.0)
            
        except Exception:
            return 0.5
    
    def _noise_analysis(self, img):
        """Analyze noise patterns"""
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            noise = cv2.subtract(gray, blurred)
            
            noise_std = np.std(noise)
            noise_mean = np.mean(np.abs(noise))
            
            hist = cv2.calcHist([noise], [0], None, [256], [-128, 128])
            hist_normalized = hist.flatten() / np.sum(hist)
            
            entropy = -np.sum(hist_normalized * np.log2(hist_normalized + 1e-10))
            uniformity_score = 1.0 - (entropy / 8.0)
            
            return max(0, uniformity_score)
            
        except Exception:
            return 0.5
    
    def _compression_analysis(self, img):
        """Analyze compression artifacts"""
        try:
            yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
            y_channel = yuv[:, :, 0]
            
            blocks_suspicious = 0
            total_blocks = 0
            
            for i in range(0, y_channel.shape[0] - 8, 8):
                for j in range(0, y_channel.shape[1] - 8, 8):
                    block = y_channel[i:i+8, j:j+8]
                    
                    if i > 0 and j > 0:
                        top_block = y_channel[i-8:i, j:j+8]
                        left_block = y_channel[i:i+8, j-8:j]
                        
                        top_edge_diff = np.mean(np.abs(block[0, :] - top_block[-1, :]))
                        left_edge_diff = np.mean(np.abs(block[:, 0] - left_block[:, -1]))
                        
                        edge_score = (top_edge_diff + left_edge_diff) / 2
                        if edge_score > 10:
                            blocks_suspicious += 1
                    total_blocks += 1
            
            if total_blocks == 0:
                return 0.5
            
            return blocks_suspicious / total_blocks
            
        except Exception:
            return 0.5
    
    def _metadata_analysis(self, image_path):
        """Analyze image metadata"""
        try:
            img = Image.open(image_path)
            
            if hasattr(img, '_getexif') and img._getexif():
                return 0.2
            else:
                return 0.7
                
        except Exception:
            return 0.5
    
    def _calculate_image_confidence(self, results):
        """Calculate overall confidence for image"""
        weights = {
            'ela_score': 0.25,
            'face_consistency': 0.30,
            'noise_analysis': 0.20,
            'compression_analysis': 0.15,
            'metadata_analysis': 0.10
        }
        
        confidence = 0
        for key, weight in weights.items():
            if key in results:
                confidence += results[key] * weight
        
        return min(confidence, 1.0)
    
    # Video analysis helper methods
    def _analyze_video_frame(self, frame):
        """Analyze individual video frame"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            
            if len(faces) == 0:
                return 0.4
            
            frame_score = 0
            for (x, y, w, h) in faces:
                face_region = frame[y:y+h, x:x+w]
                
                face_blur = cv2.Laplacian(cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
                surrounding = frame[max(0, y-10):min(frame.shape[0], y+h+10), 
                                  max(0, x-10):min(frame.shape[1], x+w+10)]
                surrounding_blur = cv2.Laplacian(cv2.cvtColor(surrounding, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
                
                if surrounding_blur > 0:
                    blur_ratio = abs(face_blur - surrounding_blur) / surrounding_blur
                    frame_score += min(blur_ratio / 100, 1.0)
            
            return min(frame_score / len(faces), 1.0)
            
        except Exception:
            return 0.5
    
    def _detect_faces_in_frame(self, frame):
        """Detect faces in frame"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            return faces.tolist()
        except Exception:
            return []
    
    def _face_tracking_consistency(self, prev_faces, current_faces):
        """Check face tracking consistency"""
        try:
            if not prev_faces or not current_faces:
                return 0.5
            
            if len(prev_faces) != len(current_faces):
                return 0.8
            
            total_movement = 0
            for prev, curr in zip(prev_faces, current_faces):
                prev_center = (prev[0] + prev[2]/2, prev[1] + prev[3]/2)
                curr_center = (curr[0] + curr[2]/2, curr[1] + curr[3]/2)
                
                movement = np.sqrt((prev_center[0] - curr_center[0])**2 + 
                                 (prev_center[1] - curr_center[1])**2)
                total_movement += movement
            
            avg_movement = total_movement / len(prev_faces)
            return min(avg_movement / 50.0, 1.0)
            
        except Exception:
            return 0.5
    
    def _temporal_consistency_analysis(self, frame_scores):
        """Analyze temporal consistency"""
        try:
            if len(frame_scores) < 2:
                return 0.5
            
            score_variance = np.var(frame_scores)
            return min(score_variance * 2, 1.0)
            
        except Exception:
            return 0.5
    
    # Audio analysis helper methods
    def _spectral_analysis(self, y, sr):
        """Analyze audio spectrum"""
        try:
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
            
            centroid_variance = np.var(spectral_centroids)
            rolloff_variance = np.var(spectral_rolloff)
            
            consistency_score = 1.0 - min((centroid_variance + rolloff_variance) / 1000000, 1.0)
            
            return consistency_score
            
        except Exception:
            return 0.5
    
    def _pitch_consistency_analysis(self, y, sr):
        """Analyze pitch consistency"""
        try:
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
            
            pitch_track = []
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                if pitch > 0:
                    pitch_track.append(pitch)
            
            if len(pitch_track) < 10:
                return 0.5
            
            pitch_variance = np.var(pitch_track)
            pitch_mean = np.mean(pitch_track)
            
            if pitch_mean > 0:
                cv = pitch_variance / pitch_mean
                return 1.0 - min(cv / 0.1, 1.0)
            
            return 0.5
            
        except Exception:
            return 0.5
    
    def _formant_analysis(self, y, sr):
        """Simple formant analysis"""
        try:
            frame_length = int(0.025 * sr)
            hop_length = int(0.01 * sr)
            
            frames = librosa.util.frame(y, frame_length=frame_length, hop_length=hop_length)
            formant_consistency = []
            
            for frame in frames.T:
                if len(frame) > 0 and np.var(frame) > 0.001:
                    fft = np.abs(np.fft.fft(frame))
                    peaks = np.where(fft[1:] > fft[:-1])[0] + 1
                    
                    if len(peaks) > 2:
                        peak_spacing = np.diff(peaks[:3])
                        consistency = np.std(peak_spacing) if len(peak_spacing) > 1 else 0
                        formant_consistency.append(consistency)
            
            if formant_consistency:
                avg_consistency = np.mean(formant_consistency)
                return min(avg_consistency / 100, 1.0)
            
            return 0.5
            
        except Exception:
            return 0.5
    
    def _audio_temporal_analysis(self, y, sr):
        """Analyze temporal patterns"""
        try:
            frame_length = int(0.025 * sr)
            hop_length = int(0.01 * sr)
            
            rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
            energy_variance = np.var(rms)
            
            return 1.0 - min(energy_variance / 0.01, 1.0)
            
        except Exception:
            return 0.5
    
    def _audio_noise_analysis(self, y, sr):
        """Analyze noise patterns"""
        try:
            segment_length = sr // 4
            segments = [y[i:i+segment_length] for i in range(0, len(y), segment_length)]
            
            noise_levels = []
            for segment in segments:
                if len(segment) > 100:
                    energies = np.abs(segment)
                    noise_floor = np.percentile(energies, 10)
                    noise_levels.append(noise_floor)
            
            if noise_levels:
                noise_variance = np.var(noise_levels)
                return min(1.0 - (noise_variance / 0.001), 1.0)
            
            return 0.5
            
        except Exception:
            return 0.5
    
    def _calculate_audio_confidence(self, results):
        """Calculate overall confidence for audio"""
        weights = {
            'spectral_analysis': 0.25,
            'pitch_consistency': 0.25,
            'formant_analysis': 0.20,
            'temporal_analysis': 0.15,
            'noise_analysis': 0.15
        }
        
        confidence = 0
        for key, weight in weights.items():
            if key in results:
                confidence += results[key] * weight
        
        return min(confidence, 1.0)

# Initialize detector
detector = HawkEyeDetector()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    """API documentation homepage"""
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>HawkEye 2.0 - AI Detection API</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }
            .container { background-color: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            h1 { color: #333; border-bottom: 3px solid #007bff; padding-bottom: 10px; }
            .endpoint { background-color: #f8f9fa; padding: 20px; margin: 20px 0; border-radius: 8px; border-left: 4px solid #007bff; }
            .method { color: #28a745; font-weight: bold; }
            code { background-color: #e9ecef; padding: 2px 4px; border-radius: 3px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🦅 HawkEye 2.0 - AI Detection API</h1>
            <p><strong>Status:</strong> ✅ Online and Ready</p>
            <p><strong>Version:</strong> 2.0</p>
            
            <h2>Available Endpoints:</h2>
            
            <div class="endpoint">
                <h3><span class="method">POST</span> /api/detect/image</h3>
                <p><strong>Description:</strong> Detect image manipulation and deepfakes</p>
                <p><strong>Parameters:</strong> <code>image</code> (file)</p>
                <p><strong>Formats:</strong> PNG, JPG, JPEG, GIF</p>
            </div>
            
            <div class="endpoint">
                <h3><span class="method">POST</span> /api/detect/video</h3>
                <p><strong>Description:</strong> Detect video deepfakes</p>
                <p><strong>Parameters:</strong> <code>video</code> (file)</p>
                <p><strong>Formats:</strong> MP4, AVI, MOV</p>
            </div>
            
            <div class="endpoint">
                <h3><span class="method">POST</span> /api/detect/audio</h3>
                <p><strong>Description:</strong> Detect synthetic audio</p>
                <p><strong>Parameters:</strong> <code>audio</code> (file)</code>
                <p><strong>Formats:</strong> WAV, MP3</p>
            </div>
            
            <div class="endpoint">
                <h3><span class="method">GET</span> /api/health</h3>
                <p><strong>Description:</strong> API health check</p>
            </div>
            
            <h2>Integration Example:</h2>
            <pre><code>fetch('/api/detect/image', {
    method: 'POST',
    body: formData
}).then(response => response.json())</code></pre>
            
            <p><strong>Max file size:</strong> 100MB</p>
            <p><strong>Built for:</strong> HawkEye 2.0 Project Team</p>
        </div>
    </body>
    </html>
    """
    return html

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'HawkEye 2.0 Detection API',
        'version': '2.0',
        'capabilities': [
            'image_manipulation_detection',
            'video_deepfake_detection',
            'audio_synthesis_detection'
        ],
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/detect/image', methods=['POST'])
def detect_image():
    """Image detection endpoint"""
    try:
        if 'image' not in request.files:
            return jsonify({'status': 'error', 'message': 'No image file provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'status': 'error', 'message': 'No file selected'}), 400
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            try:
                result = detector.detect_image_manipulation(file_path)
                os.remove(file_path)  # Clean up
                
                if 'error' in result:
                    return jsonify({'status': 'error', 'message': result['error']}), 500
                
                return jsonify({'status': 'success', 'result': result})
                
            except Exception as e:
                if os.path.exists(file_path):
                    os.remove(file_path)
                return jsonify({'status': 'error', 'message': f'Analysis failed: {str(e)}'}), 500
        
        return jsonify({'status': 'error', 'message': 'Invalid file type'}), 400
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/detect/video', methods=['POST'])
def detect_video():
    """Video detection endpoint"""
    try:
        if 'video' not in request.files:
            return jsonify({'status': 'error', 'message': 'No video file provided'}), 400
        
        file = request.files['video']
        if file.filename == '':
            return jsonify({'status': 'error', 'message': 'No file selected'}), 400
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            try:
                result = detector.detect_video_deepfake(file_path)
                os.remove(file_path)  # Clean up
                
                if 'error' in result:
                    return jsonify({'status': 'error', 'message': result['error']}), 500
                
                return jsonify({'status': 'success', 'result': result})
                
            except Exception as e:
                if os.path.exists(file_path):
                    os.remove(file_path)
                return jsonify({'status': 'error', 'message': f'Analysis failed: {str(e)}'}), 500
        
        return jsonify({'status': 'error', 'message': 'Invalid file type'}), 400
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/detect/audio', methods=['POST'])
def detect_audio():
    """Audio detection endpoint"""
    try:
        if 'audio' not in request.files:
            return jsonify({'status': 'error', 'message': 'No audio file provided'}), 400
        
        file = request.files['audio']
        if file.filename == '':
            return jsonify({'status': 'error', 'message': 'No file selected'}), 400
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            try:
                result = detector.detect_audio_deepfake(file_path)
                os.remove(file_path)  # Clean up
                
                if 'error' in result:
                    return jsonify({'status': 'error', 'message': result['error']}), 500
                
                return jsonify({'status': 'success', 'result': result})
                
            except Exception as e:
                if os.path.exists(file_path):
                    os.remove(file_path)
                return jsonify({'status': 'error', 'message': f'Analysis failed: {str(e)}'}), 500
        
        return jsonify({'status': 'error', 'message': 'Invalid file type'}), 400
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    print("🦅 HawkEye 2.0 - AI Detection System Starting...")
    print("📊 Available Endpoints:")
    print("   • POST /api/detect/image - Image manipulation detection")
    print("   • POST /api/detect/video - Video deepfake detection")  
    print("   • POST /api/detect/audio - Audio synthesis detection")
    print("   • GET  /api/health - Health check")
    print("   • GET  / - API documentation")
    print("🚀 Server ready for integration!")
    
    # For development
    app.run(host='0.0.0.0', port=5000, debug=True)
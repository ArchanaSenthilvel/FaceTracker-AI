# api_server.py
# Simple HTTP API server without Flask - uses only built-in Python libraries

from http.server import HTTPServer, BaseHTTPRequestHandler
import json
from database import Database
from datetime import datetime
import os
from pathlib import Path

class FaceTrackerAPI(BaseHTTPRequestHandler):
    
    def _set_headers(self, content_type='application/json'):
        """Set response headers"""
        self.send_response(200)
        self.send_header('Content-type', content_type)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
    
    def do_OPTIONS(self):
        """Handle CORS preflight request"""
        self._set_headers()
    
    def do_GET(self):
        """Handle GET requests"""
        try:
            if self.path == '/api/stats':
                self.get_stats()
            elif self.path == '/api/faces':
                self.get_faces()
            elif self.path == '/api/events':
                self.get_events()
            elif self.path.startswith('/api/image/'):
                self.get_image()
            else:
                self.send_error(404, 'Endpoint not found')
        except Exception as e:
            self.send_error(500, str(e))
    
    def get_stats(self):
        """Get statistics"""
        db = Database()
        
        unique_visitors = db.get_unique_visitor_count()
        total_entries = db.events.count_documents({'event_type': 'entry'})
        total_exits = db.events.count_documents({'event_type': 'exit'})
        
        stats = {
            'uniqueVisitors': unique_visitors,
            'totalEntries': total_entries,
            'totalExits': total_exits
        }
        
        db.close()
        
        self._set_headers()
        self.wfile.write(json.dumps(stats).encode())
    
    def get_faces(self):
        """Get all registered faces"""
        db = Database()
        
        faces = []
        for face in db.faces.find().sort('first_seen', -1):
            faces.append({
                'face_id': face['face_id'],
                'first_seen': face['first_seen'].isoformat() if isinstance(face['first_seen'], datetime) else str(face['first_seen']),
                'last_seen': face['last_seen'].isoformat() if isinstance(face['last_seen'], datetime) else str(face['last_seen']),
                'total_visits': face.get('total_visits', 1)
            })
        
        db.close()
        
        self._set_headers()
        self.wfile.write(json.dumps(faces).encode())
    
    def get_events(self):
        """Get all events"""
        db = Database()
        
        events = []
        for event in db.events.find().sort('timestamp', -1).limit(50):
            events.append({
                'face_id': event['face_id'],
                'event_type': event['event_type'],
                'timestamp': event['timestamp'].isoformat() if isinstance(event['timestamp'], datetime) else str(event['timestamp']),
                'image_path': event['image_path']
            })
        
        db.close()
        
        self._set_headers()
        self.wfile.write(json.dumps(events).encode())
    
    def get_image(self):
        """Serve face images"""
        try:
            # Extract image path from URL
            image_path = self.path.replace('/api/image/', '')
            image_path = image_path.replace('%5C', '/').replace('%2F', '/')
            
            if os.path.exists(image_path):
                self._set_headers('image/jpeg')
                with open(image_path, 'rb') as f:
                    self.wfile.write(f.read())
            else:
                self.send_error(404, 'Image not found')
        except Exception as e:
            self.send_error(500, str(e))
    
    def log_message(self, format, *args):
        """Custom log format"""
        print(f"[{self.log_date_time_string()}] {format % args}")

def run_server(port=5000):
    """Run the API server"""
    server_address = ('', port)
    httpd = HTTPServer(server_address, FaceTrackerAPI)
    
    print(f"""
╔═══════════════════════════════════════════════════════════╗
║          Face Tracker API Server Started                  ║
╠═══════════════════════════════════════════════════════════╣
║  Server running at: http://localhost:{port}                ║
║                                                           ║
║  Available endpoints:                                     ║
║    • GET /api/stats    - Get statistics                  ║
║    • GET /api/faces    - Get all faces                   ║
║    • GET /api/events   - Get all events                  ║
║    • GET /api/image/<path> - Get face image              ║
║                                                           ║
║  Open dashboard.html in your browser!                     ║
║  Press Ctrl+C to stop the server                          ║
╚═══════════════════════════════════════════════════════════╝
    """)
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n\n✅ Server stopped successfully!")
        httpd.shutdown()

if __name__ == '__main__':
    run_server()
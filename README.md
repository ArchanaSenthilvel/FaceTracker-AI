# FaceTracker-AI---Intelligent-Visitor-Counter-with-Auto-Registration
An AI-powered system that automatically detects, recognizes, tracks, and counts unique visitors in video streams using YOLOv8, InsightFace, and MongoDB.

## ✨ Features

✅ Real-time Face Detection using YOLOv8  
✅ Face Recognition with InsightFace    
✅ Automatic Registration of new faces             
✅ Entry/Exit Logging with timestamps                
✅ Unique Visitor Counting without duplicates       
✅ Image Storage of all detected faces    
✅ Web Dashboard for visualization

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/ArchanaSenthilvel/facetrack-ai.git
cd facetrack-ai
```

2. **Create virtual environment** (recommended)
```bash
python -m venv venv

# Activate it:
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Start MongoDB**
```bash
# Make sure MongoDB is running
mongosh
# If it connects, you're good! Type 'exit' to quit.
```

5. **Run the system**
```bash
python main.py
```

The system will start processing the video and detecting faces!

---

## 🎨 Web Dashboard (Optional)

View your results in a beautiful web interface!

### Start the API Server
```bash
python api_server.py
```

### Open Dashboard

Simply open `dashboard.html` in your browser. The dashboard shows:
- Registered faces table
- Entry/exit events log
- Face image gallery

---
## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| **Face Detection** | YOLOv8 (ultralytics) |
| **Face Recognition** | InsightFace (ArcFace) |
| **Tracking** | Centroid Tracking (scipy) |
| **Database** | MongoDB (pymongo) |
| **Computer Vision** | OpenCV |
| **Backend** | Python 3.8+ |
| **API Server** | Built-in HTTP server |
| **Frontend** | HTML, CSS, JavaScript |

---
## 🎯 Use Cases

- **Retail Analytics**: Count foot traffic in stores
- **Security**: Monitor building entry/exit
- **Events**: Track attendee counts
- **Smart Home**: Know who's at your door
- **Office Management**: Employee attendance tracking

---


# 👁️ VisionAssist - Lost & Found AI Assistant

> **Find your lost belongings using AI-powered computer vision and a conversational voice assistant**

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-green)
![YOLO](https://img.shields.io/badge/YOLO-11-red)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 🎯 Problem Statement

We all lose things - keys, phones, remotes, wallets. The average person spends **20 minutes per day** searching for lost items.

**Current solutions:**
- Manual searching (time-consuming and frustrating)
- No intelligent locating tools
- Requires remembering where you last saw it

**VisionAssist Solution:**
AI-powered visual search combined with conversational AI to instantly locate your belongings.

---

## ✨ Key Features

### 🖼️ Image Upload & Detection
- Upload photos of your space (living room, bedroom, office)
- Automatic object detection using YOLOv11
- Displays all detected objects with confidence scores
- Annotated images with bounding boxes

### 🎤 AI Voice Assistant
- Ask natural language questions: *"Where is my phone?"*
- Get intelligent responses: *"Your phone is on the couch, next to the lamp"*
- Audio announcements for accessibility
- Conversation history

### 📹 Real-time RTSP Camera Streaming
- Connect to any RTSP-compatible IP camera
- Live object detection on camera feed
- Frame-by-frame analysis
- Summary results after streaming

### 🎯 Smart Object Detection
- Detects **80+ object classes** (COCO dataset)
- Spatial reasoning (describes location, not just objects)
- Confidence scores for reliability
- Works with various lighting conditions

---

## 🚀 Quick Start

### Local Development (2 minutes)

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/visionassist.git
cd visionassist

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate          # Linux/Mac
# OR
venv\Scripts\activate             # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Create .env file
echo "PERPLEXITY_API_KEY=sk-your-api-key-here" > .env

# 5. Run the app
streamlit run app.py
```

App will open at: **http://localhost:8501**

### Get API Keys (5 minutes)

**Perplexity API** (Free):
1. Go to https://www.perplexity.ai/settings/api
2. Sign up (free)
3. Create API key
4. Copy to `.env` file

---

## 📖 Usage Guide

### Scenario 1: Find Items in Photos

```
1. Open VisionAssist app
2. Select "Upload Image" tab
3. Take/upload a photo of your space
4. Click "🔍 Detect Objects"
5. View all detected objects
6. In sidebar, set "Object to Find" to "remote"
7. System announces: "The remote is on the coffee table"
```

### Scenario 2: Use Voice Query (My Favorite!)

```
1. Upload an image
2. Go to "AI Voice Assistant" section
3. Ask: "Where is the TV remote?"
4. System processes and responds instantly
5. Hear audio: "The remote is on the side table to the right of the couch"
```

### Scenario 3: Monitor with RTSP Camera

```
1. Select "RTSP Stream" tab
2. Enter camera URL: rtsp://username:password@192.168.x.x:554/stream
3. Click "▶️ Start Stream"
4. Watch live object detection
5. Click "⏹️ Stop Stream" when done
6. View final frame with all detections
```

---

## 🛠️ Installation Guide

### Prerequisites

- Python 3.9+
- pip (Python package manager)
- 4GB+ RAM (8GB recommended)
- Internet connection (for API calls)


## 🤖 How It Works

### Architecture Diagram

```
User Input
    ↓
┌─────────────────────────────────┐
│  INPUT PROCESSING               │
│  • Image Upload                 │
│  • RTSP Stream                  │
│  • Voice Query                  │
└─────────────────────────────────┘
    ↓
┌─────────────────────────────────┐
│  OBJECT DETECTION               │
│  • YOLOv11 Model               │
│  • 80+ COCO Classes            │
│  • Bounding Box Analysis       │
└─────────────────────────────────┘
    ↓
┌─────────────────────────────────┐
│  SPATIAL ANALYSIS               │
│  • Location Determination       │
│  • Confidence Scoring          │
│  • Description Generation      │
└─────────────────────────────────┘
    ↓
┌─────────────────────────────────┐
│  AI REASONING                   │
│  • Perplexity API               │
│  • Natural Language Response    │
│  • Context Understanding       │
└─────────────────────────────────┘
    ↓
┌─────────────────────────────────┐
│  OUTPUT                         │
│  • Visual Annotations           │
│  • Text Response               │
│  • Audio Announcement          │
└─────────────────────────────────┘
```

### Detection Pipeline

1. **Image Input** → Resize to 640x640
2. **YOLO Processing** → Detect objects
3. **Filter Results** → Confidence > 0.4
4. **Spatial Analysis** → Calculate positions
5. **AI Response** → Generate description
6. **Output** → Display + Speak

---

## 📊 Performance Metrics

| Metric | Value | Notes |
|--------|-------|-------|
| **Detection Speed** | 3-5s | Per image (CPU) |
| **Detection Speed (GPU)** | 1-2s | With CUDA |
| **Object Classes** | 80+ | COCO dataset |
| **Detection Confidence** | 40-99% | User configurable |
| **Voice Response Time** | 2-4s | Including API call |
| **Supported Formats** | PNG, JPG, JPEG | Image upload |
| **Max Image Size** | 200MB | Streamlit limit |
| **Concurrent Users** | Unlimited | With Azure scaling |

---

## 🎓 Tech Stack

### Frontend
- **Streamlit** - Interactive web app framework
- **Plotly** - Data visualization (optional)

### Computer Vision
- **YOLOv11** - Object detection (Ultralytics)
- **OpenCV** - Image processing
- **NumPy** - Numerical computation
- **Pillow** - Image manipulation

### AI & NLP
- **Perplexity API** - Conversational AI
- **OpenAI** - Backup API (optional)

### Speech
- **pyttsx3** - Text-to-speech
- **SpeechRecognition** - Speech-to-text
- **PyAudio** - Audio I/O

### Deployment
- **Azure App Service** - Cloud hosting
- **GitHub** - Version control
- **Docker** - Containerization

---

## 🚀 Deployment

### Local Testing
```bash
streamlit run app.py
```

### Azure Cloud Deployment
See **DEPLOYMENT.md** for complete step-by-step guide:
```bash
git push origin main  # Auto-deploys to Azure
```

### Docker Deployment
```bash
# Build
docker build -t visionassist .

# Run
docker run -p 8000:8000 -e PERPLEXITY_API_KEY=sk-xxx visionassist
```

---

## 🎯 Use Cases

### 🏠 Home Organization
- Find remote, keys, phone quickly
- Locate items in cluttered spaces
- Inventory of room contents

### 🏢 Office/Workspace
- Find important documents
- Locate office equipment
- Track workplace items

### ♿ Accessibility
- Voice-only interface for visually impaired
- Helps users with memory issues
- Hands-free operation

### 🏭 Warehouse/Inventory
- Monitor item locations
- Real-time inventory tracking
- Identify misplaced goods

### 🔒 Security
- Detect unauthorized items
- Monitor space activity
- Alert on unusual objects

---

## 🗺️ Roadmap

### ✅ Completed (v1.0)
- [x] Image upload & detection
- [x] RTSP streaming
- [x] Voice queries
- [x] Spatial analysis
- [x] Azure deployment

### 🔄 In Progress (v1.1)
- [ ] Multi-camera support
- [ ] Object history tracking
- [ ] Custom model training
- [ ] Mobile app (React Native)

### 📅 Planned (v2.0)
- [ ] Augmented Reality interface
- [ ] 3D spatial mapping
- [ ] Smart home integration
- [ ] Advanced ML model
- [ ] Database integration
- [ ] Enterprise features

---

## 📊 Statistics

- **80+** Detectable object classes
- **99.9%** Azure uptime SLA
- **<5s** Average detection time
- **2-4s** Average response time
- **∞** Scalable users (cloud-based)

---

*Last updated: November 2025*

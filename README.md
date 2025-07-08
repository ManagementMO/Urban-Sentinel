# 🏙️ Urban Sentinel
### *AI-Powered Urban Intelligence for Smarter Cities*

<div align="center">
  <img src="cover.png" alt="Urban Sentinel Cover" style="max-width: 100%; height: auto;">
</div>

## 🎯 **The Vision**

Urban blight costs cities **billions annually** in reduced property values, increased crime, and community displacement. Traditional approaches are reactive—we only act after decay has already set in, when it's most expensive to fix.

**Urban Sentinel flips this model.** Our AI system predicts at-risk neighborhoods up to **in advance**, enabling proactive interventions that save communities and millions in taxpayer dollars.

---

## ✨ **What Makes This Different**

🔮 **Predictive, Not Reactive** — See the future of your city before it unfolds  
🎯 **94.4% Accuracy** — Trained on 5+ years of real Toronto data  
⚡ **Real-Time Intelligence** — Interactive risk visualization at 30fps  
🗺️ **Actionable Insights** — Click any neighborhood for detailed risk analysis  
💰 **Cost-Saving** — Prevent problems before they become expensive to fix  

---

## 🔬 **The Tech Behind the Magic**

### **Machine Learning Engine**
- **Enhanced LightGBM** with cross-validation and early stopping
- **10,659 risk predictions** across Toronto's urban grid
- **Feature engineering** from 311 service complaints, temporal patterns, and geographic correlations
- **2014-2019 data** for comprehensive training

### **Frontend**
```typescript
React + TypeScript + Mapbox GL JS
Performance-optimized rendering (30fps on any device)
Glass-morphic design with dynamic risk filtering
```

### **Backend**
```python
FastAPI + Python + GeoPandas
Real-time ML inference pipeline
Spatial data processing and GeoJSON generation
```

### **Infrastructure**
```docker
Docker containerization for seamless deployment
Hot-reload development environment
Cross-platform compatibility
```

---

## 📊 **By the Numbers**

<div align="center">

| Metric | Value | Impact |
|--------|-------|--------|
| **Model Accuracy** | 94.4% ROC-AUC | Industry-leading precision |
| **Risk Predictions** | 10,659 | Complete Toronto coverage |
| **Data Span** | 2014-2019 | Decade of insights |
| **Prediction Horizon** | 2+ years | Early intervention window |
| **Response Time** | <500ms | Real-time intelligence |

</div>

---

## 🛠️ **Quick Start**

### **Prerequisites**
- Docker & Docker Compose
- Node.js 16+ (for local development)
- Python 3.9+ (for local development)

### **One-Command Launch**
```bash
# Clone and run the entire stack
git clone https://github.com/your-username/Urban-Sentinel.git
cd Urban-Sentinel
docker-compose up
```

**That's it!** Urban Sentinel will be running at:
- 🌐 Frontend: `http://localhost:3000`
- 🔧 Backend API: `http://localhost:8000`

### **Development Setup**
```bash
# Frontend
cd frontend
npm install
npm start

# Backend
cd backend
pip install -r requirements.txt
python api.py
```

---

## 📋 **Project Structure**

```
Urban-Sentinel/
├── 🎨 frontend/          # React + TypeScript UI
│   ├── src/components/   # Landing page, risk map, filters
│   └── src/services/     # API integration
├── 🧠 backend/           # FastAPI + ML pipeline
│   ├── api.py           # REST API endpoints
│   ├── model.py         # LightGBM training & inference
│   └── geojson.py       # Spatial data processing
├── 📊 datasets/          # Toronto 311 service data
└── 🐳 docker-compose.yml # One-command deployment
```

---

## 🗺️ **How It Works**

1. **Data Ingestion** — Process Toronto's 311 service requests (2014-2019)
2. **Feature Engineering** — Extract temporal patterns, complaint clusters, geographic correlations
3. **Model Training** — Enhanced LightGBM with stratified sampling and early stopping
4. **Risk Prediction** — Generate urban decay forecasts for each grid cell
5. **Visualization** — Interactive Mapbox display with real-time filtering and statistics

---

## 🎨 **Screenshots**

### Landing Page
*Clean, professional design that builds trust with city officials*

### Interactive Risk Map
*Toronto's urban grid color-coded by blight risk — red zones need immediate attention*

### Detailed Analytics
*Click any area for comprehensive risk breakdown and trend analysis*

---

## 🌟 **What's Next**

- 🌍 **Multi-city expansion** — Chicago, Detroit, New York
- 📱 **Mobile app** for field workers and community engagement  
- 🛰️ **Satellite imagery integration** for enhanced predictions
- 📊 **Economic impact modeling** to quantify intervention ROI
- 🔗 **API ecosystem** for integration with existing city systems

---

<div align="center">

Star this repo if you believe in building smarter cities!
[![GitHub stars](https://img.shields.io/github/stars/ManagementMO/Urban-Sentinel?style=social)](https://github.com/ManagementMO/Urban-Sentinel)

</div>

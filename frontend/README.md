# Urban Decay Map - Frontend

A modern, interactive web application for visualizing urban decay in Toronto using 3D maps and data from 311 calls and satellite imagery.

## Features

- **3D Interactive Map**: Powered by Mapbox GL JS with full pan, zoom, and rotation capabilities
- **Decay Visualization**: Heat map "aura" effects showing decay intensity with color gradients
- **Data Source Filtering**: Toggle between 311 calls, satellite data, or combined view
- **Responsive Design**: Works seamlessly on desktop, tablet, and mobile devices
- **Real-time Tooltips**: Click on decay zones to see detailed information

## Setup

### Prerequisites

- Node.js (v14 or higher)
- npm or yarn
- Mapbox account and API token

### Installation

1. Install dependencies:
```bash
npm install
```

2. Configure environment variables:
   - Create a `.env` file in the frontend directory
   - Add your Mapbox token:
```
REACT_APP_MAPBOX_TOKEN=your_mapbox_token_here
REACT_APP_API_URL=http://localhost:8000
```

3. Start the development server:
```bash
npm start
```

The app will open at [http://localhost:3000](http://localhost:3000)

## Color Legend

- **Blue/White**: Minimal decay (0.0 - 0.2)
- **Yellow**: Mild decay (0.2 - 0.5)
- **Orange**: Noticeable decay (0.5 - 0.8)
- **Red**: Severe decay (0.8 - 1.0)

## Technology Stack

- **React** with TypeScript
- **Mapbox GL JS** for 3D map rendering
- **Axios** for API communication
- **CSS3** with modern features (backdrop-filter, animations)

## API Integration

The frontend expects a backend API endpoint at `/decay` that returns data in this format:

```json
[
  {
    "latitude": 43.6532,
    "longitude": -79.3832,
    "decay_level": 0.8,
    "source": "311"
  }
]
```

If the API is unavailable, the app will use mock data for demonstration.

## Building for Production

```bash
npm run build
```

This creates an optimized production build in the `build` folder.

## Design Principles

- **Minimalistic UI**: Clean, uncluttered interface focusing on the map
- **Professional Aesthetic**: City of Toronto blue accent color (#0057c0)
- **Smooth Interactions**: 60fps performance target with hardware acceleration
- **Accessibility**: High contrast colors and readable text at all zoom levels

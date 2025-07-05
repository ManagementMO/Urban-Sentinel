# Mapbox Token Setup

To use the Urban Decay Map, you need a Mapbox access token.

## Steps:

1. **Get a Mapbox Token**
   - Go to [https://account.mapbox.com/](https://account.mapbox.com/)
   - Sign up for a free account (if you don't have one)
   - Navigate to your account dashboard
   - Copy your default public token or create a new one

2. **Add Token to the App**
   
   **Option A: Environment Variable (Recommended)**
   - Create a `.env` file in the `frontend` directory
   - Add this line:
     ```
     REACT_APP_MAPBOX_TOKEN=your_mapbox_token_here
     ```
   - Replace `your_mapbox_token_here` with your actual token
   
   **Option B: Direct in Code**
   - Open `frontend/src/components/Map.tsx`
   - Find line 8:
     ```typescript
     mapboxgl.accessToken = process.env.REACT_APP_MAPBOX_TOKEN || 'YOUR_MAPBOX_TOKEN_HERE';
     ```
   - Replace `'YOUR_MAPBOX_TOKEN_HERE'` with your actual token

3. **Restart the App**
   - Stop the development server (Ctrl+C)
   - Run `npm start` again

## Testing Without a Token

The app will show the map container but won't load map tiles without a valid token. You'll still be able to see the UI components and test the data filtering functionality. 
# Backend Keep-Alive System

This document explains the automatic keep-alive system implemented to prevent the backend from sleeping on Render's free tier.

## Problem

Render's free tier puts services to sleep after 15 minutes of inactivity, causing delays when users visit your site. This creates a poor user experience as the first request can take 30-60 seconds to wake up the backend.

## Solution

The frontend implements a comprehensive keep-alive system with multiple strategies:

### 1. **Automatic Keep-Alive Service** (`src/services/keepAlive.ts`)

- **Primary Keep-Alive**: Sends a robust health check every 13 minutes
- **Light Pings**: Sends lightweight pings every 5 minutes (only during user activity)
- **Activity Tracking**: Monitors user interactions to avoid unnecessary pings
- **Recovery System**: Multiple retry attempts with exponential backoff
- **Visibility Handling**: Checks backend when user returns to tab

### 2. **Configuration Options**

```typescript
interface KeepAliveConfig {
  primaryInterval: number;      // 13 minutes (beats 15-min timeout)
  pingInterval: number;         // 5 minutes (light pings)
  maxInactivityTime: number;    // 30 minutes (stop after inactivity)
  retryAttempts: number;        // 3 (recovery attempts)
  enabled: boolean;             // true (enable/disable)
}
```

### 3. **Usage in App**

```typescript
import { keepAliveService } from './services/keepAlive';

// In useEffect
keepAliveService.setStatusCallback(setBackendStatus);
keepAliveService.start();

// Cleanup
return () => keepAliveService.stop();
```

## Features

### ✅ **Smart Activity Detection**
- Tracks mouse movements, clicks, scrolls, keyboard input
- Only pings when user is actively using the app
- Stops pinging after 30 minutes of inactivity

### ✅ **Window Visibility Handling**
- Detects when user switches back to the tab
- Immediately checks backend status
- Attempts wake-up if needed

### ✅ **Progressive Recovery**
- Multiple retry attempts with increasing delays
- Different strategies (health check vs. light ping)
- Intelligent fallback mechanisms

### ✅ **Status Indicators**
- Visual notification when backend is sleeping
- Development mode shows service status
- Console logging for debugging

### ✅ **Performance Optimized**
- Lightweight ping endpoints
- Configurable intervals
- Passive event listeners
- Automatic cleanup

## Backend Endpoints Used

1. **`/health`** - Full health check with retry logic
2. **`/`** - Lightweight ping for frequent checks

## How It Works

1. **On App Start**: Immediately checks backend status
2. **During Use**: Sends keep-alive pings every 13 minutes
3. **Background**: Light pings every 5 minutes (if user active)
4. **Tab Switch**: Checks status when user returns
5. **Recovery**: Multiple attempts if backend seems down

## Status Indicators

- **🔄 Backend warming up...** - Backend is sleeping/unresponsive
- **No indicator** - Backend is awake and healthy
- **Development status** - Shows service state in dev mode

## Benefits

- **Faster Load Times**: Backend stays warm for active users
- **Better UX**: No 30-60 second delays on requests
- **Resource Efficient**: Only pings during user activity
- **Robust**: Multiple fallback strategies
- **Configurable**: Easy to adjust timing and behavior

## Monitoring

Check browser console for keep-alive logs:
- ✅ Successful pings
- ⚠️ Warning messages
- ❌ Error conditions
- 🔄 Recovery attempts

This system ensures your Urban Sentinel app provides a smooth, responsive experience even on Render's free tier! 
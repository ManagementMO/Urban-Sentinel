import axios from 'axios';
import { DecayData } from '../types';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

// Mock data for development/testing
const mockDecayData: DecayData[] = [
  { latitude: 43.6532, longitude: -79.3832, decay_level: 0.8, source: '311' },
  { latitude: 43.6426, longitude: -79.3871, decay_level: 0.3, source: 'satellite' },
  { latitude: 43.6700, longitude: -79.3900, decay_level: 0.6, source: 'combined' },
  { latitude: 43.6600, longitude: -79.4000, decay_level: 0.9, source: '311' },
  { latitude: 43.6450, longitude: -79.3800, decay_level: 0.4, source: 'satellite' },
  { latitude: 43.6550, longitude: -79.3950, decay_level: 0.7, source: 'combined' },
  { latitude: 43.6480, longitude: -79.3750, decay_level: 0.5, source: '311' },
  { latitude: 43.6650, longitude: -79.3850, decay_level: 0.2, source: 'satellite' },
];

export const fetchDecayData = async (): Promise<DecayData[]> => {
  try {
    // Try to fetch from the backend API
    const response = await axios.get<DecayData[]>(`${API_BASE_URL}/decay`);
    return response.data;
  } catch (error) {
    console.warn('Failed to fetch from API, using mock data:', error);
    // Return mock data if API is not available
    return mockDecayData;
  }
}; 
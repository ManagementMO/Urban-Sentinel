import axios from 'axios';
import { 
  parseWKTToGeoJSON, 
  convertRiskDataToGeoJSON,
  getRiskLevel,
  getRiskColor 
} from '../utils/geoHelpers';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

// ============================================================================
// --- TYPE DEFINITIONS ---
// ============================================================================

export interface RiskGridCell {
  cell_id: number;
  geometry: string; // WKT polygon string or GeoJSON geometry
  risk_score: number;
  total_complaints_mean?: number;
  blight_complaints_mean?: number;
  total_complaints_std?: number;
  blight_complaints_std?: number;
  blight_complaints_recent_mean?: number;
  blight_complaints_trend?: number;
  blight_complaints_2019?: number;
  blight_complaints_2020?: number;
  is_blighted?: boolean;
  target_blight_count?: number;
  overall_most_common_blight?: string;
  recent_most_common_blight?: string;
  [key: string]: any; // For other dynamic features
}

export interface FeatureImportance {
  feature: string;
  importance: number;
  rank: number;
  description: string;
}

export interface FeatureImportanceResponse {
  top_features: FeatureImportance[];
  all_features: FeatureImportance[];
  total_features: number;
  model_type: string;
}

export interface ApiStats {
  total_cells: number;
  columns: string[];
  [key: string]: number | string | string[]; // For dynamic numeric stats like mean, std
}

export interface CellDetails {
  cell_id: number;
  risk_score: number;
  risk_level: string;
  coordinates: {
    geometry: string | null;
  };
  features: {
    [key: string]: {
      value: number;
      description: string;
    };
  };
  historical_data: {
    is_blighted: boolean;
    target_blight_count: number;
    overall_most_common_blight: string;
    recent_most_common_blight: string;
  };
}

export interface TopRiskArea {
  cell_id: number;
  risk_score: number;
  risk_level: string;
  total_complaints_mean: number;
  blight_complaints_mean: number;
  is_blighted: boolean;
}

export interface HealthCheck {
  status: string;
  model_loaded: boolean;
  data_loaded: boolean;
  total_cells: number;
}

// ============================================================================
// --- API FUNCTIONS ---
// ============================================================================

// Health check endpoint
export const checkHealth = async (): Promise<HealthCheck> => {
  try {
    const response = await axios.get<HealthCheck>(`${API_BASE_URL}/health`);
    return response.data;
  } catch (error) {
    console.error('Health check failed:', error);
    throw error;
  }
};

// Main risk prediction endpoint - returns all grid cells with risk scores
export const fetchRiskData = async (): Promise<RiskGridCell[]> => {
  try {
    const response = await axios.get<RiskGridCell[]>(`${API_BASE_URL}/api/predict-risk`);
    return response.data;
  } catch (error) {
    console.error('Failed to fetch risk data:', error);
    throw error;
  }
};

// Get feature importance to understand what drives risk predictions
export const fetchFeatureImportance = async (): Promise<FeatureImportanceResponse> => {
  try {
    const response = await axios.get<FeatureImportanceResponse>(`${API_BASE_URL}/api/feature-importance`);
    return response.data;
  } catch (error) {
    console.error('Failed to fetch feature importance:', error);
    throw error;
  }
};

// Get basic statistics about the dataset
export const fetchStats = async (): Promise<ApiStats> => {
  try {
    const response = await axios.get<ApiStats>(`${API_BASE_URL}/api/stats`);
    return response.data;
  } catch (error) {
    console.error('Failed to fetch stats:', error);
    throw error;
  }
};

// Get detailed information about a specific cell
export const fetchCellDetails = async (cellId: number): Promise<CellDetails> => {
  try {
    const response = await axios.get<CellDetails>(`${API_BASE_URL}/api/cell-details/${cellId}`);
    return response.data;
  } catch (error) {
    console.error(`Failed to fetch details for cell ${cellId}:`, error);
    throw error;
  }
};

// Get top highest-risk areas
export const fetchTopRiskAreas = async (limit: number = 20): Promise<TopRiskArea[]> => {
  try {
    const response = await axios.get<TopRiskArea[]>(`${API_BASE_URL}/api/top-risk-areas?limit=${limit}`);
    return response.data;
  } catch (error) {
    console.error('Failed to fetch top risk areas:', error);
    throw error;
  }
};

// ============================================================================
// --- UTILITY FUNCTIONS ---
// ============================================================================

// Error handler for API calls
export const handleApiError = (error: any, context: string): never => {
  if (axios.isAxiosError(error)) {
    if (error.response) {
      console.error(`${context} - API Error:`, error.response.status, error.response.data);
      throw new Error(`API Error: ${error.response.data.detail || error.response.statusText}`);
    } else if (error.request) {
      console.error(`${context} - Network Error:`, error.request);
      throw new Error('Network Error: Unable to reach the API server');
    }
  }
  console.error(`${context} - Unknown Error:`, error);
  throw new Error(`Unknown Error: ${error.message || 'Something went wrong'}`);
};

// Batch API calls with error handling
export const fetchAllRiskData = async () => {
  try {
    const [riskData, featureImportance, stats, topRiskAreas] = await Promise.all([
      fetchRiskData(),
      fetchFeatureImportance(),
      fetchStats(),
      fetchTopRiskAreas(10)
    ]);

    return {
      riskData,
      featureImportance,
      stats,
      topRiskAreas,
      geoJsonData: convertRiskDataToGeoJSON(riskData)
    };
  } catch (error) {
    handleApiError(error, 'fetchAllRiskData');
  }
};
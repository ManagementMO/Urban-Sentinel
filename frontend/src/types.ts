// Import types from api service
import { RiskGridCell, FeatureImportanceResponse, ApiStats, TopRiskArea } from './services/api';

export interface DecayData {
  latitude: number;
  longitude: number;
  decay_level: number; // 0.0 (none) to 1.0 (severe)
  source: '311' | 'satellite' | 'combined';
}

export interface TooltipData {
  latitude: number;
  longitude: number;
  decayLevel: number;
  source: string;
  description?: string;
}

export interface AppRiskData {
  riskData: RiskGridCell[];
  geoJsonData: GeoJSON.FeatureCollection;
  featureImportance: FeatureImportanceResponse;
  stats: ApiStats;
  topRiskAreas: TopRiskArea[];
} 
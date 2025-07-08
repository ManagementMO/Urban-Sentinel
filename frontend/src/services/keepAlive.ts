import { keepBackendAlive, pingBackend } from './api';

interface KeepAliveConfig {
  primaryInterval: number; // Main keep-alive interval (minutes)
  pingInterval: number; // Light ping interval (minutes)
  maxInactivityTime: number; // Stop pinging after inactivity (minutes)
  retryAttempts: number; // Number of retry attempts on failure
  enabled: boolean; // Enable/disable keep-alive
}

const DEFAULT_CONFIG: KeepAliveConfig = {
  primaryInterval: 13, // 13 minutes (beats Render's 15-minute timeout)
  pingInterval: 5, // 5 minutes
  maxInactivityTime: 30, // 30 minutes
  retryAttempts: 3,
  enabled: true
};

export class KeepAliveService {
  private config: KeepAliveConfig;
  private primaryInterval: NodeJS.Timeout | null = null;
  private pingInterval: NodeJS.Timeout | null = null;
  private lastActivityTime: number = Date.now();
  private activityListeners: string[] = ['mousedown', 'mousemove', 'keypress', 'scroll', 'touchstart', 'click'];
  private isRunning: boolean = false;
  private statusCallback?: (status: 'awake' | 'sleeping' | 'unknown') => void;

  constructor(config?: Partial<KeepAliveConfig>) {
    this.config = { ...DEFAULT_CONFIG, ...config };
    this.handleActivity = this.handleActivity.bind(this);
    this.handleVisibilityChange = this.handleVisibilityChange.bind(this);
  }

  // Set callback for status updates
  setStatusCallback(callback: (status: 'awake' | 'sleeping' | 'unknown') => void) {
    this.statusCallback = callback;
  }

  // Start the keep-alive service
  start(): void {
    if (!this.config.enabled || this.isRunning) return;

    console.log('🚀 Starting Enhanced Keep-Alive Service');
    this.isRunning = true;

    // Set up primary keep-alive interval
    this.primaryInterval = setInterval(async () => {
      try {
        const success = await keepBackendAlive();
        this.statusCallback?.(success ? 'awake' : 'sleeping');
        
        if (success) {
          console.log('✅ Primary keep-alive successful');
        } else {
          console.warn('⚠️  Primary keep-alive failed - attempting recovery');
          await this.attemptRecovery();
        }
      } catch (error) {
        console.error('❌ Primary keep-alive error:', error);
        this.statusCallback?.('sleeping');
      }
    }, this.config.primaryInterval * 60 * 1000);

    // Set up light ping interval
    this.pingInterval = setInterval(async () => {
      const timeSinceActivity = Date.now() - this.lastActivityTime;
      
      // Only ping if there's been recent activity
      if (timeSinceActivity < this.config.maxInactivityTime * 60 * 1000) {
        try {
          const success = await pingBackend();
          this.statusCallback?.(success ? 'awake' : 'sleeping');
        } catch (error) {
          console.warn('⚠️  Background ping failed:', error);
        }
      }
    }, this.config.pingInterval * 60 * 1000);

    // Set up activity tracking
    this.setupActivityTracking();

    // Set up visibility change handling
    document.addEventListener('visibilitychange', this.handleVisibilityChange);

    // Initial backend check
    this.performInitialCheck();
  }

  // Stop the keep-alive service
  stop(): void {
    if (!this.isRunning) return;

    console.log('🛑 Stopping Keep-Alive Service');
    this.isRunning = false;

    if (this.primaryInterval) {
      clearInterval(this.primaryInterval);
      this.primaryInterval = null;
    }

    if (this.pingInterval) {
      clearInterval(this.pingInterval);
      this.pingInterval = null;
    }

    // Remove activity listeners
    this.activityListeners.forEach(event => {
      document.removeEventListener(event, this.handleActivity);
    });

    // Remove visibility change listener
    document.removeEventListener('visibilitychange', this.handleVisibilityChange);
  }

  // Manually trigger a keep-alive check
  async manualCheck(): Promise<boolean> {
    try {
      const success = await keepBackendAlive();
      this.statusCallback?.(success ? 'awake' : 'sleeping');
      return success;
    } catch (error) {
      console.error('Manual keep-alive check failed:', error);
      this.statusCallback?.('sleeping');
      return false;
    }
  }

  // Update configuration
  updateConfig(newConfig: Partial<KeepAliveConfig>): void {
    this.config = { ...this.config, ...newConfig };
    
    if (this.isRunning) {
      this.stop();
      this.start();
    }
  }

  // Get current configuration
  getConfig(): KeepAliveConfig {
    return { ...this.config };
  }

  // Check if service is running
  isActive(): boolean {
    return this.isRunning;
  }

  private setupActivityTracking(): void {
    this.activityListeners.forEach(event => {
      document.addEventListener(event, this.handleActivity, { passive: true });
    });
  }

  private handleActivity(): void {
    this.lastActivityTime = Date.now();
  }

  private async handleVisibilityChange(): Promise<void> {
    if (!document.hidden && document.visibilityState === 'visible') {
      console.log('👁️  Window became visible - checking backend status');
      this.handleActivity();
      
      try {
        const success = await keepBackendAlive();
        this.statusCallback?.(success ? 'awake' : 'sleeping');
        
        if (!success) {
          await this.attemptRecovery();
        }
      } catch (error) {
        console.error('Visibility change check failed:', error);
        this.statusCallback?.('sleeping');
      }
    }
  }

  private async performInitialCheck(): Promise<void> {
    try {
      const success = await keepBackendAlive();
      this.statusCallback?.(success ? 'awake' : 'sleeping');
      
      if (success) {
        console.log('✅ Initial backend check successful');
      } else {
        console.warn('⚠️  Initial backend check failed - attempting wake-up');
        await this.attemptRecovery();
      }
    } catch (error) {
      console.error('Initial backend check failed:', error);
      this.statusCallback?.('sleeping');
    }
  }

  private async attemptRecovery(): Promise<void> {
    console.log('🔄 Attempting backend recovery...');
    
    for (let i = 0; i < this.config.retryAttempts; i++) {
      try {
        await new Promise(resolve => setTimeout(resolve, 2000 * (i + 1))); // Progressive delay
        const success = await pingBackend();
        
        if (success) {
          console.log(`✅ Recovery successful on attempt ${i + 1}`);
          this.statusCallback?.('awake');
          return;
        }
      } catch (error) {
        console.warn(`Recovery attempt ${i + 1} failed:`, error);
      }
    }
    
    console.error('❌ All recovery attempts failed');
    this.statusCallback?.('sleeping');
  }
}

// Export a singleton instance for easy use
export const keepAliveService = new KeepAliveService();

// Export configuration for customization
export type { KeepAliveConfig }; 
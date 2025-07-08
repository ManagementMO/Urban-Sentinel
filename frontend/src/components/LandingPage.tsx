import React, { useEffect, useRef, useState } from 'react';
import './LandingPage.css';

interface LandingPageProps {
  onTryItOut: () => void;
  loading?: boolean;
  error?: string | null;
}

interface TimelineSection {
  id: string;
  title: string;
  content: string;
  image: string;
}

const timelineSections: TimelineSection[] = [
  {
    id: 'advanced-ml',
    title: 'Advanced Machine Learning',
    content: 'Our enhanced LightGBM model achieves 94.4% ROC-AUC accuracy, trained on 10+ years of Toronto 311 service data. Using sophisticated feature engineering and early stopping, we predict urban blight risk with unprecedented precision.',
    image: 'https://images.unsplash.com/photo-1453500920688-4442e4d16f50?w=1200&auto=format&fit=crop&q=60&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxzZWFyY2h8Mnx8VXJiYW4lMjBEZWNheSUyMFRvcm9udG98ZW58MHx8MHx8fDA%3D'
  },
  {
    id: 'real-time-visualization',
    title: 'Real-Time Risk Visualization',
    content: 'Interactive Mapbox integration with performance-optimized rendering delivers smooth 30fps visualization on any device. Dynamic risk filtering, hover effects, and detailed popups provide actionable insights for urban planners.',
    image: 'https://images.unsplash.com/photo-1584291527935-456e8e2dd734?q=80&w=2050&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D'
  },
  {
    id: 'intelligent-prediction',
    title: 'Intelligent Early Warning',
    content: 'By analyzing complaint patterns, temporal trends, and geographic correlations, Urban Sentinel identifies at-risk neighborhoods before visible decay occurs. This proactive approach enables targeted interventions that save cities millions in remediation costs.',
    image: 'https://images.unsplash.com/photo-1605810230434-7631ac76ec81?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&q=80'
  }
];

const LandingPage: React.FC<LandingPageProps> = ({ onTryItOut }) => {
  const [activeSection, setActiveSection] = useState(0);
  const [showTimeline, setShowTimeline] = useState(false);
  const sectionsRef = useRef<(HTMLDivElement | null)[]>([]);
  const heroRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const observerOptions = {
      root: null,
      rootMargin: '-20% 0px -20% 0px',
      threshold: 0.5
    };

    const observer = new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          const sectionIndex = sectionsRef.current.findIndex(
            (ref) => ref === entry.target
          );
          if (sectionIndex !== -1) {
            setActiveSection(sectionIndex);
          }
        }
      });
    }, observerOptions);

    // Observer for hero section to control timeline visibility
    const heroObserver = new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        if (entry.target === heroRef.current) {
          setShowTimeline(!entry.isIntersecting);
        }
      });
    }, {
      root: null,
      rootMargin: '0px',
      threshold: 0.1
    });

    sectionsRef.current.forEach((section) => {
      if (section) observer.observe(section);
    });

    if (heroRef.current) {
      heroObserver.observe(heroRef.current);
    }

    return () => {
      sectionsRef.current.forEach((section) => {
        if (section) observer.unobserve(section);
      });
      if (heroRef.current) {
        heroObserver.unobserve(heroRef.current);
      }
    };
  }, []);

  return (
    <div className="landing-page">
      {/* Hero Section */}
      <div className="hero-section" ref={heroRef}>
        <header className="landing-header">
          <h1 className="brand-title">URBAN SENTINEL</h1>
        </header>
        
        <main className="landing-main">
          <div className="content-section">
            {/* Subtitle overlays that will be behind the image */}
            <div className="background-text">
              <div className="subtitle-overlay subtitle-overlay-1">
                <span className="subtitle-text">Predict • Prevent • Protect</span>
              </div>
              <div className="subtitle-overlay subtitle-overlay-2">
                <span className="subtitle-text1">94.4% Accuracy</span>
              </div>
              <div className="subtitle-overlay subtitle-overlay-3">
                <span className="subtitle-text2">Real-Time Intelligence</span>
              </div>
              <div className="subtitle-overlay subtitle-overlay-4">
                <span className="subtitle-text3">Smart Cities</span>
              </div>
            </div>
            
            <div className="image-section">
              <div className="toronto-image">
                <img 
                  src="https://images.unsplash.com/photo-1517935706615-2717063c2225?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1000&q=80" 
                  alt="Toronto Skyline - Urban Intelligence"
                  className="skyline-img"
                />
              </div>
            </div>
            
            <div className="text-section">
              <h1 className="main-title">
                Preventing Urban Decay
                <span className="subtitle">Before It Happens</span>
              </h1>
              
              <div className="description">
                <p className="lead-text">
                  <strong>Urban blight costs cities billions annually</strong> in reduced property values, increased crime, and community displacement.
                </p>
                
                <div className="stats-highlight">
                  <div className="stat">
                    <span className="stat-number">94.4%</span>
                    <span className="stat-label">Model Accuracy</span>
                  </div>
                  <div className="stat">
                    <span className="stat-number">10,659</span>
                    <span className="stat-label">Risk Predictions</span>
                  </div>
                  <div className="stat">
                    <span className="stat-number">2014-2024</span>
                    <span className="stat-label">Years of Data</span>
                  </div>
                </div>
                
                <button className="cta-button" onClick={onTryItOut}>
                  <span>Explore Live Demo</span>
                  <div className="button-glow"></div>
                </button>
              </div>
            </div>
          </div>
        </main>
      </div>

      {/* Fixed Timeline Sidebar - Only show when not on hero section */}
      <div className={`timeline-sidebar ${showTimeline ? 'visible' : 'hidden'}`}>
        <div className="timeline-line"></div>
        <div 
          className="timeline-indicator"
          style={{
            transform: `translateY(${activeSection * 33.33 + 6.8}vh)`
          }}
        ></div>
        
        <div className="timeline-labels">
          {timelineSections.map((section, index) => (
            <div 
              key={section.id}
              className={`timeline-label ${activeSection === index ? 'active' : ''}`}
              style={{ top: `${index * 33.33 + 11}vh` }}
            >
              <div className="timeline-dot"></div>
              <h3>{section.title}</h3>
            </div>
          ))}
        </div>
      </div>

      {/* Full-Page Timeline Sections */}
      {timelineSections.map((section, index) => (
        <div
          key={section.id}
          ref={(el) => {
            sectionsRef.current[index] = el;
          }}
          className={`timeline-page ${activeSection === index ? 'active' : ''}`}
          data-section={index}
        >
          <div className={`timeline-page-content ${showTimeline ? 'with-sidebar' : 'without-sidebar'}`}>
            <div className="section-image">
              <img src={section.image} alt={section.title} />
              <div className="image-overlay"></div>
            </div>
            <div className="section-text">
              <h2>{section.title}</h2>
              <p>{section.content}</p>
              
              {index === 0 && (
                <div className="tech-specs">
                  <div className="tech-item">⚡ LightGBM Enhanced Model</div>
                  <div className="tech-item">📊 Cross-Validated Performance</div>
                  <div className="tech-item">🔧 Feature Engineering Pipeline</div>
                </div>
              )}
              
              {index === 1 && (
                <div className="tech-specs">
                  <div className="tech-item">🗺️ Mapbox GL JS Integration</div>
                  <div className="tech-item">⚡ 30fps Optimized Rendering</div>
                  <div className="tech-item">🎯 Dynamic Risk Filtering</div>
                </div>
              )}
              
              {index === 2 && (
                <div className="tech-specs">
                  <div className="tech-item">🔮 Predictive Analytics</div>
                  <div className="tech-item">📈 Temporal Pattern Analysis</div>
                  <div className="tech-item">🏙️ Geospatial Intelligence</div>
                </div>
              )}
            </div>
          </div>
        </div>
              ))}
    </div>
  );
};

export default LandingPage; 
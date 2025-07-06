import React, { useEffect, useRef, useState } from 'react';
import './LandingPage.css';

interface LandingPageProps {
  onTryItOut: () => void;
}

interface TimelineSection {
  id: string;
  title: string;
  content: string;
  image: string;
}

const timelineSections: TimelineSection[] = [
  {
    id: 'kids-grow-up',
    title: 'Safer Streets ',
    content: 'As children grow up in the city and begin to explore, creating a safer and more welcoming environment benefits not only our kids, but also the elderly. It encourages everyone to learn, discover, and experience their surroundings without fear.',
    image: 'https://images.unsplash.com/photo-1596464716127-f2a82984de30?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&q=80'
  },
  {
    id: 'people-having-fun',
    title: 'Stronger Community',
    content: 'A strong community is built on mutual respect and understanding. By fostering a sense of belonging and providing opportunities for engagement, we can create a city where everyone feels valued and supported.',
    image: 'https://images.unsplash.com/photo-1501386761578-eac5c94b800a?q=80&w=2070&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D'
  },
  {
    id: 'why-we-care',
    title: 'WHY WE CARE',
    content: 'Lorem Ipsum has been the industry\'s standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen',
    image: 'https://images.unsplash.com/photo-1645415070366-2dd8a830e97b?q=80&w=1030&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D'
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
                <span className="subtitle-text">Toronto City</span>
              </div>
              <div className="subtitle-overlay subtitle-overlay-2">
                <span className="subtitle-text1">Toronto City</span>
              </div>
              <div className="subtitle-overlay subtitle-overlay-3">
                <span className="subtitle-text2">Toronto City</span>
              </div>
              <div className="subtitle-overlay subtitle-overlay-4">
                <span className="subtitle-text3">Toronto City</span>
              </div>
            </div>
            
            <div className="image-section">
              <div className="toronto-image">
                <img 
                  src="https://images.unsplash.com/photo-1517935706615-2717063c2225?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1000&q=80" 
                  alt="Toronto Skyline with CN Tower"
                  className="skyline-img"
                />
              </div>
            </div>
            
            <div className="text-section">
              <h1 className="main-title">
                Toronto City
              </h1>
              
              <div className="description">
                <p>
                  <strong>MAJORITY</strong> of cities are either to late to recover, and 
                  with earlier notice, we could of tackled, <strong>so this is 
                  where we come in to tackle this, and save cities before 
                  to late</strong>
                </p>
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
            </div>
            <div className="section-text">
              <h2>{section.title}</h2>
              <p>{section.content}</p>
            </div>
          </div>
        </div>
      ))}
    </div>
  );
};

export default LandingPage; 
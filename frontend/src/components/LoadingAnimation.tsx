import React, { useEffect, useState } from 'react';
import './LoadingAnimation.css';

interface LoadingAnimationProps {
  onAnimationComplete: () => void;
}

const LoadingAnimation: React.FC<LoadingAnimationProps> = ({ onAnimationComplete }) => {
  const [animationPhase, setAnimationPhase] = useState<'initial' | 'strip-slide' | 'cube-fade'>('initial');

  useEffect(() => {
    // Start the strip slide animation after 0.5 seconds
    const stripSlideTimeout = setTimeout(() => {
      setAnimationPhase('strip-slide');
    }, 500);

    // Start the cube fade animation after the strip slides off (1 second total)
    const cubeFadeTimeout = setTimeout(() => {
      setAnimationPhase('cube-fade');
    }, 1000);

    // Complete the animation after cube fade (2 seconds total)
    const completeTimeout = setTimeout(() => {
      onAnimationComplete();
    }, 2000);

    return () => {
      clearTimeout(stripSlideTimeout);
      clearTimeout(cubeFadeTimeout);
      clearTimeout(completeTimeout);
    };
  }, [onAnimationComplete]);

  return (
    <div className={`loading-animation ${animationPhase}`}>
      <div className="city-background">
        <img 
          src="https://images.unsplash.com/photo-1449824913935-59a10b8d2000?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=2000&q=80"
          alt="3D City View"
          className="city-image"
        />
        <div className="city-overlay"></div>
      </div>
      
      <div className="blue-strip-container">
        <div className="blue-strip"></div>
      </div>
      
      <div className="cube-fragments">
        <div className="cube-fragment fragment-1"></div>
        <div className="cube-fragment fragment-2"></div>
        <div className="cube-fragment fragment-3"></div>
        <div className="cube-fragment fragment-4"></div>
        <div className="cube-fragment fragment-5"></div>
        <div className="cube-fragment fragment-6"></div>
        <div className="cube-fragment fragment-7"></div>
        <div className="cube-fragment fragment-8"></div>
        <div className="cube-fragment fragment-9"></div>
      </div>
    </div>
  );
};

export default LoadingAnimation; 
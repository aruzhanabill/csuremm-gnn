// src/components/Hero.js
import React, { useEffect, useRef, useState } from 'react';

const Hero = () => {
  const vantaRef = useRef(null);
  const [vantaEffect, setVantaEffect] = useState(null);

  useEffect(() => {
    // Ensure VANTA and THREE are available globally
    if (!vantaEffect && window.VANTA && window.THREE) {
      const effect = window.VANTA.NET({
        el: vantaRef.current,
        mouseControls: true,
        touchControls: true,
        gyroControls: false,
        minHeight: 200.0,
        minWidth: 200.0,
        scale: 1.0,
        scaleMobile: 1.0,
        color: 0xeb941a,
        backgroundColor: 0x1378ab,
        points: 20.0,
        maxDistance: 34.0,
        spacing: 13.0,
        showDots: true
      });
      setVantaEffect(effect);
    }
    return () => {
      if (vantaEffect) vantaEffect.destroy();
    };
  }, [vantaEffect]);

  return (
    <header ref={vantaRef} style={{ 
        height: '200px',
        width: '100vw', 
        color: 'white', 
        display: 'flex', 
        flexDirection: 'column', 
        justifyContent: 'center', 
        alignItems: 'center',
        overflow: 'hidden',
        }}>
      <h1 style={{ fontSize: '3rem', zIndex: 1 }}>An Introduction to OmniGNN</h1>
      <h1a style={{ fontSize: '1.25rem', zIndex: 1 }}>Globalizing the topology of Graph Neural Networks (GNNs) for robust performance under shock.</h1a>
    </header>
  );
};

export default Hero;
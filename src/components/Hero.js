import React, { useEffect, useRef, useState } from 'react';

const Hero = () => {
  const vantaRef = useRef(null);
  const [vantaEffect, setVantaEffect] = useState(null);

  useEffect(() => {
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
    <header
      ref={vantaRef}
      style={{
        height: '240px',
        width: '100vw',
        color: 'white',
        display: 'flex',
        flexDirection: 'column',
        justifyContent: 'center',
        alignItems: 'center',
        overflow: 'hidden',
      }}
    >
      <h1 style={{ fontSize: '3rem', zIndex: 1 }}>Structure Over Signal</h1>
      <p style={{ fontSize: '1.2rem', zIndex: 1, margin: 0 }}>
        Globalizing the topology of Graph Neural Networks (GNNs) for robust performance under shock.
      </p>
      <div style={{ marginTop: '10px', zIndex: 1 }}>
        <a
          href="https://drive.google.com/file/d/1tEP-hnOzikhs5X4bJmCRFy7TeKR0sZsK/view?usp=sharing"
          target="_blank"
          rel="noopener noreferrer"
          style={{ color: 'white', textDecoration: 'underline', marginRight: '20px' }}
        >
          [Paper]
        </a>
        <a
          href="https://github.com/amberhli/OmniGNN"
          target="_blank"
          rel="noopener noreferrer"
          style={{ color: 'white', textDecoration: 'underline', marginRight: '20px' }}
        >
          [Code & Models]
        </a>
        <a
          href="/bibtex.txt"
          target="_blank"
          rel="noopener noreferrer"
          style={{ color: 'white', textDecoration: 'underline' }}
        >
          [BibTeX]
        </a>
      </div>
    </header>
  );
};

export default Hero;

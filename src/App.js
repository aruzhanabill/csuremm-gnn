// src/App.js
import React from 'react';
import Hero from './components/Hero';
import Section from './components/Section';
import InteractiveDiagram from './components/InteractiveDiagram';
import Footer from './components/Footer';

function App() {
  return (
    <div>
      <Hero />
      <Section title="Why Graph Neural Networks?" content="Why this model matters and what problem it solves." />
      <Section title="Constructing the Graph" content="Overview of the architecture, layers, and flow." />
      <Section title="Embedding the Graph" content="Overview of the architecture, layers, and flow." />
      <Section title="Sequencing the Graphs" content="Overview of the architecture, layers, and flow." />
      <InteractiveDiagram />
      <Footer />
    </div>
  );
}

export default App;
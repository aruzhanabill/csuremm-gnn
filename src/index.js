// src/index.js
import React from 'react';
import ReactDOM from 'react-dom/client';
import './styles/main.css';
import App from './App';
import 'katex/dist/katex.min.css';
 
const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
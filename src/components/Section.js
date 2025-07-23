/* === Section.js === */
import React from 'react';

const Section = ({ title, content }) => (
  <section>
    <h2>{title}</h2>
    {content}
  </section>
);
 
export default Section;
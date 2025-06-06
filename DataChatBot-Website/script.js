// Simple scroll animation using Intersection Observer
// Adds 'visible' class to elements with 'animate-on-scroll' when they enter view

document.addEventListener('DOMContentLoaded', () => {
  const options = { threshold: 0.15 };
  const observer = new IntersectionObserver((entries, obs) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.classList.add('visible');
        obs.unobserve(entry.target);
      }
    });
  }, options);

  document.querySelectorAll('.animate-on-scroll').forEach(el => observer.observe(el));
});

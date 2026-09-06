/**
 * dashboard.js - Dashboard utilities (stat counters and animations)
 * Customer Complaint Classification System
 */

document.addEventListener('DOMContentLoaded', () => {
  // Animate integer stat counters (avoid corrupting float percentages like 98.37%)
  document.querySelectorAll('.kpi-value-int').forEach(el => {
    const raw = el.textContent.trim();
    const target = parseInt(raw, 10);
    if (isNaN(target) || target === 0) return;
    let current = 0;
    const step = Math.max(1, Math.ceil(target / 25));
    const timer = setInterval(() => {
      current = Math.min(current + step, target);
      el.textContent = current;
      if (current >= target) clearInterval(timer);
    }, 25);
  });
});

/**
 * app.js - Global utility and home page scripts
 * Customer Complaint Classification System
 */

// Expose setExample globally (called from inline HTML)
window.setExample = function(text) {
  const textarea = document.getElementById('complaintInput');
  if (!textarea) return;
  textarea.value = text;
  const counter = document.getElementById('charCount');
  if (counter) counter.textContent = text.length;
  textarea.focus();
  textarea.scrollIntoView({ behavior: 'smooth', block: 'center' });
};

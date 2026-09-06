/**
 * history.js - History page utilities
 * Customer Complaint Classification System
 */

// Debounce search input to auto-submit filter form
document.addEventListener('DOMContentLoaded', () => {
  const searchInput = document.getElementById('searchInput');
  const filterForm  = document.getElementById('filterForm');
  if (!searchInput || !filterForm) return;

  let debounceTimer;
  searchInput.addEventListener('input', () => {
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(() => filterForm.submit(), 500);
  });
});

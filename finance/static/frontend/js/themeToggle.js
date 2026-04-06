// Theme Toggle
const themeToggleButton = document.getElementById('themeToggleButton');
const bodyElement = document.body;

themeToggleButton.onclick = function() {
    bodyElement.classList.toggle('dark-theme');

    // Change button icon based on theme
    if (bodyElement.classList.contains('dark-theme')) {
        themeToggleButton.innerHTML = '🌙'; // Moon icon for dark
        themeToggleButton.setAttribute('aria-label', 'Switch to light theme');
    } else {
        themeToggleButton.innerHTML = '☀️'; // Sun icon for light
        themeToggleButton.setAttribute('aria-label', 'Switch to dark theme');
    }
    // Optional: Save theme preference in localStorage
    localStorage.setItem('theme', bodyElement.classList.contains('dark-theme') ? 'dark' : 'light');
}

// Optional: Load theme preference on page load
document.addEventListener('DOMContentLoaded', () => {
    const savedTheme = localStorage.getItem('theme');
    if (savedTheme === 'dark') {
        bodyElement.classList.add('dark-theme');
        themeToggleButton.innerHTML = '🌙';
        themeToggleButton.setAttribute('aria-label', 'Switch to light theme');
    } else {
        bodyElement.classList.remove('dark-theme'); // Ensure light is default
        themeToggleButton.innerHTML = '☀️';
        themeToggleButton.setAttribute('aria-label', 'Switch to dark theme');
    }
});

// Theme Toggle
function initTheme() {
    const themeToggleButton = document.getElementById('themeToggleButton');
    const bodyElement = document.body;
    if (!themeToggleButton || !bodyElement) return;

    const applyTheme = (isDark) => {
        bodyElement.classList.toggle('dark-theme', isDark);
        themeToggleButton.innerHTML = isDark ? '🌙' : '☀️';
        themeToggleButton.setAttribute('aria-label', isDark ? 'Switch to light theme' : 'Switch to dark theme');
    };

    const savedTheme = localStorage.getItem('theme');
    applyTheme(savedTheme === 'dark');

    themeToggleButton.addEventListener('click', () => {
        const nextDark = !bodyElement.classList.contains('dark-theme');
        applyTheme(nextDark);
        localStorage.setItem('theme', nextDark ? 'dark' : 'light');
    });
}

document.addEventListener('DOMContentLoaded', initTheme);

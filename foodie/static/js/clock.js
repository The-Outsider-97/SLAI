document.addEventListener('DOMContentLoaded', function() {
    // Initialize clock
    updateClock();
    setInterval(updateClock, 1000);

    // Apply saved theme
    applyTheme();
});

function updateClock() {
    const clockElement = document.getElementById('nav-clock');
    if (!clockElement) return;

    const settings = JSON.parse(localStorage.getItem('userSettings')) || {};
    const timeFormat = settings.timeFormat || '12h';
    const now = new Date();

    if (timeFormat === '12h') {
        // 12-hour format: HH:MM:SS AM/PM
        let hours = now.getHours();
        const ampm = hours >= 12 ? 'p.m.' : 'a.m.';
        hours = hours % 12 || 12;
        clockElement.textContent = `${hours}:${padZero(now.getMinutes())}:${padZero(now.getSeconds())} ${ampm}`;
    } else {
        // 24-hour format: HH:MM:SS
        clockElement.textContent = `${padZero(now.getHours())}:${padZero(now.getMinutes())}:${padZero(now.getSeconds())}`;
    }
}

function padZero(num) {
    return num.toString().padStart(2, '0');
}

function applyTheme() {
    const settings = JSON.parse(localStorage.getItem('userSettings')) || {};
    const theme = settings.theme || 'dark';
    document.body.setAttribute('data-theme', theme);
}
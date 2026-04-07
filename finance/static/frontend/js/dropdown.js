// Dropdown Menu Toggle
function initDropdown() {
    const menuButton = document.getElementById('menuButton');
    const dropdownMenu = document.getElementById('dropdownMenu');
    if (!menuButton || !dropdownMenu) return;

    menuButton.addEventListener('click', (event) => {
        event.stopPropagation();
        dropdownMenu.classList.toggle('show');
    });

    document.addEventListener('click', (event) => {
        const clickedInside = dropdownMenu.contains(event.target) || menuButton.contains(event.target);
        if (!clickedInside) {
            dropdownMenu.classList.remove('show');
        }
    });
}

document.addEventListener('DOMContentLoaded', initDropdown);

// Make sure your existing analyzeText function is still here
// async function analyzeText() { ... }

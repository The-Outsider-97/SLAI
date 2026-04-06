// Dropdown Menu Toggle
const menuButton = document.getElementById('menuButton');
const dropdownMenu = document.getElementById('dropdownMenu');

menuButton.onclick = function(event) {
    event.stopPropagation(); // Prevent click from closing menu immediately
    dropdownMenu.classList.toggle('show');
}

// Close the dropdown if the user clicks outside of it
window.onclick = function(event) {
    if (!event.target.matches('#menuButton')) {
        if (dropdownMenu.classList.contains('show')) {
            dropdownMenu.classList.remove('show');
        }
    }
}


// Make sure your existing analyzeText function is still here
// async function analyzeText() { ... }
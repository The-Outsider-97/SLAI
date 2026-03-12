
const { app, BrowserWindow } = require('electron')
const { spawn } = require('child_process')
const path = require('path')

let flaskProcess
let mainWindow

function createWindow() {
  // Start Flask backend
  flaskProcess = spawn('python', ['app.py'])
  
  mainWindow = new BrowserWindow({ width: 1200, height: 800 })
  mainWindow.loadFile('static/index.html')
}

document.addEventListener('DOMContentLoaded', function() {
    // --- Mobile Hamburger Menu Functionality ---
    const hamburger = document.getElementById('hamburger');
    const navLinks = document.getElementById('nav-links');

    if (hamburger && navLinks) {
        hamburger.addEventListener('click', () => {
            navLinks.classList.toggle('active');
        });
    }

    // --- Placeholder for Language Button ---
    const langButton = document.querySelector('.lang-btn');
    if(langButton) {
        langButton.addEventListener('click', (event) => {
           event.preventDefault();
           alert('Language selection feature coming soon!'); 
        });
    }
    // Dynamic card loading
    const path = window.location.pathname;
    let endpoint, cardType;
    
    if (path.includes('restaurants.html')) {
        endpoint = '/api/restaurants';
        cardType = 'restaurant';
    } else if (path.includes('indie.html')) {
        endpoint = '/api/indie';
        cardType = 'indie';
    }
    
    if (endpoint) {
        fetch(endpoint)
            .then(response => response.json())
            .then(data => {
                const container = document.getElementById('cards-container');
                container.innerHTML = data.map(card => 
                    generateCardHTML(card, cardType)
                ).join('');
            })
            .catch(error => {
                console.error('Error loading cards:', error);
                document.getElementById('cards-container').innerHTML = 
                    '<p class="error">Failed to load content. Please try again later.</p>';
            });
    }

    function generateCardHTML(card, type) {
        const ratingHTML = card.rating > 0 ? 
            `<span class="rating"><i class="fa-solid fa-star"></i> ${card.rating} (${card.review_count}+)</span>` : 
            '';
        
        const linkText = type === 'restaurant' ? 'View Menu' : 'View Offerings';
        
        return `
            <div class="card">
                <img src="${card.image_url}" alt="${card.alt_text}">
                <div class="card-content">
                    <h3>${card.name}</h3>
                    <p>${card.description}</p>
                    ${ratingHTML}
                    <a href="${type === 'restaurant' ? '#' : '#'}" class="card-link">${linkText}</a>
                </div>
            </div>
        `;
    } 
});


// Parallax effect for hero background
document.addEventListener('DOMContentLoaded', function () {
    window.addEventListener('scroll', function () {
        const scrolled = window.scrollY;
        const bg = document.querySelector('.hero-bg');
        if (bg) {
            bg.style.transform = `translateY(${scrolled * 0.3}px)`;
        }
    });
});


// theme manager
document.addEventListener('DOMContentLoaded', () => {
    // Load theme settings from localStorage
    const settings = JSON.parse(localStorage.getItem('userSettings')) || {};
    
    // Apply theme to body
    if (settings.theme) {
        document.body.setAttribute('data-theme', settings.theme);
    }
    
    // Apply theme to header elements
    const headerElements = document.querySelectorAll('.main-nav, .tab-link, .nav-icon');
    headerElements.forEach(el => {
        el.setAttribute('data-theme', settings.theme || 'light');
    });
});
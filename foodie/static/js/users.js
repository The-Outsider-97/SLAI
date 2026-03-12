const authContainer = document.getElementById('auth-container');
const profileLayout = document.getElementById('profile-layout');

// Font Families for Settings (Preserved from your original code)
const fontFamilies = [
    "Arial", "Verdana", "Tahoma", "Trebuchet MS", "Times New Roman", "Georgia",
    "Garamond", "Courier New", "Brush Script MT", "Comic Sans MS", "Impact",
    "Lucida Console", "Lucida Sans Unicode", "Palatino Linotype", "Book Antiqua",
    "Candara", "Calibri", "Optima", "Geneva", "Century Gothic", "Consolas",
    "Roboto", "Open Sans", "Lato", "Montserrat", "Butler", "Arial Black", "Helvetica Neue",
    "Segoe UI", "Ubuntu", "Droid Sans", "Source Sans Pro"
];

// --- Main Entry Point ---
document.addEventListener('DOMContentLoaded', () => {
    // This is the single entry point that controls the page's state.
    if (document.getElementById('user-content-wrapper')) {
        initializeUserProfilePage();
    }
});

/**
 * Checks for a logged-in user and displays the correct view (profile or auth forms).
 * This is the master controller for the page.
 */
function initializeUserProfilePage() {
    const userId = localStorage.getItem('userId');
    if (userId) {
        // User is logged in, so we show their profile.
        fetchAndDisplayUserProfile(userId);
    } else {
        // No user is logged in, so we show the login/register forms.
        displayAuthForms();
    }
    // ID to logout button
    const logoutBtn = document.createElement('button');
    logoutBtn.id = 'logout-btn';
    logoutBtn.className = 'tab-link';
    logoutBtn.innerHTML = '<i class="fa-solid fa-right-from-bracket"></i> Logout';
    document.querySelector('.tabs').appendChild(logoutBtn);
    
    // ID to welcome header
    document.querySelector('.page-header h1').id = 'welcome-header';
}




// --- 1. Authentication and View Switching Logic ---

function displayAuthForms() {
    authContainer.style.display = 'block';
    profileLayout.style.display = 'none';
    attachAuthFormListeners();
}

async function fetchAndDisplayUserProfile(userId) {
    try {
        const response = await fetch(`/api/user/profile?userId=${userId}`);
        if (!response.ok) {
            // If the user ID from localStorage is invalid, log them out.
            throw new Error('Your session has expired. Please log in again.');
        }
        const userData = await response.json();
        
        // Show profile view, hide auth forms
        document.getElementById('auth-container').style.display = 'none';
        document.getElementById('profile-layout').style.display = 'flex';

        // --- IMPORTANT: Now call all your original setup functions ---
        populateProfileData(userData);
        setupProfileTabs();
        setupSettings(); 
        setupDeleteAccount();

    } catch (error) {
        console.error('Error:', error);
        alert(error.message);
        handleLogout(); // Log out if profile fetch fails
    }
}

function attachAuthFormListeners() {
    const loginForm = document.getElementById('login-form');
    const registerForm = document.getElementById('register-form');
    loginForm.onsubmit = handleLogin;
    registerForm.onsubmit = handleRegister;
    document.getElementById('show-register').onclick = (e) => { e.preventDefault(); loginForm.style.display = 'none'; registerForm.style.display = 'block'; };
    document.getElementById('show-login').onclick = (e) => { e.preventDefault(); registerForm.style.display = 'none'; loginForm.style.display = 'block'; };
}

async function handleLogin(e) {
    e.preventDefault();
    const identifier = document.getElementById('login-identifier').value;
    const password = document.getElementById('login-password').value;
    const errorDiv = document.getElementById('login-error');
    errorDiv.style.display = 'none';

    try {
        const response = await fetch('/api/user/login', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ identifier, password })
        });
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || 'Login failed');
        localStorage.setItem('userId', data.userId);
        initializeUserProfilePage(); // Reload the page state
        // Show profile after login
        authContainer.style.display = 'block';
        profileLayout.style.display = 'flex';
        document.getElementById('welcome-header').textContent = `Welcome Back, ${data.username}!`;
    } catch (error) {
        errorDiv.textContent = error.message;
        errorDiv.style.display = 'block';
    }
}

async function handleRegister(e) {
    e.preventDefault();
    const errorDiv = document.getElementById('register-error');
    errorDiv.style.display = 'none';

    // --- Collect ALL form data ---
    const registrationData = {
        fullname: document.getElementById('register-fullname').value,
        email: document.getElementById('register-email').value,
        password: document.getElementById('register-password').value,
        age: document.getElementById('register-age').value,
        gender: document.getElementById('register-gender').value,
        phone: document.getElementById('register-phone').value,
        address: document.getElementById('register-address').value,
        card: document.getElementById('register-card').value,
        // Photo handling is complex, for now we just acknowledge it
        photo: document.getElementById('register-photo').files[0]?.name || null
    };

    // --- Client-side Password Validation ---
    const passRegex = /^(?=.*[A-Z])(?=.*[0-9])(?=.*[!@#$%^&*])(?=.{10,})/;
    if (!passRegex.test(registrationData.password)) {
        errorDiv.textContent = 'Password must be 10+ chars with an uppercase, a number, and a special symbol.';
        errorDiv.style.display = 'block';
        return;
    }

    try {
        const response = await fetch('/api/user/register', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(registrationData) // Send the complete data object
        });
        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || 'Registration failed');
        }
        
        alert('Registration successful! You have been awarded 300 points.');
        localStorage.setItem('userId', data.userId);
        initializeUserProfilePage(); // Reload state to show profile

    } catch (error) {
        errorDiv.textContent = error.message;
        errorDiv.style.display = 'block';
    }
}

function handleLogout() {
    localStorage.removeItem('userId');
    authContainer.style.display = 'block';
    profileLayout.style.display = 'none';
    
    // Reset forms
    document.getElementById('login-form').reset();
    document.getElementById('register-form').reset();
}

// Event listener for logout
document.addEventListener('click', (e) => {
    if (e.target.id === 'logout-btn') {
        handleLogout();
    }
});

// --- 2. Profile Data Population and UI Setup (Your Original Logic, Preserved) ---

function populateProfileData(userData) {
    document.querySelector('.page-header h1').textContent = `Welcome Back, ${userData.username}!`;

    const personalInfoTab = document.getElementById('personal-info');
    if (personalInfoTab) {
        personalInfoTab.querySelector('p:nth-of-type(1)').innerHTML = `<strong>Name:</strong> ${userData.username}`;
        personalInfoTab.querySelector('p:nth-of-type(2)').innerHTML = `<strong>Email:</strong> ${userData.email}`;
        
        const addressContainer = personalInfoTab.querySelector('.addresses-container');
        addressContainer.innerHTML = (userData.addresses && userData.addresses.length) ?
            userData.addresses.map(addr => `<p><strong>${addr.type}:</strong> ${addr.address} <a href="#" class="edit-link">Edit</a></p>`).join('') :
            '<p>No saved addresses.</p>';
        
        const paymentContainer = personalInfoTab.querySelector('.payments-container');
        paymentContainer.innerHTML = (userData.payment_methods && userData.payment_methods.length) ?
            userData.payment_methods.map(p => `<p><strong>${p.type}</strong> ending in ${p.last4} <a href="#" class="edit-link">Edit</a></p>`).join('') :
            '<p>No saved payment methods.</p>';
    }

    const orderContainer = document.querySelector('.orders-container');
    if (orderContainer) {
        orderContainer.innerHTML = (userData.orders && userData.orders.length) ?
            userData.orders.map(order => `
                <div class="order-item">
                    <p><strong>Order #${order.id}</strong> - ${order.restaurant} - ${order.amount} <a href="#" class="card-link">Reorder</a></p>
                    <small>Delivered on: ${order.date}</small>
                </div>
            `).join('') :
            '<p>No past orders.</p>';
    }

    const pointsElement = document.querySelector('.points-total');
    if (pointsElement) {
        pointsElement.textContent = userData.points || 0;
    }
}

function setupSettings() {
    const settingsTab = document.getElementById('settings');
    if (!settingsTab || settingsTab.querySelector('.user-settings')) return; // Prevents recreating the form
    
    settingsTab.querySelector('.profile-form').innerHTML = `
        <div class="user-settings">
            <h3>Account Settings</h3>
            <div class="setting-item">
                <label for="theme-preference">Theme:</label>
                <select id="theme-preference"><option value="dark">Dark</option><option value="light">Light</option></select>
            </div>
            <div class="setting-item">
                <label for="font-size">Font Size:</label>
                <select id="font-size">${[10, 12, 14, 16, 18, 20, 22, 24].map(s => `<option value="${s}" ${s === 16 ? 'selected' : ''}>${s}px</option>`).join('')}</select>
            </div>
            <div class="setting-item">
                <label for="font-family">Font Family:</label>
                <select id="font-family">${fontFamilies.map(f => `<option value="${f}">${f}</option>`).join('')}</select>
            </div>
            <div class="setting-item">
                <label for="animations-enabled">Enable Animations:</label>
                <input type="checkbox" id="animations-enabled" checked>
            </div>
            <div class="setting-item">
                <label for="notification-sound">Notification Sound:</label>
                <select id="notification-sound"><option value="off">Off</option><option value="default" selected>Default</option><option value="chime">Chime</option><option value="alert">Alert</option><option value="ding">Ding</option></select>
            </div>
            <div class="setting-item">
                <label for="time-format">Time Format:</label>
                <select id="time-format"><option value="12h" selected>12-hour</option><option value="24h">24-hour</option></select>
            </div>
            <button type="button" id="save-settings" class="submit-btn">Save Settings</button>
        </div>`;
    
    loadSettings();
    document.getElementById('save-settings').onclick = saveSettings;
    document.getElementById('theme-preference').onchange = (e) => document.body.setAttribute('data-theme', e.target.value);
}

function loadSettings() {
    const settings = JSON.parse(localStorage.getItem('userSettings')) || {};
    if (settings.theme) { document.body.setAttribute('data-theme', settings.theme); document.getElementById('theme-preference').value = settings.theme; }
    if (settings.fontSize) { document.getElementById('font-size').value = settings.fontSize; document.body.style.fontSize = `${settings.fontSize}px`; }
    if (settings.fontFamily) { document.getElementById('font-family').value = settings.fontFamily; document.body.style.fontFamily = settings.fontFamily; }
    if (settings.animationsEnabled !== undefined) { document.getElementById('animations-enabled').checked = settings.animationsEnabled; }
    if (settings.notificationSound) { document.getElementById('notification-sound').value = settings.notificationSound; }
    if (settings.timeFormat) { document.getElementById('time-format').value = settings.timeFormat; }
}

function saveSettings() {
    const settings = {
        theme: document.getElementById('theme-preference').value,
        fontSize: document.getElementById('font-size').value,
        fontFamily: document.getElementById('font-family').value,
        animationsEnabled: document.getElementById('animations-enabled').checked,
        notificationSound: document.getElementById('notification-sound').value,
        timeFormat: document.getElementById('time-format').value
    };
    document.body.style.fontSize = `${settings.fontSize}px`;
    document.body.style.fontFamily = settings.fontFamily;
    localStorage.setItem('userSettings', JSON.stringify(settings));
    saveSettingsToServer(settings);
    alert('Settings saved successfully!');
}

async function saveSettingsToServer(settings) {
    const userId = localStorage.getItem('userId');
    if (!userId) return;
    try {
        await fetch('/api/user/settings', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ userId, settings })
        });
    } catch (error) { console.error('Error saving settings:', error); }
}

function setupProfileTabs() {
    const tabLinks = document.querySelectorAll('.tab-link');
    tabLinks.forEach(tab => {
        tab.onclick = function(e) {
            e.preventDefault();
            if (this.id === 'logout-btn') { handleLogout(); return; }
            tabLinks.forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(c => c.style.display = 'none');
            this.classList.add('active');
            document.getElementById(this.getAttribute('data-tab')).style.display = 'block';
        };
    });
    if (tabLinks.length > 0) tabLinks[0].click();
}

function setupDeleteAccount() {
    const deleteButton = document.querySelector('.submit-btn-danger');
    if (deleteButton) {
        deleteButton.onclick = () => {
            if (confirm('Are you sure you want to delete your account? This action cannot be undone.')) {
                deleteUserAccount();
            }
        };
    }
}

async function deleteUserAccount() {
    const userId = localStorage.getItem('userId');
    if (!userId) return;
    try {
        const response = await fetch(`/api/user/delete`, {
            method: 'DELETE',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ userId })
        });
        if (response.ok) {
            alert('Account deleted successfully');
            handleLogout();
        } else {
            const data = await response.json();
            throw new Error(data.error || 'Failed to delete account');
        }
    } catch (error) {
        console.error('Error deleting account:', error);
        alert(`Error: ${error.message}`);
    }
}
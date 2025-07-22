// Add these functions at the bottom
function renderVendorCards(vendors) {
    const container = document.getElementById('vendor-results-container');
    container.innerHTML = '';
    
    vendors.forEach(vendor => {
        const card = document.createElement('div');
        card.className = 'card';
        card.dataset.vendorId = vendor.id;
        card.dataset.vendorType = vendor.type;
        
        card.innerHTML = `
            <img src="${vendor.image_url}" alt="${vendor.name}">
            <div class="card-content">
                <h3>${vendor.name}</h3>
                <p>${vendor.description}</p>
                <span class="rating"><i class="fa-solid fa-star"></i> ${vendor.rating} (${vendor.review_count}+)</span>
                <p><strong>Business Discount:</strong> ${vendor.business_discount}%</p>
                <button class="select-vendor card-link">Select Vendor</button>
            </div>
        `;
        container.appendChild(card);
    });
    
    // Add event listeners to select buttons
    document.querySelectorAll('.select-vendor').forEach(button => {
        button.addEventListener('click', function() {
            const card = this.closest('.card');
            selectVendor(
                card.dataset.vendorId,
                card.querySelector('h3').textContent
            );
        });
    });
}

function selectVendor(vendorId, vendorName) {
    document.getElementById('selected-vendor-name').textContent = vendorName;
    document.getElementById('order-confirmation').style.display = 'block';
    
    // Store selected vendor in form data
    document.getElementById('vendorId').value = vendorId;
    
    // Scroll to confirmation
    document.getElementById('order-confirmation').scrollIntoView({
        behavior: 'smooth'
    });
}

function filterVendors() {
    const searchTerm = document.getElementById('vendorSearch').value.toLowerCase();
    const filterType = document.getElementById('vendorTypeFilter').value;
    const cards = document.querySelectorAll('.card');
    
    cards.forEach(card => {
        const name = card.querySelector('h3').textContent.toLowerCase();
        const type = card.dataset.vendorType;
        const matchesSearch = name.includes(searchTerm);
        const matchesType = !filterType || type === filterType;
        
        card.style.display = (matchesSearch && matchesType) ? 'flex' : 'none';
    });
}

// Add to submit handler after formData creation
formData.vendor_id = ''; // Will be populated when vendor is selected

// Add to fetch .then() after showSuccessMessage
showSuccessMessage('Order submitted! Please select a vendor from the list below.');

// Show vendor section
document.getElementById('vendor-results').style.display = 'block';

// Fetch relevant vendors
fetch('/api/business_order/vendors', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
        employee_count: formData.employee_count,
        cuisine_preferences: formData.cuisine_preferences.split(','),
        location: formData.company_address
    })
})
.then(response => response.json())
.then(vendors => renderVendorCards(vendors))
.catch(error => console.error('Error loading vendors:', error));

// Add Finalize Order button handler
document.getElementById('finalize-order').addEventListener('click', function() {
    if (!document.getElementById('vendorId').value) {
        alert('Please select a vendor first');
        return;
    }
    
    // Submit finalized order
    fetch('/api/business_order/finalize', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(formData)
    })
    .then(response => response.json())
    .then(data => {
        showSuccessMessage(`Order confirmed with ${data.vendor_name}! ${data.message}`);
        businessOrderForm.reset();
        document.getElementById('vendor-results').style.display = 'none';
    })
    .catch(error => {
        console.error('Finalization error:', error);
        showErrorMessage(error.message || 'Error finalizing order');
    });
});

// Add filter handlers
document.getElementById('filterVendors').addEventListener('click', filterVendors);
document.getElementById('vendorSearch').addEventListener('input', filterVendors);
document.getElementById('vendorTypeFilter').addEventListener('change', filterVendors);

// Add hidden input for vendor ID in form
const vendorInput = document.createElement('input');
vendorInput.type = 'hidden';
vendorInput.id = 'vendorId';
vendorInput.name = 'vendorId';
businessOrderForm.appendChild(vendorInput);
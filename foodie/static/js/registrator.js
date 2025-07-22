
document.addEventListener('DOMContentLoaded', function() {
    const registrationForm = document.querySelector('.registration-section .contact-form');

    if (registrationForm) {
        // Get the submit button and remember its original HTML
        const submitBtn = registrationForm.querySelector('button[type="submit"]');
        const originalBtnText = submitBtn.innerHTML;

        registrationForm.addEventListener('submit', function(event) {
            event.preventDefault();

            // ==== ADD LOADING INDICATOR HERE ==== //
            submitBtn.disabled = true;
            submitBtn.innerHTML = '<i class="fa fa-spinner fa-spin"></i> Processing...';

            // Determine if it's indie registration
            const isIndie = document.querySelector('form.contact-form input[id="cuisineType"]') !== null && 
                          !document.querySelector('form.contact-form input[id="companyAddress"]');

            // Collect form data conditionally
            const sellerData = isIndie ? {
                name: document.getElementById('sellerName').value,
                contact_email: document.getElementById('sellerEmail').value,
                company_website: document.getElementById('companyWebsite').value || '',
                company_motto: document.getElementById('companyMotto').value,
                cuisine_type: document.getElementById('cuisineType').value,
                // Default values for indie sellers
                company_name: document.getElementById('sellerName').value,
                company_address: 'Home Kitchen',
                company_phone: '+297000000',
                registration_number: 'INDIE-' + Date.now(),
                contact_name: document.getElementById('sellerName').value
            } : {
                // Original restaurant data collection
                name: document.getElementById('restaurantName').value,
                company_name: document.getElementById('companyName').value,
                company_address: document.getElementById('companyAddress').value,
                company_phone: document.getElementById('companyPhone').value,
                registration_number: document.getElementById('registrationNumber').value,
                company_website: document.getElementById('companyWebsite').value || '',
                company_motto: document.getElementById('companyMotto').value,
                contact_name: document.getElementById('sellerName').value,
                contact_email: document.getElementById('sellerEmail').value,
                cuisine_type: document.getElementById('cuisineType').value
            };
            // API call
            fetch('/api/seller/register', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(sellerData)
            })
            .then(async response => {
                if (!response.ok) {
                    const err = await response.json();
                    throw new Error(err.error);
                }
                return response.json();
            })
            .then(data => {
                console.log('Success:', data);
                alert(`Registration successful! Your Seller ID is: ${data.profile.seller_id}`);
                registrationForm.reset();

                // Store seller ID and type
                localStorage.setItem('sellerId', data.profile.seller_id);
                localStorage.setItem('sellerType', isIndie ? 'indie' : 'restaurant');

                // Redirect to menu setup
                window.location.href = 'menu_setup.html';
            })
            .catch((error) => {
                console.error('Error:', error);
                alert(`Registration failed: ${error.message}`);

                submitBtn.disabled = false;
                submitBtn.innerHTML = originalBtnText;
            });
        });
    }
});
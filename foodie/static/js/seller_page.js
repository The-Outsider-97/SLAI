document.addEventListener('DOMContentLoaded', function() {
    const sellerId = localStorage.getItem('sellerId');
    const sellerType = localStorage.getItem('sellerType');
    
    if (!sellerId || !sellerType) {
        // This is handled by inline script in HTML, but kept here as a fallback.
        const errorDiv = document.getElementById('error-message');
        if (errorDiv) {
            errorDiv.textContent = 'Seller information missing. Redirecting...';
            errorDiv.style.display = 'block';
        }
        window.location.href = sellerType === 'indie' ? '../indie.html' : '../restaurants.html';
        return;
    }

    // Fetch seller data from server
    fetch(`/api/seller/${sellerId}`)
        .then(response => {
            if (!response.ok) {
                throw new Error(`Seller not found (ID: ${sellerId})`);
            }
            return response.json();
        })
        .then(seller => {
            // Populate seller info in Center Column
            document.getElementById('seller-name').textContent = seller.name;
            document.getElementById('seller-motto').textContent = seller.motto;
            document.getElementById('seller-banner-img').src = `../${seller.logo}`; // Assuming logo is banner
            
            // Generate rating stars
            const ratingContainer = document.getElementById('seller-rating');
            ratingContainer.innerHTML = '';
            const ratingValue = Math.round(seller.rating); // Round to nearest whole number
            for (let i = 0; i < 5; i++) {
                const star = document.createElement('i');
                star.className = i < ratingValue ? 'fa-solid fa-star' : 'fa-regular fa-star';
                ratingContainer.appendChild(star);
            }
            
            // Render menu items
            renderMenuItems(seller.menu);
            
            // Render reviews
            renderReviews(seller.reviews);
        })
        .catch(error => {
            console.error('Failed to load seller data:', error);
            const errorDiv = document.getElementById('error-message');
            if (errorDiv) {
                errorDiv.textContent = `Error: ${error.message}. Could not load seller page.`;
                errorDiv.style.display = 'block';
            }
        });
});

function renderMenuItems(menu) {
    const container = document.getElementById('menu-container');
    container.innerHTML = '';
    
    if (!menu || menu.length === 0) {
        container.innerHTML = '<p>This seller has not added any menu items yet.</p>';
        return;
    }
    
    // Group menu items by category
    const categorizedMenu = menu.reduce((acc, item) => {
        const category = item.category || 'Uncategorized';
        if (!acc[category]) {
            acc[category] = [];
        }
        acc[category].push(item);
        return acc;
    }, {});

    // Render each category and its items
    for (const category in categorizedMenu) {
        const categorySection = document.createElement('div');
        categorySection.className = 'menu-category';

        const categoryTitle = document.createElement('h2');
        categoryTitle.className = 'menu-category-title';
        categoryTitle.textContent = category.charAt(0).toUpperCase() + category.slice(1);
        
        const menuGrid = document.createElement('div');
        menuGrid.className = 'menu-grid';

        categorizedMenu[category].forEach(item => {
            const itemElement = document.createElement('div');
            itemElement.className = 'menu-item-card';
            itemElement.innerHTML = `
                <img src="../${item.image}" alt="${item.name}">
                <div class="menu-item-card-content">
                    <h3>${item.name}</h3>
                    <p>${item.description}</p>
                    <button class="menu-item-card-btn">See Options</button>
                </div>
            `;
            menuGrid.appendChild(itemElement);
        });

        categorySection.appendChild(categoryTitle);
        categorySection.appendChild(menuGrid);
        container.appendChild(categorySection);
    }
}

function renderReviews(reviews) {
    const container = document.getElementById('reviews-list');
    container.innerHTML = '';

    const exampleReviews = [
        { user: 'ChirpyDove4077', rating: 5, date: '02/20/2025', comment: 'I love the veggie mix over the top, it adds another layer of flavor!', avatar: { icon: 'fa-kiwi-bird', color: '#1DA1F2' } },
        { user: 'JazzySieve9415', rating: 5, date: '02/20/2025', comment: 'The crunch on the eggplant is my favorite part! It is a great way to enjoy eggplant!', avatar: { icon: 'fa-music', color: '#17BF63' } },
        { user: 'EagerPlate3761', rating: 4, date: '02/20/2025', comment: 'So well balanced and surprisingly easy to bring together. I love this method. It\'s as pleasantly crispy as fried eggplant.', avatar: { icon: 'fa-plate-wheat', color: '#F45D22' } }
    ];
    
    const reviewsToRender = (reviews && reviews.length > 0) ? reviews : exampleReviews;
    
    reviewsToRender.forEach(review => {
        const reviewElement = document.createElement('div');
        reviewElement.className = 'review';
        
        const ratingStars = '<i class="fa-solid fa-star"></i>'.repeat(review.rating) + 
                            '<i class="fa-regular fa-star"></i>'.repeat(5 - review.rating);
        
        const avatar = review.avatar || { icon: 'fa-user', color: '#777' };

        reviewElement.innerHTML = `
            <div class="avatar" style="background-color: ${avatar.color};">
                <i class="fa-solid ${avatar.icon}"></i>
            </div>
            <div class="review-content">
                <div class="review-header">
                    <span class="user-info">${review.user || 'Anonymous'}</span>
                    <span class="review-date">${review.date}</span>
                </div>
                <div class="rating">${ratingStars}</div>
                <p>${review.comment}</p>
                <button class="review-helpful"><i class="fa-regular fa-thumbs-up"></i> Helpful (0)</button>
            </div>
        `;
        container.appendChild(reviewElement);
    });
}
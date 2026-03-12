
let sellerId = localStorage.getItem('sellerId');
if (!sellerId) {
    alert('Seller registration incomplete. Redirecting to registration...');
    window.location.href = 'restaurants.html'; // Or appropriate registration page
}

document.addEventListener('DOMContentLoaded', function() {
    const menuItems = [];
    const ingredients = [];
    const tags = [];
    let sellerId = localStorage.getItem('sellerId');
    
    // DOM Elements
    const itemNameInput = document.getElementById('item-name');
    const itemPriceInput = document.getElementById('item-price');
    const itemImageInput = document.getElementById('item-image');
    const imagePreview = document.getElementById('image-preview');
    const ingredientInput = document.getElementById('ingredient-input');
    const ingredientsContainer = document.getElementById('ingredients-container');
    const addIngredientBtn = document.getElementById('add-ingredient');
    const tagInput = document.getElementById('tag-input');
    const tagsContainer = document.getElementById('tags-container');
    const addTagBtn = document.getElementById('add-tag');
    const addItemBtn = document.getElementById('add-item');
    const menuPreview = document.getElementById('menu-preview');
    const itemCountSpan = document.getElementById('item-count');
    const finishSetupBtn = document.getElementById('finish-setup');
    const priceComparison = document.getElementById('price-comparison');
    const comparisonResults = document.getElementById('comparison-results');
    const toggleNutritionBtn = document.getElementById('toggle-nutrition');
    const nutritionFields = document.getElementById('nutrition-fields');

    // Event listener for the image input to show a preview
    itemImageInput.addEventListener('change', function() {
        const file = this.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = function(e) {
                imagePreview.src = e.target.result;
                imagePreview.style.display = 'block';
            }
            reader.readAsDataURL(file);
        } else {
            imagePreview.src = '';
            imagePreview.style.display = 'none';
        }
    });
    
    // Add ingredients functionality
    addIngredientBtn.addEventListener('click', function() {
        const ingredientValue = ingredientInput.value.trim();
        if (ingredientValue && !ingredients.includes(ingredientValue)) {
            ingredients.push(ingredientValue);
            renderIngredients();
            ingredientInput.value = '';
        }
    });
    
    ingredientInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            addIngredientBtn.click();
            e.preventDefault();
        }
    });

    function renderIngredients() {
        ingredientsContainer.innerHTML = '';
        ingredients.forEach(ingredient => {
            const ingredientElement = document.createElement('div');
            ingredientElement.className = 'ingredient';
            ingredientElement.innerHTML = `
                ${ingredient}
                <span class="ingredient-remove" data-ingredient="${ingredient}">&times;</span>
            `;
            ingredientsContainer.appendChild(ingredientElement);
        });
        
        // Add remove functionality
        document.querySelectorAll('.ingredient-remove').forEach(btn => {
            btn.addEventListener('click', function() {
                const ingredientToRemove = this.getAttribute('data-ingredient');
                ingredients.splice(ingredients.indexOf(ingredientToRemove), 1);
                renderIngredients();
            });
        });
    }

    // Add tag functionality
    addTagBtn.addEventListener('click', function() {
        const tagValue = tagInput.value.trim();
        if (tagValue && !tags.includes(tagValue)) {
            tags.push(tagValue);
            renderTags();
            tagInput.value = '';
        }
    });
    
    tagInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            addTagBtn.click();
            e.preventDefault();
        }
    });
    
    function renderTags() {
        tagsContainer.innerHTML = '';
        tags.forEach(tag => {
            const tagElement = document.createElement('div');
            tagElement.className = 'tag';
            tagElement.innerHTML = `
                ${tag}
                <span class="tag-remove" data-tag="${tag}">&times;</span>
            `;
            tagsContainer.appendChild(tagElement);
        });
        
        // Add remove functionality
        document.querySelectorAll('.tag-remove').forEach(btn => {
            btn.addEventListener('click', function() {
                const tagToRemove = this.getAttribute('data-tag');
                tags.splice(tags.indexOf(tagToRemove), 1);
                renderTags();
            });
        });
    }


    // Toggle nutrition fields
    toggleNutritionBtn.addEventListener('click', function() {
        nutritionFields.style.display = nutritionFields.style.display === 'none' ? 'block' : 'none';
        this.innerHTML = nutritionFields.style.display === 'none' ? 
            '<i class="fa-solid fa-apple-alt"></i> Add Nutritional Information' :
            '<i class="fa-solid fa-times"></i> Hide Nutritional Information';
    });
    
    // Price comparison when item name changes
    itemNameInput.addEventListener('input', function() {
        const itemName = this.value.trim();
        if (itemName.length > 3) {
            fetchSimilarItems(itemName);
        } else {
            priceComparison.style.display = 'none';
        }
    });
    
    function fetchSimilarItems(itemName) {
        // Simulated API call - in real implementation, fetch from server
        setTimeout(() => {
            priceComparison.style.display = 'block';
            comparisonResults.innerHTML = `
                <div class="comparison-item">
                    <span>${itemName} at Gianni's</span>
                    <span>AFL 12.99</span>
                </div>
                <div class="comparison-item">
                    <span>Similar dish at Zeerovers</span>
                    <span>AFL 14.50</span>
                </div>
                <div class="comparison-item">
                    <span>Average price in your area</span>
                    <span>AFL 13.75</span>
                </div>
                <div class="comparison-item">
                    <span>Recommended price range</span>
                    <span><strong>AFL 12.00 - 15.00</strong></span>
                </div>
            `;
        }, 500);
    }
    
    // Add item to menu
    addItemBtn.addEventListener('click', function() {
        if (!validateForm()) return;
        
        const file = itemImageInput.files[0];
        // Use a Promise to handle the asynchronous file reading
        const imagePromise = new Promise((resolve, reject) => {
            if (file) {
                const reader = new FileReader();
                reader.onload = (e) => resolve(e.target.result); // Resolve with Base64 Data URL
                reader.onerror = (e) => reject(e);
                reader.readAsDataURL(file);
            } else {
                resolve('img/default_food.jpg'); // Resolve with default image path
            }
        });
        
        imagePromise.then(imageUrl => {
            const menuItem = {
                id: 'item_' + Date.now(),
                name: itemNameInput.value.trim(),
                description: document.getElementById('item-description').value.trim(),
                category: document.getElementById('item-category').value,
                size: document.getElementById('item-size').value,
                temperature: document.getElementById('item-temperature').value,
                price: parseFloat(itemPriceInput.value),
                ingredients: [...ingredients],
                tags: [...tags],
                image: imageUrl,
                nutrition: {
                    weight: document.getElementById('item-weight').value ? 
                        parseFloat(document.getElementById('item-weight').value) : null,
                    calories: document.getElementById('item-calories').value ? 
                        parseInt(document.getElementById('item-calories').value) : null,
                    sugar: document.getElementById('item-sugar').value ? 
                        parseInt(document.getElementById('item-sugar').value) : null,
                    protein: document.getElementById('item-protein').value ? 
                        parseInt(document.getElementById('item-protein').value) : null,
                    alcohol: document.getElementById('item-alcohol').value ? 
                        parseFloat(document.getElementById('item-alcohol').value) : null
                }
            };
            
            menuItems.push(menuItem);
            renderMenuPreview();
            resetForm();

            // Enable finish button if at least 5 items
            finishSetupBtn.disabled = menuItems.length < 5;
        }).catch(error => {
            console.error('Error reading image file:', error);
            alert('There was an error processing the image. Please try another one.');
        });
    });
    
    function validateForm() {
        if (!itemNameInput.value.trim()) {
            alert('Item name is required');
            return false;
        }
        
        if (!itemPriceInput.value || parseFloat(itemPriceInput.value) <= 0) {
            alert('Please enter a valid price');
            return false;
        }

        if (ingredients.length < 2) { 
            alert('Please add at least 2 ingredients');
            return false;
        }
        
        if (tags.length < 2) {
            alert('Please add at least 2 tags');
            return false;
        }
        
        return true;
    }
    
    function resetForm() {
        itemNameInput.value = '';
        document.getElementById('item-description').value = '';
        itemImageInput.value = ''; // Clears the selected file
        imagePreview.src = '';
        imagePreview.style.display = 'none';
        document.getElementById('item-category').value = '';
        document.getElementById('item-size').value = '';
        document.getElementById('item-temperature').value = '';
        document.getElementById('item-weight').value = '';
        document.getElementById('item-calories').value = '';
        document.getElementById('item-sugar').value = '';
        document.getElementById('item-protein').value = '';
        document.getElementById('item-alcohol').value = '';
        nutritionFields.style.display = 'none';
        toggleNutritionBtn.innerHTML = '<i class="fa-solid fa-apple-alt"></i> Add Nutritional Information';
        itemPriceInput.value = '';
        ingredients.length = 0;
        tags.length = 0;
        renderTags();
        priceComparison.style.display = 'none';
    }
    
    function renderMenuPreview() {
        itemCountSpan.textContent = menuItems.length;
        
        if (menuItems.length === 0) {
            menuPreview.innerHTML = `
                <div class="empty-state">
                    <i class="fa-solid fa-utensils"></i>
                    <p>No items added yet</p>
                </div>
            `;
            return;
        }
        
        menuPreview.innerHTML = '';
        
        // Group items by category
        const categories = {};
        menuItems.forEach(item => {
            if (!categories[item.category]) {
                categories[item.category] = [];
            }
            categories[item.category].push(item);
        });
        
        // Render each category
        for (const [category, items] of Object.entries(categories)) {
            const categoryHeader = document.createElement('h3');
            categoryHeader.textContent = category.toUpperCase();
            categoryHeader.style.gridColumn = '1 / -1';
            categoryHeader.style.marginTop = '20px';
            menuPreview.appendChild(categoryHeader);
            
            items.forEach(item => {
                const card = document.createElement('div');
                card.className = 'preview-card';
                card.innerHTML = `
                    <img src="${item.image}" alt="${item.name}">
                    <div class="preview-content">
                        <h4>${item.name}</h4>
                        <p>${item.description || 'No description'}</p>
                        <p><strong>AFL ${item.price.toFixed(2)}</strong></p>
                        <div class="tags">${item.tags.map(tag => `<span class="tag">${tag}</span>`).join('')}</div>
                        <button class="remove-item" data-id="${item.id}">
                            <i class="fa-solid fa-trash"></i> Remove
                        </button>
                    </div>
                `;
                menuPreview.appendChild(card);
            });
        }
        
        // Add remove functionality
        document.querySelectorAll('.remove-item').forEach(btn => {
            btn.addEventListener('click', function() {
                const itemId = this.getAttribute('data-id');
                const index = menuItems.findIndex(item => item.id === itemId);
                if (index !== -1) {
                    menuItems.splice(index, 1);
                    renderMenuPreview();
                    finishSetupBtn.disabled = menuItems.length < 5;
                }
            });
        });
    }
    
    // Finish setup
    finishSetupBtn.addEventListener('click', function() {
        // Send menu data to server
        saveMenuToServer();

    });
    
    // Menu setup
    function saveMenuToServer() {
        fetch('/api/seller/menu/setup', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                sellerId: sellerId,
                menuItems: menuItems.map(item => ({
                    id: 'item_' + Date.now(),
                    item_id: 'item_' + Date.now(),  // Add this for backend compatibility
                    name: itemNameInput.value.trim(),
                    description: document.getElementById('item-description').value.trim(),
                    category: document.getElementById('item-category').value,
                    size: document.getElementById('item-size').value,
                    temperature: document.getElementById('item-temperature').value,
                    price: parseFloat(itemPriceInput.value),
                    ingredients: [...ingredients],
                    tags: [...tags],
                    image: 'img/default_food.jpg',
                    is_available: true  // Add availability flag
                }))
            })
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
            alert('Menu setup complete! Your items are now live.');
            // Clear draft and redirect
            localStorage.removeItem('menuDraft');
            window.location.href = '/pages/seller_TEMPLATE.html';
        })
        .catch(error => {
            console.error('Error:', error);
            alert(`Failed to save menu: ${error.message}`);
        });
    }
    
    // Save draft functionality
    document.getElementById('save-draft').addEventListener('click', function() {
        localStorage.setItem('menuDraft', JSON.stringify(menuItems));
        alert('Draft saved successfully!');
    });
    
    // Load draft if exists
    const savedDraft = localStorage.getItem('menuDraft');
    if (savedDraft) {
        menuItems.push(...JSON.parse(savedDraft));
        renderMenuPreview();
        finishSetupBtn.disabled = menuItems.length < 5;
    }
});
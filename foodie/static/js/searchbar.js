
document.addEventListener('DOMContentLoaded', function() {
    
    // --- Global Search Bar (on index.html) ---
    const globalSearchForm = document.getElementById('search-form');
    if (globalSearchForm) {
        globalSearchForm.addEventListener('submit', function(event) {
            event.preventDefault();
            const searchInput = this.querySelector('.search-input');
            const query = searchInput.value.trim();

            if (query) {
                // Now, this will redirect to the restaurants page with the query
                console.log(`Searching for: ${query}`);
                window.location.href = `/restaurants.html?q=${encodeURIComponent(query)}`;
            }
        });
    }

    // --- In-Page Filtering (on restaurants.html, indie.html) ---
    const filterBar = document.querySelector('.filter-bar');
    if (filterBar) {
        const performFilter = () => {
            const filterSearchInput = filterBar.querySelector('.filter-search');
            const filterSelect = filterBar.querySelector('.filter-select');
            
            const searchTerm = filterSearchInput.value;
            const filters = {
                cuisine: filterSelect ? filterSelect.value : ''
            };

            // Call the backend API to get filtered results
            fetch('/api/search', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query: searchTerm, filters: filters })
            })
            .then(response => response.json())
            .then(results => {
                console.log('Search results from backend:', results);
                // Now, you would dynamically render these results into the .featured-grid
                // For this demo, we'll just log them.
                const grid = document.querySelector('.featured-grid');
                grid.innerHTML = '<h2>Search Results (See Console for Data)</h2>';
                results.forEach(item => {
                    // This is a very basic render, you can make it prettier
                    const doc = item[1]; // The document is the second element in the tuple
                    const card = document.createElement('div');
                    card.className = 'card';
                    card.innerHTML = `<div class="card-content"><h3>${doc.text.substring(0, 50)}...</h3><p>Score: ${item[0].toFixed(2)}</p></div>`;
                    grid.appendChild(card);
                });
            })
            .catch(error => {
                console.error('Error during search:', error);
                alert('An error occurred while searching.');
            });
        };

        // Automatically search when the page loads if there's a query in the URL
        const urlParams = new URLSearchParams(window.location.search);
        const queryFromUrl = urlParams.get('q');
        if (queryFromUrl) {
            filterBar.querySelector('.filter-search').value = queryFromUrl;
            performFilter();
        }

        // Add event listener to the filter button
        const filterButton = filterBar.querySelector('.filter-btn');
        filterButton.addEventListener('click', performFilter);
    }
});
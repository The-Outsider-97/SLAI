const map = L.map('world-map', {
    zoomControl: true,
    minZoom: 2,
    maxZoom: 10,
    worldCopyJump: true,
}).setView([20, 0], 2);

const streetLayer = L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '&copy; OpenStreetMap contributors',
    maxZoom: 19,
});

const terrainLayer = L.tileLayer('https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png', {
    attribution: '&copy; OpenStreetMap contributors, SRTM | &copy; OpenTopoMap',
    maxZoom: 17,
});

streetLayer.addTo(map);

L.control.layers(
    {
        'Street (OSM)': streetLayer,
        'Terrain (OpenTopoMap)': terrainLayer,
    },
    {},
    { position: 'topright' }
).addTo(map);

const countryInfo = document.getElementById('country-info');
let activeLayer = null;

const formatNumber = (value) => {
    if (!value || Number.isNaN(Number(value))) {
        return 'N/A';
    }
    return Number(value).toLocaleString();
};

const countryStyle = {
    color: '#314d72',
    weight: 0.9,
    fillColor: '#4f88d5',
    fillOpacity: 0.25,
};

const highlightStyle = {
    color: '#f27329',
    weight: 2,
    fillOpacity: 0.5,
};

function updateInfo(properties) {
    const name = properties.ADMIN || properties.name || 'Unknown';
    const iso = properties.ISO_A3 || properties.iso_a3 || 'N/A';
    const pop = properties.POP_EST || properties.population || null;
    const continent = properties.CONTINENT || properties.continent || 'N/A';

    countryInfo.innerHTML = `
        <h3>Selected Country</h3>
        <p><strong>Name:</strong> ${name}</p>
        <p><strong>ISO Code:</strong> ${iso}</p>
        <p><strong>Continent:</strong> ${continent}</p>
        <p><strong>Population (est.):</strong> ${formatNumber(pop)}</p>
    `;
}

function onEachCountry(feature, layer) {
    const props = feature.properties || {};
    const name = props.ADMIN || props.name || 'Unknown';
    const iso = props.ISO_A3 || props.iso_a3 || 'N/A';
    const continent = props.CONTINENT || props.continent || 'N/A';
    const pop = formatNumber(props.POP_EST || props.population || null);

    layer.bindPopup(`
        <div class="country-popup">
            <strong>${name}</strong><br>
            ISO: ${iso}<br>
            Continent: ${continent}<br>
            Population: ${pop}
        </div>
    `);

    layer.on({
        mouseover: (event) => {
            event.target.setStyle(highlightStyle);
        },
        mouseout: (event) => {
            if (activeLayer !== event.target) {
                countriesLayer.resetStyle(event.target);
            }
        },
        click: (event) => {
            if (activeLayer) {
                countriesLayer.resetStyle(activeLayer);
            }
            activeLayer = event.target;
            activeLayer.setStyle(highlightStyle);
            map.fitBounds(activeLayer.getBounds(), { padding: [25, 25], maxZoom: 5 });
            updateInfo(props);
        },
    });
}

let countriesLayer;

fetch('https://raw.githubusercontent.com/datasets/geo-countries/master/data/countries.geojson')
    .then((response) => {
        if (!response.ok) {
            throw new Error('Country data could not be loaded.');
        }
        return response.json();
    })
    .then((geojson) => {
        countriesLayer = L.geoJSON(geojson, {
            style: countryStyle,
            onEachFeature: onEachCountry,
        }).addTo(map);
    })
    .catch((error) => {
        countryInfo.innerHTML = `
            <h3>Selected Country</h3>
            <p>Unable to load country boundaries right now.</p>
            <p><small>${error.message}</small></p>
        `;
    });

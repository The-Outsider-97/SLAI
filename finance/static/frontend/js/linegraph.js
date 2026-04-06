

document.addEventListener("DOMContentLoaded", function () {
    const selector = document.getElementById("stockSelector");
    const ctx = document.getElementById("stockLineChart").getContext("2d");
    let lineChart;

    // Fetch the stock list dynamically
    fetch("/api/stocks/list")
        .then(res => res.json())
        .then(stockList => {
            stockList.forEach(stock => {
                const option = document.createElement("option");
                option.value = stock.symbol;
                option.textContent = `${stock.symbol} - ${stock.name}`;
                selector.appendChild(option);
            });
        })
        .catch(err => console.error("Failed to load stock list:", err));

    // Chart rendering logic
    selector.addEventListener("change", function () {
        const symbol = selector.value;
        if (!symbol) return;

        fetch(`/api/batch_data/${symbol}`)
            .then(res => res.json())
            .then(data => {
                if (data.error) {
                    alert(data.error);
                    return;
                }
                const labels = data.map(entry =>
                    new Date(entry.timestamp * 1000).toLocaleTimeString()
                );
                const prices = data.map(entry => entry.price);

                if (lineChart) lineChart.destroy(); // Refresh chart
                lineChart = new Chart(ctx, {
                    type: 'line',
                    data: {
                        labels: labels,
                        datasets: [{
                            label: `${symbol} Price`,
                            data: prices,
                            fill: false,
                            borderColor: 'blue',
                            tension: 0.1
                        }]
                    },
                    options: {
                        responsive: true,
                        scales: {
                            x: {
                                title: { display: true, text: 'Time' }
                            },
                            y: {
                                title: { display: true, text: 'Price ($)' }
                            }
                        }
                    }
                });
            })
            .catch(err => console.error("Failed to load batch data:", err));
    });
});
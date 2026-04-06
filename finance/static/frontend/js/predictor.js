
let fetchInProgress = false;

// Ensure these are in a scope accessible by event listeners
function openModal() {
    const modal = document.getElementById('explanationModal');
    if (modal) modal.style.display = 'block';
}

function closeModal() {
    const modal = document.getElementById('explanationModal');
    if (modal) modal.style.display = 'none';
    const content = document.getElementById('explanation-content');
    if (content) content.textContent = 'Loading explanation...';
}

function applySorting() {
    const sortSelector = document.getElementById('sortSelector');
    if (!sortSelector) return;
    const sortBy = sortSelector.value;
    fetch(`/api/predictions?sort=${sortBy}`)
        .then(response => response.json())
        .then(sortedData => {
            if (sortedData.error) {
                console.error("Error sorting predictions:", sortedData.details);
                return;
            }
            if (typeof updatePredictionTable === "function") {
                updatePredictionTable(sortedData);
            }
        })
        .catch(error => console.error('Error applying sorting:', error));
}

function updateSentimentDisplay() {
    const sentimentSymbolEl = document.getElementById('sentiment-symbol');
    const sentimentValueEl = document.getElementById('sentiment-value');
    if (!sentimentSymbolEl || !sentimentValueEl) return; // Elements might not be on every page

    fetch('/api/trends')
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                sentimentSymbolEl.textContent = "Error";
                sentimentValueEl.textContent = data.details || "N/A";
                return;
            }
            if (data.symbol_sentiment && Object.keys(data.symbol_sentiment).length > 0) {
                const symbol = Object.keys(data.symbol_sentiment)[0];
                const score = data.symbol_sentiment[symbol];
                sentimentSymbolEl.textContent = symbol || "N/A";
                sentimentValueEl.textContent = typeof score === 'number' ? score.toFixed(2) : (score || "N/A");
            } else {
                sentimentSymbolEl.textContent = "N/A";
                sentimentValueEl.textContent = "N/A";
            }
        })
        .catch(error => {
            console.error('Error fetching sentiment:', error);
            if (sentimentSymbolEl) sentimentSymbolEl.textContent = "Error";
            if (sentimentValueEl) sentimentValueEl.textContent = "Fetch Error";
        });
}

function fetchAndDisplayPredictions() { // For index.html prediction table
    fetch('/api/predictions')
        .then(response => {
            if (!response.ok) throw new Error(`HTTP ${response.status} - ${response.statusText}`);
            return response.json();
        })
        .then(predictionsData => {
            const predictionTableBody = document.getElementById('prediction-data');
            if (!predictionTableBody) return;
            predictionTableBody.innerHTML = '';

            if (predictionsData.error) {
                predictionTableBody.innerHTML = `<tr><td colspan="16">Error loading predictions: ${predictionsData.details}</td></tr>`;
                return;
            }
            if (!predictionsData || predictionsData.length === 0) {
                predictionTableBody.innerHTML = '<tr><td colspan="16">No predictions available.</td></tr>';
                return;
            }

            predictionsData.forEach(item => {
                const row = `
                    <tr class="prediction-row" data-symbol="${item["Ticker Symbol"] || ''}">
                        <td>${item["Stocks"] || 'N/A'}</td>
                        <td>${item["Ticker Symbol"] || 'N/A'}</td>
                        <td>${item["Price ($)"] || 'N/A'}</td>
                        <td>${item["MarketCap"] || 'N/A'}</td>
                        <td>${item["24h Volume"] || 'N/A'}</td>
                        <td>${item["1h Prediction"] || 'N/A'}</td>
                        <td>${item["24h Prediction"] || 'N/A'}</td>
                        <td>${item["1w Prediction"] || 'N/A'}</td>
                        <td>${item["ELR Component"] || 'N/A'}</td>
                        <td>${item["Sentiment Impact"] || 'N/A'}</td>
                        <td>${item["Uncertainty"] || 'N/A'}</td>
                        <td>${item["p_value"] || 'N/A'}</td>
                        <td>${item["Critical Value"] || 'N/A'}</td>
                        <td>${item["Confidence"] || 'N/A'}</td>
                        <td>${item["Lower Bound"] || 'N/A'}</td>
                        <td>${item["Upper Bound"] || 'N/A'}</td>
                    </tr>
                `;
                predictionTableBody.innerHTML += row;
            });
        })
        .catch(error => {
            console.error('Prediction refresh failed:', error);
            const predictionTableBody = document.getElementById('prediction-data');
            if (predictionTableBody) predictionTableBody.innerHTML = `<tr><td colspan="16">Failed to refresh predictions: ${error.message}</td></tr>`;
            showNotification('⚠️ Failed to refresh predictions', 'error');
        });
}


// Compliance disclosure fetching
function fetchComplianceDisclosures() {
    fetch('/api/compliance/disclosures')
        .then(response => response.json())
        .then(data => {
            const complianceDiv = document.getElementById('compliance-disclosures');
            if (complianceDiv) {
                if (data.error) {
                    complianceDiv.innerHTML = `<p>Error loading compliance information: ${data.details}</p>`;
                    return;
                }
                complianceDiv.innerHTML = `
                    <p><strong>Risk Statement:</strong> ${data.risk_statement || "N/A"}</p>
                    <p>Model Registry ID: ${data.model_registry || "N/A"}</p>
                    <p>Data Sources: ${(data.data_sources || []).join(', ') || "N/A"}</p>
                    <p>Last Compliance Check: ${data.last_compliance_check || "N/A"}</p>
                `;
            }
        })
        .catch(error => {
            console.error('Error fetching compliance disclosures:', error);
            const complianceDiv = document.getElementById('compliance-disclosures');
            if (complianceDiv) {
                complianceDiv.innerHTML = `<p>Error loading compliance information.</p>`;
            }
        });
}

// Click handler for explanation (if explanation modal exists and rows have .prediction-row & data-symbol)
document.addEventListener('click', (e) => {
    const explanationModal = document.getElementById('explanationModal');
    if (!explanationModal) return; // Only proceed if modal exists

    const row = e.target.closest('.prediction-row');
    if (row && row.dataset.symbol) {
        const symbol = row.dataset.symbol;
        const explanationContentEl = document.getElementById('explanation-content');
        if (explanationContentEl) {
            explanationContentEl.textContent = `Fetching explanation for ${symbol}...`;
            openModal();
            fetch(`/api/explain/prediction/${symbol}`)
                .then(response => response.json())
                .then(explanation => {
                    if (explanation.error) {
                        explanationContentEl.textContent = `Error loading explanation: ${explanation.details || explanation.error}`;
                    } else {
                        explanationContentEl.textContent = JSON.stringify(explanation, null, 2);
                    }
                })
                .catch(error => {
                    console.error(`Error fetching explanation for ${symbol}:`, error);
                    explanationContentEl.textContent = `Error loading explanation: ${error.message}`;
                });
        }
    }
});


function showNotification(message, type = 'info') {
    const existing = document.getElementById('refresh-notification');
    if (existing) existing.remove();
    
    const notification = document.createElement('div');
    notification.id = 'refresh-notification';
    notification.className = `notification ${type}`;
    notification.textContent = message;
    notification.style.cssText = `
        position: fixed; bottom: 20px; right: 20px;
        padding: 10px 20px; border-radius: 4px;
        background: ${type === 'error' ? '#ff6b6b' : '#4ecdc4'};
        color: white; z-index: 1000; box-shadow: 0 2px 10px rgba(0,0,0,0.2);
    `;
    
    document.body.appendChild(notification);
    setTimeout(() => notification.remove(), 5000);
}

function fetchAndDisplayIndicators() {
    fetch('/api/indicators')
        .then(response => response.json())
        .then(indicatorsData => {
            const indicatorsTableBody = document.getElementById('indicators-data');
            if (!indicatorsTableBody) return;
            indicatorsTableBody.innerHTML = '';
            if (indicatorsData.error) {
                indicatorsTableBody.innerHTML = `<tr><td colspan="4">Error loading indicators: ${indicatorsData.details}</td></tr>`;
                return;
            }
            if (!indicatorsData || indicatorsData.length === 0) {
                indicatorsTableBody.innerHTML = '<tr><td colspan="4">No indicators available.</td></tr>';
                return;
            }
            indicatorsData.forEach(item => {
                const row = `
                    <tr>
                        <td>${item["Indicator"] || 'N/A'}</td>
                        <td>${item["Value"] || 'N/A'}</td>
                        <td>${item["Status"] || 'N/A'}</td>
                        <td>${item["Thresholds"] || 'N/A'}</td>
                    </tr>
                `;
                indicatorsTableBody.innerHTML += row;
            });
        })
        .catch(error => {
            console.error('Error fetching indicators:', error);
            const indicatorsTableBody = document.getElementById('indicators-data');
            if (indicatorsTableBody) {
                indicatorsTableBody.innerHTML = `<tr><td colspan="4">Error loading indicators: ${error.message}</td></tr>`;
            }
        });
}

function fetchAndDisplayDiagnostics() {
    fetch('/api/diagnostics')
        .then(response => response.json())
        .then(diagnosticsData => {
            const diagnosticsTableBody = document.getElementById('diagnostics-data');
            if (!diagnosticsTableBody) return;
            diagnosticsTableBody.innerHTML = '';
             if (diagnosticsData.error) {
                diagnosticsTableBody.innerHTML = `<tr><td colspan="2">Error loading diagnostics: ${diagnosticsData.details}</td></tr>`;
                return;
            }
            if (Object.keys(diagnosticsData).length === 0 ) {
                diagnosticsTableBody.innerHTML = '<tr><td colspan="2">No diagnostics available.</td></tr>';
                return;
            }
            for (const metric in diagnosticsData) {
                const row = `
                    <tr>
                        <td>${metric}</td>
                        <td>${diagnosticsData[metric]}</td>
                    </tr>
                `;
                diagnosticsTableBody.innerHTML += row;
            }
        })
        .catch(error => {
            console.error('Error fetching diagnostics:', error);
            const diagnosticsTableBody = document.getElementById('diagnostics-data');
             if (diagnosticsTableBody) {
                diagnosticsTableBody.innerHTML = `<tr><td colspan="2">Error loading diagnostics: ${error.message}</td></tr>`;
            }
        });
}


function fetchAndDisplaySignals() {
    fetch('/api/signals')
        .then(response => response.json())
        .then(data => {
            const signalsTableBody = document.querySelector('#signals table tbody');
            if (!signalsTableBody) return;
            signalsTableBody.innerHTML = '';

            if (data.error) {
                signalsTableBody.innerHTML = `<tr><td colspan="4">Error loading signals: ${data.details}</td></tr>`;
                return;
            }
            if (!data || data.length === 0) {
                signalsTableBody.innerHTML = '<tr><td colspan="4">No signals available.</td></tr>';
                return;
            }
            data.forEach(item => {
                const row = `
                    <tr>
                        <td>${item.source || 'N/A'}</td>
                        <td>${item.activity || 'N/A'}</td>
                        <td>${item.impact || 'N/A'}</td>
                        <td>${item.confidence || 'N/A'}</td>
                    </tr>
                `;
                signalsTableBody.innerHTML += row;
            });
        })
        .catch(error => {
            console.error('Error fetching signals:', error);
             const signalsTableBody = document.querySelector('#signals table tbody');
             if (signalsTableBody) {
                signalsTableBody.innerHTML = `<tr><td colspan="4">Error loading signals: ${error.message}</td></tr>`;
            }
        });
}

function fetchAndDisplayLearningInsights() {
    fetch('/api/learning_insights')
        .then(response => response.json())
        .then(data => {
            const learningTableBody = document.getElementById('learning-data');
            if (!learningTableBody) return;
            learningTableBody.innerHTML = '';

            if (data.error) {
                learningTableBody.innerHTML = `<tr><td colspan="2">Error loading learning insights: ${data.details}</td></tr>`;
                return;
            }
             if (Object.keys(data).length === 0 ) {
                learningTableBody.innerHTML = '<tr><td colspan="2">No learning insights available.</td></tr>';
                return;
            }
            
            const alsData = data.adaptive_learner || {};
            // const laData = data.learning_agent || {}; // Not directly used in provided HTML structure

            const metrics = [
                { name: "Volatility Adjustment", value: alsData.learning_rate ? alsData.learning_rate.toFixed(4) : (alsData.volatility_adjustment || "N/A") }, // Assuming learning_rate is the adjustment
                { name: "Ensemble Diversity", value: alsData.model_diversity ? alsData.model_diversity.toFixed(3) : (alsData.ensemble_diversity || "N/A") },
                { name: "Recent Errors (Mean)", value: alsData.recent_error ? alsData.recent_error.toFixed(4) : "N/A" },
                { name: "Current Learning Rate", value: alsData.learning_rate ? alsData.learning_rate.toFixed(5) : "N/A" },
                { name: "Current Model version", value: "ALS-1.0" } // Example, or get from data if available
            ];

            metrics.forEach(metric => {
                const row = `
                    <tr>
                        <td>${metric.name}</td>
                        <td>${metric.value}</td>
                    </tr>
                `;
                learningTableBody.innerHTML += row;
            });
        })
        .catch(error => {
            console.error('Error fetching learning insights:', error);
            const learningTableBody = document.getElementById('learning-data');
            if (learningTableBody) {
                learningTableBody.innerHTML = `<tr><td colspan="2">Error loading learning insights: ${error.message}</td></tr>`;
            }
        });
}

// This function populates the "Portfolio Metrics" section on index.html
function fetchAndDisplayPortfolioMetrics() { // For index.html's portfolio section
    fetch('/api/portfolio')
        .then(response => response.json())
        .then(data => {
            const portfolioTableBody = document.getElementById('portfolio-metrics-data');
            if (!portfolioTableBody) return;
            portfolioTableBody.innerHTML = '';

            if (data.error) {
                portfolioTableBody.innerHTML = `<tr><td colspan="2">Error loading portfolio metrics: ${data.details}</td></tr>`;
                return;
            }

            const backtestMetrics = data.backtest_metrics || {};
            const metricsToDisplay = [
                { name: "Current Total Portfolio Value", value: typeof data.current_total_portfolio_value_incl_unrealized === 'number' ? `$${data.current_total_portfolio_value_incl_unrealized.toFixed(2)}` : "N/A" },
                { name: "Cash Balance", value: typeof data.current_portfolio_value_cash === 'number' ? `$${data.current_portfolio_value_cash.toFixed(2)}` : "N/A" },
                { name: "Total Unrealized P&L", value: typeof data.unrealized_pnl_total === 'number' ? `$${data.unrealized_pnl_total.toFixed(2)}` : "N/A" },
                { name: "Monthly P&L", value: typeof data.monthly_pnl === 'number' ? `$${data.monthly_pnl.toFixed(2)}` : "N/A" },
                { name: "Monthly Profit Goal Status", value: (data.financial_goals?.monthly_profit?.status || "N/A").replace(/_/g, ' ') },
                { name: "Drawdown Limit Status", value: (data.financial_goals?.max_drawdown?.status || "N/A").replace(/_/g, ' ') },
                { name: "Backtest Sharpe Ratio", value: backtestMetrics.sharpe_ratio || "N/A" },
                { name: "Backtest Max Drawdown", value: backtestMetrics.max_drawdown_backtest || "N/A" },
                // { name: "Backtest Mean Return", value: backtestMetrics.mean_return_backtest || "N/A" }, // Already in backtest table if separate
            ];

            metricsToDisplay.forEach(metric => {
                const row = `<tr><td>${metric.name}</td><td>${metric.value}</td></tr>`;
                portfolioTableBody.innerHTML += row;
            });
        })
        .catch(error => {
            console.error('Error fetching portfolio metrics for index page:', error);
            const portfolioTableBody = document.getElementById('portfolio-metrics-data');
            if (portfolioTableBody) {
                portfolioTableBody.innerHTML = `<tr><td colspan="2">Error loading portfolio metrics: ${error.message}</td></tr>`;
            }
        });
}

function fetchAndDisplayCulturalTrends() {
    fetch('/api/trends')
        .then(response => response.json())
        .then(data => {
            const trendsTableBody = document.getElementById('trends-data');
            if (!trendsTableBody) return;
            trendsTableBody.innerHTML = '';

            if (data.error) {
                trendsTableBody.innerHTML = `<tr><td colspan="2">Error loading cultural trends: ${data.details}</td></tr>`;
                return;
            }
            if (Object.keys(data).length === 0 ) {
                trendsTableBody.innerHTML = '<tr><td colspan="2">No cultural trends available.</td></tr>';
                return;
            }
            
            let symbol = "N/A";
            if (data.symbol_sentiment && Object.keys(data.symbol_sentiment).length > 0) {
                symbol = Object.keys(data.symbol_sentiment)[0];
            } else if (data.symbol_correlation && Object.keys(data.symbol_correlation).length > 0) {
                symbol = Object.keys(data.symbol_correlation)[0];
            }

            const topTrends = data.top_trend_terms || {};
            const topTrendsString = Object.entries(topTrends)
                                      .map(([term, score]) => `${term}: ${Number(score).toFixed(2)}`)
                                      .join(', ') || "N/A";

            const metrics = [
                { name: `Latest Sentiment (${symbol})`, value: data.symbol_sentiment ? (data.symbol_sentiment[symbol] || "N/A") : "N/A" },
                { name: `Recent Correlation (${symbol})`, value: data.symbol_correlation ? (data.symbol_correlation[symbol] || "N/A") : "N/A" },
                { name: "Top Trend Terms (TF-IDF)", value: topTrendsString },
                { name: "Emerging Narratives", value: Array.isArray(data.emerging_narratives) && data.emerging_narratives.length > 0 ? data.emerging_narratives.join(', ') : "N/A" }
            ];

            metrics.forEach(metric => {
                const row = `
                    <tr>
                        <td>${metric.name}</td>
                        <td>${metric.value}</td>
                    </tr>
                `;
                trendsTableBody.innerHTML += row;
            });
        })
        .catch(error => {
            console.error('Error fetching cultural trends:', error);
            const trendsTableBody = document.getElementById('trends-data');
            if (trendsTableBody) {
                trendsTableBody.innerHTML = `<tr><td colspan="2">Error loading cultural trends: ${error.message}</td></tr>`;
            }
        });
}

function fetchAndDisplayDataQuality() {
    fetch('/api/data_quality')
        .then(response => response.json())
        .then(data => {
            const dataQualityTableBody = document.getElementById('data-quality-data');
            if (!dataQualityTableBody) return;
            dataQualityTableBody.innerHTML = '';

            if (data.error) {
                dataQualityTableBody.innerHTML = `<tr><td colspan="2">Error loading data quality: ${data.details}</td></tr>`;
                return;
            }
             if (Object.keys(data).length === 0) {
                dataQualityTableBody.innerHTML = '<tr><td colspan="2">No data quality information available.</td></tr>';
                return;
            }

            const metrics = [
                { name: "Source Reliability", value: data.source_reliability || "N/A" },
                { name: "Freshness Score", value: data.freshness_score || "N/A" },
                { name: "Fallback Triggers", value: data.fallback_triggers === undefined ? "N/A" : data.fallback_triggers }
            ];

            metrics.forEach(metric => {
                const row = `
                    <tr>
                        <td>${metric.name}</td>
                        <td>${metric.value}</td>
                    </tr>
                `;
                dataQualityTableBody.innerHTML += row;
            });
        })
        .catch(error => {
            console.error('Error fetching data quality:', error);
             const dataQualityTableBody = document.getElementById('data-quality-data');
            if (dataQualityTableBody) {
                dataQualityTableBody.innerHTML = `<tr><td colspan="2">Error loading data quality: ${error.message}</td></tr>`;
            }
        });
}

function fetchAndDisplayUltimateSuggestions() {
        fetch('/api/ultimate_suggestions')
            .then(response => {
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                return response.json();
            })
            .then(data => {
                const suggestionsTableBody = document.getElementById('ultimate-suggestions-data');
                const portfolioValueEl = document.getElementById('portfolio-value-suggestions');
                const timestampEl = document.getElementById('suggestions-timestamp');

                if (!suggestionsTableBody || !portfolioValueEl || !timestampEl) return;

                if (data.error) {
                    suggestionsTableBody.innerHTML = `<tr><td colspan="7">Error loading suggestions: ${data.details}</td></tr>`;
                    portfolioValueEl.textContent = "Error";
                    timestampEl.textContent = "Error";
                    return;
                }
                portfolioValueEl.textContent = data.current_portfolio_value_display || "N/A";
                timestampEl.textContent = data.timestamp || "N/A";
                suggestionsTableBody.innerHTML = ''; 

                if (!data.suggestions || data.suggestions.length === 0) {
                    suggestionsTableBody.innerHTML = '<tr><td colspan="5">No ultimate suggestions available at this time.</td></tr>';
                    return;
                }

                data.suggestions.forEach(item => {
                    const row = `
                        <tr>
                            <td>${item.type || 'N/A'}</td>
                            <td>${item.stock_name || 'N/A'} (${item.stock_symbol || 'N/A'})</td>
                            <td>${item.action || 'N/A'}</td>
                            <td>${item.confidence_pct !== undefined ? item.confidence_pct.toFixed(2) + '%' : 'N/A'}</td>
                            <td>${item.expected_gain_loss_pct !== undefined ? item.expected_gain_loss_pct.toFixed(2) + '%' : 'N/A'}</td>
                            <td>${item.current_price !== undefined ? item.current_price.toFixed(2) : 'N/A'}</td>
                            <td>${item.target_price_24h !== undefined ? item.target_price_24h.toFixed(2) : 'N/A'}</td>
                        </tr>
                    `;
                    suggestionsTableBody.innerHTML += row;
                });
            })
            .catch(error => {
                console.error('Error fetching ultimate suggestions:', error);
                const suggestionsTableBody = document.getElementById('ultimate-suggestions-data');
                if (suggestionsTableBody) suggestionsTableBody.innerHTML = `<tr><td colspan="7">Error loading suggestions: ${error.message}</td></tr>`;
                const portfolioValueEl = document.getElementById('portfolio-value-suggestions');
                if (portfolioValueEl) portfolioValueEl.textContent = "Error";
                const timestampEl = document.getElementById('suggestions-timestamp');
                if (timestampEl) timestampEl.textContent = "Error";
            });
    }

function updateMarketStatus() {
    fetch('/api/market/status')
        .then(response => response.json())
        .then(data => {
            const banner = document.getElementById('market-status-banner');
            if (!banner) return;
            
            if (data.error) {
                banner.innerHTML = `<div class="market-closed">Market Status: Error (${data.details})</div>`;
                banner.className = 'market-closed-banner'; // Use closed style for error
                return;
            }
            if (data.is_open) {
                banner.innerHTML = '<div class="market-open">Market is OPEN</div>';
                banner.className = 'market-open-banner';
            } else {
                banner.innerHTML = '<div class="market-closed">Market is CLOSED</div>';
                banner.className = 'market-closed-banner';
            }
        })
        .catch(error => {
            console.error('Error fetching market status:', error);
            const banner = document.getElementById('market-status-banner');
            if (banner) {
                 banner.innerHTML = `<div class="market-closed">Market Status: Fetch Error</div>`;
                 banner.className = 'market-closed-banner';
            }
        });
}

// This function is specifically for portfolio.html
function fetchAndDisplayPortfolioPageData() {
    fetch('/api/portfolio')
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            if (data.error) {
                throw new Error(data.details);
            }
            const setText = (elementId, value, prefix = '', postfix = '',toFixedValue = null) => {
                const el = document.getElementById(elementId);
                if (el) {
                    if (value === undefined || value === null || (typeof value === 'number' && isNaN(value))) {
                        el.textContent = 'N/A';
                    } else if (typeof value === 'number' && toFixedValue !== null) {
                        el.textContent = `${prefix}${value.toFixed(toFixedValue)}${postfix}`;
                    } else if (typeof value === 'number') {
                        el.textContent = `${prefix}${value}${postfix}`;
                    }
                    else {
                        el.textContent = `${prefix}${value}${postfix}`;
                    }
                } else {
                    console.warn(`Element with ID '${elementId}' not found.`);
                }
            };

            if (data.error) {
                console.error("Error from /api/portfolio for portfolio.html:", data.details);
                const errorMsg = `Error: ${data.details || data.error}`;
                setText('current-portfolio-value', errorMsg);
                setText('cash-balance', errorMsg);
                setText('total-unrealized-pnl', errorMsg);
                setText('monthly-pnl', errorMsg);
                setText('monthly-pnl-target', errorMsg);
                setText('current-drawdown', errorMsg);
                setText('max-drawdown-limit', errorMsg);
                setText('monthly-peak-value', errorMsg);
                const goalsStatusBody = document.getElementById('financial-goals-status');
                if (goalsStatusBody) goalsStatusBody.innerHTML = `<tr><td colspan="3">${errorMsg}</td></tr>`;
                const openPositionsBody = document.getElementById('open-positions-data');
                if (openPositionsBody) openPositionsBody.innerHTML = `<tr><td colspan="7">${errorMsg}</td></tr>`;
                setText('default-symbol', errorMsg);
                setText('latest-price-default-symbol', errorMsg);
                setText('daily-profit-target', errorMsg);
                setText('current-risk-factor', errorMsg);
                setText('market-regime', errorMsg);
                setText('last-daily-maintenance', errorMsg);
                return;
            }

            // Portfolio Summary
            setText('current-portfolio-value', data.current_total_portfolio_value_incl_unrealized, '$', '', 2);
            setText('cash-balance', data.current_portfolio_value_cash, '$', '', 2);
            setText('total-unrealized-pnl', data.unrealized_pnl_total, '$', '', 2);
            setText('monthly-pnl', data.monthly_pnl, '$', '', 2);
            setText('monthly-pnl-target', data.financial_goals?.monthly_profit?.target, '$', '', 2);
            setText('current-drawdown', data.current_drawdown_percentage, '', '%', 2);
            setText('max-drawdown-limit', (data.financial_goals?.max_drawdown?.limit || 0) * 100, '', '%', 1);
            setText('monthly-peak-value', data.monthly_peak_portfolio_value, '$', '', 2);

            // Financial Goals Status
            const goalsStatusBody = document.getElementById('financial-goals-status');
            if (goalsStatusBody) {
                goalsStatusBody.innerHTML = ''; // Clear previous
                if (data.financial_goals && Object.keys(data.financial_goals).length > 0) {
                    for (const [goalKey, goalDetails] of Object.entries(data.financial_goals)) {
                        let targetValueText = 'N/A';
                        let statusText = (goalDetails.status || 'N/A').replace(/_/g, ' ');

                        if (goalKey === 'monthly_profit') {
                            targetValueText = `Target: $${Number(goalDetails.target || 0).toFixed(2)}`;
                        } else if (goalKey === 'max_drawdown') {
                            targetValueText = `Limit: ${(Number(goalDetails.limit || 0) * 100).toFixed(1)}%`;
                        } else if (goalKey === 'sentiment_risk') {
                            targetValueText = `Threshold: ${goalDetails.threshold || 'N/A'}, Factor: ${goalDetails.risk_factor || 'N/A'}`;
                            // For sentiment_risk, the 'status' might not be a simple string like "ok" or "breached".
                            // It could be a boolean indicating if the risk factor is active, or the sentiment score itself.
                            // Adjust based on how you define `sentiment_risk.status` in the backend.
                            // For now, using the provided status or 'N/A'.
                            statusText = goalDetails.status || 'N/A';
                        }
                        const row = `<tr>
                                        <td>${goalKey.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}</td>
                                        <td>${targetValueText}</td>
                                        <td>${statusText}</td>
                                     </tr>`;
                        goalsStatusBody.innerHTML += row;
                    }
                } else {
                     goalsStatusBody.innerHTML = '<tr><td colspan="3">Financial goals data not available.</td></tr>';
                }
            }

            // Open Positions
            const openPositionsBody = document.getElementById('open-positions-data');
            if (openPositionsBody) {
                openPositionsBody.innerHTML = '';
                if (data.open_positions_detailed && data.open_positions_detailed.length > 0) {
                    data.open_positions_detailed.forEach(pos => {
                        const pnlClass = (pos.unrealized_pnl || 0) >= 0 ? 'profit' : 'loss';
                        const row = `<tr>
                                        <td>${pos.symbol || 'N/A'}</td>
                                        <td>${(pos.direction || 'N/A').toUpperCase()}</td>
                                        <td>${(pos.entry_price || 0).toFixed(2)}</td>
                                        <td>${(pos.current_price || 0).toFixed(2)}</td>
                                        <td>$${(pos.size_value || 0).toFixed(2)}</td>
                                        <td>${(pos.shares || 0).toFixed(4)}</td>
                                        <td class="${pnlClass}">${(pos.unrealized_pnl || 0).toFixed(2)}</td>
                                        <td>${pos.timestamp || 'N/A'}</td>
                                     </tr>`;
                        openPositionsBody.innerHTML += row;
                    });
                } else {
                    openPositionsBody.innerHTML = '<tr><td colspan="7">No open positions.</td></tr>';
                }
            }
            
            // Current Trading Parameters
            setText('default-symbol', data.default_symbol_for_analysis);
            setText('latest-price-default-symbol', data.latest_price_of_default_symbol_for_state, '$', '', 2);
            setText('daily-profit-target', data.daily_profit_target, '$', '', 2);
            setText('current-risk-factor', data.current_risk_factor, '', '', 2);
            setText('market-regime', (data.market_regime || "N/A").replace(/_/g, ' '));
            setText('last-daily-maintenance', data.last_daily_maintenance_date);

        })
        .catch(error => {
            console.error('Full error object for /api/portfolio fetch:', error);
            document.getElementById('portfolio-summary').innerHTML = 
                `<div class="error">Error loading portfolio data: ${error.message}</div>`;
            const errorMsg = `Data Fetch Error: ${error.message || 'Unknown error'}`;
            const idsToUpdateOnError = [
                'current-portfolio-value', 'cash-balance', 'total-unrealized-pnl', 
                'monthly-pnl', 'monthly-pnl-target', 'current-drawdown', 'max-drawdown-limit',
                'monthly-peak-value', 'daily-profit-target', 'current-risk-factor', 
                'market-regime', 'default-symbol', 'latest-price-default-symbol', 'last-daily-maintenance'
            ];
            idsToUpdateOnError.forEach(id => {
                const el = document.getElementById(id);
                if (el) el.textContent = errorMsg;
            });
            const goalsStatusBody = document.getElementById('financial-goals-status');
            if (goalsStatusBody) goalsStatusBody.innerHTML = `<tr><td colspan="3">${errorMsg}</td></tr>`;
            const openPositionsBody = document.getElementById('open-positions-data');
            if (openPositionsBody) openPositionsBody.innerHTML = `<tr><td colspan="7">${errorMsg}</td></tr>`;

        });
}

function fetchAndDisplayBacktestingMetrics() {
    fetch('/api/portfolio')
        .then(response => response.json())
        .then(data => {
            const backtestTableBody = document.getElementById('backtesting-metrics-data');
            if (!backtestTableBody) return;
            backtestTableBody.innerHTML = '';

            if (data.error) {
                backtestTableBody.innerHTML = `<tr><td colspan="2">Error: ${data.details}</td></tr>`;
                return;
            }

            const metrics = data.backtest_metrics;
            if (!metrics || Object.keys(metrics).length === 0) {
                backtestTableBody.innerHTML = '<tr><td colspan="2">Metrics not available yet.</td></tr>';
                return;
            }

            const metricsMap = {
                "Sharpe Ratio": metrics.sharpe_ratio,
                "Sortino Ratio": metrics.sortino_ratio,
                "Max Drawdown": metrics.max_drawdown,
                "Win Rate": metrics.win_rate,
                "Profit Factor": metrics.profit_factor,
                "Mean Return": metrics.mean_return,
                "Volatility": metrics.volatility,
                "Calmar Ratio": metrics.calmar_ratio
            };

            for (const [key, value] of Object.entries(metricsMap)) {
                if (value !== undefined && value !== null) {
                    const row = `<tr><td>${key}</td><td>${value}</td></tr>`;
                    backtestTableBody.innerHTML += row;
                }
            }
        })
        .catch(error => console.error('Error fetching backtesting metrics:', error));
    }

function fetchAndDisplayOptimizationResults() {
    fetch('/api/portfolio/optimization')
        .then(response => response.json())
        .then(data => {
            const optTableBody = document.getElementById('optimization-results-data');
            if (!optTableBody) return;
            optTableBody.innerHTML = ''; // Clear previous results

            if (!data || Object.keys(data).length === 0 || !data.best_params) {
                optTableBody.innerHTML = '<tr><td colspan="2">Optimization data not yet available.</td></tr>';
                return;
            }

            const params = data.best_params;
            const paramsMap = {
                "Best Strategy": params.strategy,
                "Optimal Position Size": params.position_size,
                "Optimal Stop Loss": params.stop_loss,
                "Optimal Take Profit": params.take_profit,
                "Resulting Sharpe Ratio": params.sharpe ? params.sharpe.toFixed(3) : 'N/A'
            };

            for (const [key, value] of Object.entries(paramsMap)) {
                if (value !== undefined) {
                    const row = `<tr><td>${key}</td><td>${value}</td></tr>`;
                    optTableBody.innerHTML += row;
                }
            }
        })
        .catch(error => {
            console.error('Error fetching optimization results:', error);
            const optTableBody = document.getElementById('optimization-results-data');
            if(optTableBody) optTableBody.innerHTML = '<tr><td colspan="2">Failed to load optimization data.</td></tr>';
        });
}

// This function is called for index.html
function fetchAllData() {
    if (fetchInProgress) return;
    fetchInProgress = true;
    console.log("Refreshing all dashboard data for index.html...");
    
    updateMarketStatus();
    fetchComplianceDisclosures();
    
    fetchAndDisplayPredictions();
    updateSentimentDisplay();
    fetchAndDisplayUltimateSuggestions();
    fetchAndDisplaySignals();
    fetchAndDisplayIndicators();
    fetchAndDisplayDiagnostics();
    fetchAndDisplayLearningInsights();
    fetchAndDisplayPortfolioMetrics(); // Populates the portfolio summary on index.html
    fetchAndDisplayCulturalTrends();
    fetchAndDisplayDataQuality();
    fetchAndDisplayBacktestingMetrics();
    fetchAndDisplayOptimizationResults();
    
    setTimeout(() => { fetchInProgress = false; }, 5000);
}

document.addEventListener('DOMContentLoaded', () => {
    if (typeof initTheme === 'function') initTheme();
    if (typeof initDropdown === 'function') initDropdown();
    
    const path = window.location.pathname;

    if (path.endsWith('/portfolio') || path.endsWith('/portfolio.html')) {
        if (typeof fetchAndDisplayPortfolioPageData === 'function') {
            fetchAndDisplayPortfolioPageData();
            setInterval(fetchAndDisplayPortfolioPageData, 30000);
        } else {
            console.error("CRITICAL: fetchAndDisplayPortfolioPageData function not found in predictor.js. Portfolio page will not load data.");
        }
        // Common elements for portfolio page
        if (typeof fetchComplianceDisclosures === 'function') fetchComplianceDisclosures();
        if (typeof updateMarketStatus === 'function') {
            updateMarketStatus();
            setInterval(updateMarketStatus, 60000);
        }
    } else if (path.endsWith('/') || path.endsWith('/index.html') || path.endsWith('/predictions') || path.endsWith('/predictions.html') || path === '') {
        if (typeof fetchAllData === 'function') {
            fetchAllData(); 
            setInterval(fetchAllData, 300000);
        } else {
            console.error("CRITICAL: fetchAllData function not found in predictor.js. Index page will not load data.");
        }
    } else {
        console.log(`On page ${path}, loading common elements if functions exist.`);
        if (typeof fetchComplianceDisclosures === 'function') fetchComplianceDisclosures();
        if (typeof updateMarketStatus === 'function') {
            updateMarketStatus();
            setInterval(updateMarketStatus, 60000);
        }
    }
});
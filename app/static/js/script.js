// Logger Setup
const logger = {
    debug: (...args) => console.log('[DEBUG]', ...args),
    error: (...args) => console.error('[ERROR]', ...args),
    info: (...args) => console.info('[INFO]', ...args)
};

// State
let sessionId = localStorage.getItem('session_id') || null;
let sentimentChart = null;
const maxLength = 500;
const baseUrl = window.location.origin;

document.addEventListener("DOMContentLoaded", function () {
    // DOM Elements
    const textInput = document.getElementById("text");
    const charCounter = document.getElementById("charCounter");
    const clearButton = document.getElementById("clearButton");
    const submitBtn = document.getElementById("submitBtn");
    const resultSection = document.getElementById("resultSection");
    const sentimentForm = document.getElementById("sentimentForm");
    const exportBtn = document.getElementById("exportBtn");
    const clearHistoryBtn = document.getElementById("clearHistoryBtn");
    const toggleHistory = document.getElementById("toggleHistory");
    const historyContent = document.getElementById("historyContent");
    const toggleIcon = document.getElementById("toggleIcon");

    // Initialize UI
    updateUI();
    initHistorySidebar();
    loadHistory();
    fetchAnalytics();

    // Event Listeners
    if (textInput) {
        textInput.addEventListener("input", updateUI);
        textInput.addEventListener("keydown", function (event) {
            if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
                event.preventDefault();
                sentimentForm.dispatchEvent(new Event("submit"));
            }
            if (event.key === "Escape") {
                event.preventDefault();
                clearButton.click();
            }
        });
    }

    if (clearButton) {
        clearButton.addEventListener("click", () => {
            textInput.value = "";
            textInput.focus();
            updateUI();
            resetResultSection();
        });
    }

    if (sentimentForm) {
        sentimentForm.addEventListener("submit", async function (event) {
            event.preventDefault();
            const text = textInput.value.trim();
            if (!text) return;

            setLoading(true);
            try {
                const timestamp = new Date().toISOString().slice(0, 19).replace('T', ' ');
                const response = await fetch(`${baseUrl}/predict`, {
                    method: "POST",
                    headers: { 
                        "Content-Type": "application/json",
                        "X-Session-ID": sessionId || ""
                    },
                    body: JSON.stringify({ text, timestamp }),
                });

                const data = await response.json();
                
                if (response.status === 429) {
                    throw new Error("Slow down! You've reached the request limit. Please wait a minute before trying again.");
                }
                
                if (!response.ok) throw new Error(data.error || "Server Error");

                if (data.session_id) {
                    sessionId = data.session_id;
                    localStorage.setItem('session_id', sessionId);
                }

                showResult(data);
                loadHistory();
                fetchAnalytics();
            } catch (error) {
                logger.error("Analysis failed:", error);
                showError(error.message);
            } finally {
                setLoading(false);
            }
        });
    }

    if (exportBtn) {
        exportBtn.addEventListener("click", () => {
            window.location.href = '/export';
        });
    }

    if (clearHistoryBtn) {
        clearHistoryBtn.addEventListener("click", async () => {
            if (confirm("Are you sure you want to clear all history?")) {
                try {
                    const response = await fetch('/clear-history', { method: 'POST' });
                    if (response.ok) {
                        localStorage.removeItem('sentiment_history');
                        fetchAnalytics();
                        loadHistory();
                    }
                } catch (error) {
                    logger.error("Failed to clear history:", error);
                }
            }
        });
    }

    // --- Functions ---

    function updateUI() {
        if (!textInput || !charCounter || !clearButton) return;
        const currentLength = textInput.value.length;
        charCounter.textContent = `${currentLength} / ${maxLength}`;
        clearButton.style.display = currentLength > 0 ? 'block' : 'none';
    }

    function setLoading(isLoading) {
        if (!submitBtn || !resultSection) return;
        if (isLoading) {
            submitBtn.disabled = true;
            submitBtn.innerHTML = `<span>Processing...</span>`;
            resultSection.innerHTML = `
                <div class="placeholder-content">
                    <div class="pulse-circle" style="animation-duration: 0.8s; background: #e0e7ff; color: #6366f1;">⚡</div>
                    <h3>Analyzing Emotions...</h3>
                    <p>Consulting the neural network</p>
                </div>
            `;
        } else {
            submitBtn.disabled = false;
            submitBtn.innerHTML = `
                <span>Analyze Text</span>
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M21 11.5a8.38 8.38 0 0 1-.9 3.8 8.5 8.5 0 0 1-7.6 4.7 8.38 8.38 0 0 1-3.8-.9L3 21l1.9-5.7a8.38 8.38 0 0 1-.9-3.8 8.5 8.5 0 0 1 4.7-7.6 8.38 8.38 0 0 1 3.8-.9h.5a8.48 8.48 0 0 1 8 8v.5z"/>
                </svg>
            `;
        }
    }

    function showResult(data) {
        if (!resultSection) return;
        const sentiment = data.sentiment;
        const confidence = (data.confidence * 100).toFixed(1);
        const isPositive = sentiment.toLowerCase() === "positive";
        const color = isPositive ? "#10b981" : "#ef4444";
        const emoji = isPositive ? "🎉" : "😔";
        const msg = isPositive ? "Great vibes detected!" : "Negative sentiment detected.";

        resultSection.innerHTML = `
            <div style="animation: fadeInUp 0.5s ease forwards;">
                <div class="emoji-xl">${emoji}</div>
                <span class="result-tag" style="background: ${color}15; color: ${color}; border: 1px solid ${color}30;">
                    ${sentiment.toUpperCase()}
                </span>
                <div class="stat-box">
                    <div style="display:flex; justify-content:space-between; margin-bottom:8px;">
                        <span style="font-weight:600; color:#64748b;">Confidence Score</span>
                        <span style="font-weight:700; color:${color}">${confidence}%</span>
                    </div>
                    <div class="progress-track">
                        <div class="progress-fill" style="width: 0%; background: ${color}"></div>
                    </div>
                    <p style="margin-top:1rem; font-size:0.95rem; color:#64748b;">${msg}</p>
                </div>
                <button onclick="window.resetResultSection()" class="try-again-btn">Try Another</button>
            </div>
        `;

        setTimeout(() => {
            const bar = resultSection.querySelector(".progress-fill");
            if (bar) bar.style.width = `${confidence}%`;
        }, 100);
    }

    function showError(msg) {
        if (!resultSection) return;
        resultSection.innerHTML = `
            <div class="placeholder-content">
                <div class="pulse-circle" style="background: #fee2e2; color: #ef4444; animation: none;">⚠️</div>
                <h3 style="color: #ef4444">Analysis Failed</h3>
                <p>${msg}</p>
                <button onclick="window.resetResultSection()" class="try-again-btn">Try Again</button>
            </div>
        `;
    }

    window.resetResultSection = function() {
        if (!resultSection) return;
        resultSection.innerHTML = `
            <div class="placeholder-content">
                <div class="pulse-circle">🔍</div>
                <h3>Ready to Analyze</h3>
                <p>Your results will appear here with detailed confidence scoring.</p>
            </div>
        `;
    };

    function initHistorySidebar() {
        if (!toggleHistory || !historyContent) return;
        toggleHistory.addEventListener('click', function() {
            const isCollapsed = historyContent.style.display === 'none';
            historyContent.style.display = isCollapsed ? 'block' : 'none';
            toggleIcon.textContent = isCollapsed ? '▼' : '▲';
        });
    }

    async function loadHistory() {
        const historyList = document.getElementById('historyList');
        const historyEmpty = document.getElementById('historyEmpty');
        const historyLoading = document.getElementById('historyLoading');
        if (!historyList || !historyEmpty || !historyLoading) return;

        try {
            historyLoading.style.display = 'block';
            historyEmpty.style.display = 'none';
            historyList.innerHTML = '';

            const response = await fetch(`${baseUrl}/history`, {
                headers: { "X-Session-ID": sessionId || "" }
            });
            
            if (response.status === 429) {
                throw new Error("Rate limit exceeded for history. Please wait.");
            }
            
            if (!response.ok) throw new Error("Failed to load history");

            const data = await response.json();
            historyLoading.style.display = 'none';

            if (!data.predictions || data.predictions.length === 0) {
                historyEmpty.style.display = 'block';
                return;
            }

            data.predictions.forEach(prediction => {
                const card = createHistoryCard(prediction);
                historyList.appendChild(card);
            });
        } catch (error) {
            logger.error("Load history failed:", error);
            historyLoading.style.display = 'none';
            historyEmpty.style.display = 'block';
        }
    }

    function createHistoryCard(prediction) {
        const card = document.createElement('div');
        card.className = 'history-card';
        
        const inputData = typeof prediction.input_data === 'string' ? JSON.parse(prediction.input_data) : prediction.input_data;
        const result = typeof prediction.prediction_result === 'string' ? JSON.parse(prediction.prediction_result) : prediction.prediction_result;
        
        const text = inputData.text || '';
        const sentiment = result.sentiment || 'Unknown';
        const confidence = result.confidence || 0;
        const isFavorite = prediction.is_favorite || false;
        const timestamp = new Date(prediction.timestamp).toLocaleString();
        
        const emoji = sentiment.toLowerCase() === "positive" ? "😊" : "😔";
        const sentimentClass = sentiment.toLowerCase() === "positive" ? "positive" : "negative";

        card.innerHTML = `
            <div class="history-card-header">
                <div class="history-card-sentiment ${sentimentClass}">
                    <span>${emoji}</span>
                    <span>${sentiment}</span>
                </div>
                <button class="favorite-btn ${isFavorite ? 'favorited' : ''}" data-id="${prediction.id}">
                    ${isFavorite ? '⭐' : '☆'}
                </button>
            </div>
            <div class="history-card-text">${text.substring(0, 100)}${text.length > 100 ? '...' : ''}</div>
            <div class="history-card-footer">
                <span>${(confidence * 100).toFixed(1)}%</span>
                <span>${timestamp}</span>
            </div>
        `;

        card.querySelector('.favorite-btn').addEventListener('click', async (e) => {
            e.stopPropagation();
            await toggleFavorite(prediction.id, e.currentTarget);
        });

        return card;
    }

    async function toggleFavorite(id, btn) {
        try {
            const response = await fetch(`${baseUrl}/favorite/${id}`, {
                method: "POST",
                headers: { "X-Session-ID": sessionId || "" }
            });
            
            if (response.status === 429) {
                logger.error("Rate limit hit for favorites");
                return;
            }
            
            if (response.ok) {
                const data = await response.json();
                btn.classList.toggle('favorited', data.is_favorite);
                btn.textContent = data.is_favorite ? '⭐' : '☆';
            }
        } catch (error) {
            logger.error("Toggle favorite failed:", error);
        }
    }

    async function fetchAnalytics() {
        try {
            const response = await fetch('/analytics');
            if (!response.ok) return;
            const data = await response.json();
            if (data.trends && data.trends.length > 0) {
                initChart(data.trends);
            }
        } catch (error) {
            logger.error("Fetch analytics failed:", error);
        }
    }

    function initChart(data) {
        const ctx = document.getElementById('sentimentChart');
        if (!ctx) return;

        const chartData = {
            datasets: [{
                label: 'Sentiment Trend',
                data: data.map(item => ({
                    x: new Date(item.timestamp),
                    y: item.sentiment === 'positive' ? 1 : 0
                })),
                borderColor: '#6366f1',
                backgroundColor: 'rgba(99, 102, 241, 0.1)',
                borderWidth: 3,
                tension: 0.3,
                fill: true
            }]
        };

        if (sentimentChart) sentimentChart.destroy();
        sentimentChart = new Chart(ctx.getContext('2d'), {
            type: 'line',
            data: chartData,
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: { type: 'time', title: { display: true, text: 'Time' } },
                    y: { 
                        min: -0.5, max: 1.5, 
                        ticks: { 
                            stepSize: 0.5,
                            callback: value => value === 1 ? 'Positive' : value === 0 ? 'Negative' : ''
                        } 
                    }
                }
            }
        });
    }
});

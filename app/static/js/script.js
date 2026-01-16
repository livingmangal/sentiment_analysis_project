
const logger = {
    debug: (...args) => console.log('[DEBUG]', ...args),
    error: (...args) => console.error('[ERROR]', ...args),
    info: (...args) => console.info('[INFO]', ...args)
};


let sessionId = localStorage.getItem('session_id') || null;
let sentimentChart = null;
const maxLength = 500;
const baseUrl = window.location.origin;

document.addEventListener("DOMContentLoaded", function () {

    
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

   
    updateUI();
    initHistorySidebar();
    loadHistory();
    fetchAnalytics();

    

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
                    <div class="pulse-circle">⚡</div>
                    <h3>Analyzing Emotions...</h3>
                    <p>Consulting the neural network</p>
                </div>
            `;
        } else {
            submitBtn.disabled = false;
            submitBtn.innerHTML = `<span>Analyze Text</span>`;
        }
    }

    

    function showResult(data) {
        if (!resultSection) return;

        const sentiment = data.sentiment;
        const confidence = (data.confidence * 100).toFixed(1);
        const isPositive = sentiment.toLowerCase() === "positive";
        const color = isPositive ? "#10b981" : "#ef4444";

        resultSection.innerHTML = `
            <div>
                <span style="color:${color}; font-weight:700;">
                    ${sentiment.toUpperCase()} (${confidence}%)
                </span>

                <div style="margin-top:10px;">
                    <button id="copyBtn">Copy</button>
                    <span id="copyFeedback" style="margin-left:8px;"></span>
                </div>

                <button onclick="window.resetResultSection()" class="try-again-btn">
                    Try Another
                </button>
            </div>
        `;

        
        const copyBtn = document.getElementById("copyBtn");
        const feedback = document.getElementById("copyFeedback");

        copyBtn.onclick = async () => {
            try {
                await navigator.clipboard.writeText(
                    `${sentiment.toUpperCase()} (${confidence}%)`
                );
                feedback.innerText = "Copied!";
                setTimeout(() => feedback.innerText = "", 1500);
            } catch {
                feedback.innerText = "Copy failed";
            }
        };
    }

    function showError(msg) {
        if (!resultSection) return;
        resultSection.innerHTML = `
            <div class="placeholder-content">
                <h3>Analysis Failed</h3>
                <p>${msg}</p>
                <button onclick="window.resetResultSection()" class="try-again-btn">
                    Try Again
                </button>
            </div>
        `;
    }

    window.resetResultSection = function () {
        if (!resultSection) return;
        resultSection.innerHTML = `
            <div class="placeholder-content">
                <div class="pulse-circle">🔍</div>
                <h3>Ready to Analyze</h3>
                <p>Your results will appear here.</p>
            </div>
        `;
    };

    

    function initHistorySidebar() {
        if (!toggleHistory || !historyContent) return;
        toggleHistory.addEventListener('click', function () {
            const isCollapsed = historyContent.style.display === 'none';
            historyContent.style.display = isCollapsed ? 'block' : 'none';
            toggleIcon.textContent = isCollapsed ? '▼' : '▲';
        });
    }

    async function loadHistory() {
        try {
            await fetch(`${baseUrl}/history`, {
                headers: { "X-Session-ID": sessionId || "" }
            });
        } catch (error) {
            logger.error("Load history failed:", error);
        }
    }

    async function fetchAnalytics() {
        try {
            const response = await fetch('/analytics');
            if (!response.ok) return;
            const data = await response.json();
            if (data.trends) initChart(data.trends);
        } catch (error) {
            logger.error("Fetch analytics failed:", error);
        }
    }

    function initChart(data) {
        const ctx = document.getElementById('sentimentChart');
        if (!ctx) return;

        if (sentimentChart) sentimentChart.destroy();
        sentimentChart = new Chart(ctx, {
            type: 'line',
            data: {
                datasets: [{
                    label: 'Sentiment Trend',
                    data: data.map(item => ({
                        x: new Date(item.timestamp),
                        y: item.sentiment === 'positive' ? 1 : 0
                    }))
                }]
            }
        });
    }
});

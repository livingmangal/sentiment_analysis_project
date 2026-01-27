
const logger = {
    debug: (...args) => console.log('[DEBUG]', ...args),
    error: (...args) => console.error('[ERROR]', ...args),
    info: (...args) => console.info('[INFO]', ...args)
};


const ERROR_MESSAGES = {
    NO_FILE: {
        title: "No file selected",
        suggestion: "Please select a CSV file before uploading."
    },
    INVALID_FILE_TYPE: {
        title: "Invalid file type",
        suggestion: "Only CSV files (.csv) are supported."
    },
    INVALID_CONTENT_TYPE: {
        title: "Invalid request format",
        suggestion: "Please refresh the page and try again."
    },
    TEXT_TOO_LONG: {
        title: "Text too long",
        suggestion: "Please shorten the text to under 1000 characters."
    },
    SERVER_ERROR: {
        title: "Server error",
        suggestion: "Please try again later."
    },
    PREDICTION_FAILED: {
        title: "Prediction failed",
        suggestion: "Please try again in a moment."
    },
    DEFAULT: {
        title: "Something went wrong",
        suggestion: "Please try again or refresh the page."
    }
};

// State
let sessionId = localStorage.getItem('session_id');
if (!sessionId) {
    sessionId = 'sess-' + Math.random().toString(36).substring(2, 15)
             + Math.random().toString(36).substring(2, 15);
    localStorage.setItem('session_id', sessionId);
}

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

    updateUI();
    initHistorySidebar();
    loadHistory();
    fetchAnalytics();

    /* =====================================================
       Sentiment Form Submit
    ===================================================== */
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
                    body: JSON.stringify({ text, timestamp })
                });

                const data = await response.json();

                if (response.status === 429) {
                    showFriendlyError("DEFAULT");
                    return;
                }

                if (!response.ok) {
                    showFriendlyError(data?.error?.code || "DEFAULT");
                    return;
                }

                if (data.session_id) {
                    sessionId = data.session_id;
                    localStorage.setItem("session_id", sessionId);
                }

                showResult(data);
                loadHistory();
                fetchAnalytics();

            } catch (error) {
                logger.error("Analysis failed:", error);
                showFriendlyError("DEFAULT");
            } finally {
                setLoading(false);
            }
        });
    }

    /* =====================================================
       ✅ NEW: Friendly Error Renderer
    ===================================================== */
    function showFriendlyError(code) {
        const err = ERROR_MESSAGES[code] || ERROR_MESSAGES.DEFAULT;
        showError(`${err.title}. ${err.suggestion}`);
    }

    /* =====================================================
       Existing Functions (UNCHANGED)
    ===================================================== */

    function updateUI() {
        if (!textInput || !charCounter || !clearButton) return;
        const currentLength = textInput.value.length;
        charCounter.textContent = `${currentLength} / ${maxLength}`;
        clearButton.style.display = currentLength > 0 ? 'block' : 'none';
    }

    function setLoading(isLoading) {
        if (!submitBtn || !resultSection) return;
        submitBtn.disabled = isLoading;
        submitBtn.innerHTML = isLoading
            ? `<span>Processing...</span>`
            : `<span>Analyze Text</span>`;
    }

    function showResult(data) {
        if (!resultSection) return;
        const sentiment = data.sentiment;
        const confidence = (data.confidence * 100).toFixed(1);
        const isPositive = sentiment.toLowerCase() === "positive";
        const emoji = isPositive ? "🎉" : "😔";
        const msg = isPositive ? "Great vibes detected!" : "Negative sentiment detected.";

        resultSection.innerHTML = `
            <div>
                <div class="emoji-xl">${emoji}</div>
                <h3>${sentiment.toUpperCase()}</h3>
                <p>${msg}</p>
                <p>Confidence: ${confidence}%</p>
                <button onclick="window.resetResultSection()">Try Another</button>
            </div>
        `;
    }

    function showError(msg) {
        if (!resultSection) return;
        resultSection.innerHTML = `
            <div class="placeholder-content">
                <h3 style="color:#ef4444">Error</h3>
                <p>${msg}</p>
                <button onclick="window.resetResultSection()">Try Again</button>
            </div>
        `;
    }

    window.resetResultSection = function () {
        if (!resultSection) return;
        resultSection.innerHTML = `
            <div class="placeholder-content">
                <h3>Ready to Analyze</h3>
                <p>Your results will appear here.</p>
            </div>
        `;
    };

    /* -------- history, analytics, chart code unchanged -------- */
});

// Logger Setup
const logger = window.log || {
    debug: console.log,
    error: console.error,
    setLevel: () => {}
};

// Config for local dev
if (window.location.hostname === "localhost" && window.log) {
    logger.setLevel("debug");
}

// Session management
let sessionId = localStorage.getItem('session_id') || null;

document.addEventListener("DOMContentLoaded", function () {
    const textInput = document.getElementById("text");
    const charCounter = document.getElementById("charCounter");
    const clearButton = document.getElementById("clearButton");
    const submitBtn = document.getElementById("submitBtn");
    const resultSection = document.getElementById("resultSection");
    const exportBtn = document.getElementById("exportBtn");
    const clearHistoryBtn = document.getElementById("clearHistoryBtn");
    const maxLength = 500;
    let sentimentChart = null;

    // --- CHART LOGIC ---
    function initChart(data) {
        const ctx = document.getElementById('sentimentChart').getContext('2d');
        
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
                pointBackgroundColor: (context) => {
                    const index = context.dataIndex;
                    const value = context.dataset.data[index];
                    return value && value.y === 1 ? '#10b981' : '#ef4444';
                },
                pointRadius: 6,
                fill: true
            }]
        };

        const config = {
            type: 'line',
            data: chartData,
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        type: 'time',
                        time: {
                            tooltipFormat: 'yyyy-MM-dd HH:mm:ss',
                            displayFormats: {
                                second: 'HH:mm:ss',
                                minute: 'HH:mm',
                                hour: 'HH:mm'
                            }
                        },
                        title: {
                            display: true,
                            text: 'Time'
                        }
                    },
                    y: {
                        min: -0.5,
                        max: 1.5,
                        ticks: {
                            stepSize: 0.5,
                            callback: function(value) {
                                if (value === 1) return 'Positive';
                                if (value === 0) return 'Negative';
                                return '';
                            },
                            font: {
                                size: 13,
                                weight: '700'
                            },
                            color: '#1e293b'
                        },
                        title: {
                            display: true,
                            text: 'Sentiment',
                            font: {
                                weight: 'bold'
                            }
                        },
                        grid: {
                            display: true,
                            drawBorder: true,
                            color: 'rgba(0,0,0,0.1)'
                        }
                    }
                },
                plugins: {
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                const val = context.parsed.y;
                                const sentiment = val === 1 ? 'Positive' : 'Negative';
                                const item = data[context.dataIndex];
                                return `Sentiment: ${sentiment} | Text: ${item.text}`;
                            }
                        }
                    }
                }
            }
        };

        if (sentimentChart) {
            sentimentChart.destroy();
        }
        sentimentChart = new Chart(ctx, config);
    }

    async function fetchAnalytics() {
        try {
            const response = await fetch('/analytics');
            const data = await response.json();
            
            // Sync with LocalStorage
            if (data.trends) {
                localStorage.setItem('sentiment_history', JSON.stringify(data.trends));
            }

            if (data.trends && data.trends.length > 0) {
                initChart(data.trends);
            } else {
                if (sentimentChart) {
                    sentimentChart.destroy();
                    sentimentChart = null;
                }
            }
        } catch (error) {
            logger.error("Failed to fetch analytics:", error);
            // Fallback to LocalStorage if offline/error
            const localData = localStorage.getItem('sentiment_history');
            if (localData) {
                initChart(JSON.parse(localData));
            }
        }
    }

    // Export Logic
    if (exportBtn) {
        exportBtn.addEventListener("click", () => {
            window.location.href = '/export';
        });
    }

    // Clear History Logic
    if (clearHistoryBtn) {
        clearHistoryBtn.addEventListener("click", async () => {
            if (confirm("Are you sure you want to clear all history? This will delete data from both the server and your browser.")) {
                try {
                    const response = await fetch('/clear-history', { method: 'POST' });
                    if (response.ok) {
                        localStorage.removeItem('sentiment_history');
                        fetchAnalytics();
                    }
                } catch (error) {
                    logger.error("Failed to clear history:", error);
                }
            }
        });
    }

    // Initial Fetch
    const localData = localStorage.getItem('sentiment_history');
    if (localData) {
        try {
            const parsedData = JSON.parse(localData);
            if (parsedData.length > 0) {
                initChart(parsedData);
            }
        } catch (e) {
            logger.error("Error parsing local history:", e);
        }
    }
    fetchAnalytics();

    // 1. Character Counter & Clear Button Logic
    function updateUI() {
        const currentLength = textInput.value.length;
        charCounter.textContent = `${currentLength} / ${maxLength}`;
        
        // Toggle Clear Button
        if (currentLength > 0) {
            clearButton.style.display = 'block';
        } else {
            clearButton.style.display = 'none';
        }
    }

    // Clear Input Action
    if (clearButton) {
        clearButton.addEventListener("click", () => {
            textInput.value = "";
            textInput.focus();
            updateUI();
            resetResultSection();
        });
    }
  }

 
  updateCharCounter();

 
  textInput.addEventListener("input", updateCharCounter);

  // Initialize history sidebar
  initHistorySidebar();
  
  // Load history on page load
  loadHistory();
});

document
  .getElementById("sentimentForm")
  .addEventListener("submit", async function (event) {
    event.preventDefault(); // Prevent form from reloading the page


    const text = document.getElementById("text").value.trim();
    const responseElement = document.getElementById("response");


    // Input Listener
    if (textInput) {
        textInput.addEventListener("input", updateUI);
        
        // Keyboard Shortcuts
        textInput.addEventListener("keydown", function (event) {
            // Ctrl+Enter to submit
            if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
                event.preventDefault();
                document.getElementById("sentimentForm").dispatchEvent(new Event("submit"));
            }
            // Esc to clear
            if (event.key === "Escape") {
                event.preventDefault();
                clearButton.click();
            }
        });
    }


    showLoading(true);
    hideResponse();


    try {
      const url = `${baseUrl}/predict`;


      logger.debug("Sending request to:", url);
      logger.debug("Request payload:", { text });


      const headers = {
        "Content-Type": "application/json",
        Accept: "application/json",
      };
      
      // Add session ID to headers if available
      if (sessionId) {
        headers["X-Session-ID"] = sessionId;
      }

      const response = await fetch(url, {
        method: "POST",
        headers: headers,
        mode: "cors",
        credentials: "omit",
        body: JSON.stringify({ text }),
      });


      logger.debug("Response status:", response.status);
      logger.debug(
        "Response headers:",
        Object.fromEntries(response.headers.entries())
      );


      // Actually read the response text from the server
      const responseText = await response.text();


      logger.debug("Raw response:", responseText);


      let responseData;
      try {
        responseData = JSON.parse(responseText);
        logger.debug("Parsed response data:", responseData);
      } catch (e) {
        logger.error("Failed to parse JSON response", e);
        logger.error("Invalid response body:", responseText);
        throw new Error("Invalid response from server");
      }


      if (!response.ok) {
        throw new Error(
          responseData.error || `Server error: ${response.status}`
        );
      }


      if (!responseData.sentiment || !responseData.confidence) {
        logger.error("Unexpected response format:", responseData);
        throw new Error("Invalid response format from server");
      }

      // Store session ID if provided
      if (responseData.session_id) {
        sessionId = responseData.session_id;
        localStorage.setItem('session_id', sessionId);
      }

      // Reload history after new prediction
      loadHistory();

      const emoji =
        responseData.sentiment === "Positive" ? "😊" : "😔";


      const resultHTML = `
        <div class="emoji">${emoji}</div>
        <div class="result-content">
          <h3>Analysis Result</h3>
          <p>Sentiment: <strong>${responseData.sentiment}</strong></p>
          <p>Confidence: <strong>${
            (responseData.confidence * 100).toFixed(1)
          }%</strong></p>
        </div>
      `;


      showResponse(
        resultHTML,
        false,
        responseData.sentiment === "Positive"
      );
    } catch (error) {
      logger.error("Sentiment analysis failed:", error);
      showResponse(`Error: ${error.message}`, true);
    } finally {
      showLoading(false);
    // 2. Form Submission Handler
    document.getElementById("sentimentForm").addEventListener("submit", async function (event) {
        event.preventDefault();

        const text = textInput.value.trim();
        
        if (!text) {
            alert("Please enter some text first!");
            return;
        }

        // --- STATE: LOADING ---
        setLoading(true);

        try {
            const url = `/predict`; 
            const timestamp = new Date().toISOString().slice(0, 19).replace('T', ' '); // Real-time local timestamp
            
            logger.debug("Sending request to:", url);

            const response = await fetch(url, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ text, timestamp }),
            });

            const data = await response.json();
            logger.debug("Response data:", data);

            if (!response.ok) throw new Error(data.error || "Server Error");

            // --- STATE: SUCCESS ---
            showResult(data);
            
            // Immediately update local storage and chart for real-time feel
            const newRecord = {
                timestamp: timestamp,
                sentiment: data.sentiment.toLowerCase(),
                text: text.length > 50 ? text.slice(0, 50) + '...' : text
            };
            
            let history = JSON.parse(localStorage.getItem('sentiment_history') || '[]');
            history.push(newRecord);
            localStorage.setItem('sentiment_history', JSON.stringify(history));
            
            initChart(history);
            
            // Still sync with server to ensure consistency
            fetchAnalytics();

        } catch (error) {
            logger.error("Analysis failed:", error);
            showError(error.message);
        } finally {
            setLoading(false);
        }
    });

    // --- HELPER FUNCTIONS ---

    function setLoading(isLoading) {
        if (isLoading) {
            submitBtn.disabled = true;
            submitBtn.innerHTML = `<span>Processing AI...</span>`;
            
            // Show Pulsing Animation in Right Panel
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
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 11.5a8.38 8.38 0 0 1-.9 3.8 8.5 8.5 0 0 1-7.6 4.7 8.38 8.38 0 0 1-3.8-.9L3 21l1.9-5.7a8.38 8.38 0 0 1-.9-3.8 8.5 8.5 0 0 1 4.7-7.6 8.38 8.38 0 0 1 3.8-.9h.5a8.48 8.48 0 0 1 8 8v.5z"/></svg>
            `;
        }
    }

    // >>> UPDATED FUNCTION WITH ANIMATION LOGIC <<<
    function showResult(data) {
        const sentiment = data.sentiment;
        const confidence = (data.confidence * 100).toFixed(1);
        const isPositive = sentiment.toLowerCase() === "positive";

        // 1. Define Visuals
        const color = isPositive ? "var(--success)" : "var(--danger)";
        const emoji = isPositive ? "🎉" : "😔";
        
        // 2. Define Animation Class (This was missing!)
        const animationClass = isPositive ? "anim-positive" : "anim-negative";

        const msg = isPositive 
            ? "Great vibes detected! The sentiment is glowing." 
            : "Negative sentiment detected. Sounds a bit serious.";

        // 3. Inject HTML
        resultSection.innerHTML = `
            <div style="animation: fadeInUp 0.5s ease forwards;">
                
                <div class="emoji-xl ${animationClass}">${emoji}</div>
                
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
                    <p style="margin-top:1rem; font-size:0.95rem; color:#64748b; line-height:1.5;">${msg}</p>
                </div>

                <button onclick="speakResult('${sentiment}', ${confidence})" class="listen-btn">
                    🔊 Listen to Result
                </button>
            </div>
        `;

        // Trigger Progress Bar Animation
        setTimeout(() => {
            const bar = resultSection.querySelector(".progress-fill");
            if (bar) bar.style.width = `${confidence}%`;
        }, 100);
    }

    function showError(msg) {
        resultSection.innerHTML = `
            <div class="placeholder-content">
                <div class="pulse-circle" style="background: #fee2e2; color: #ef4444; animation: none;">⚠️</div>
                <h3 style="color: #ef4444">Analysis Failed</h3>
                <p>${msg}</p>
                <button onclick="resetResultSection()" style="margin-top:1rem; padding:0.5rem 1rem; border:1px solid #e5e7eb; background:white; border-radius:8px; cursor:pointer;">Try Again</button>
            </div>
        `;
    }

    // Reset to "Ready to Analyze" state
    window.resetResultSection = function() {
        resultSection.innerHTML = `
            <div class="placeholder-content">
                <div class="pulse-circle">🔍</div>
                <h3>Ready to Analyze</h3>
                <p>Your results will appear here with detailed confidence scoring.</p>
            </div>
        `;
    };
});

  loadingElement.style.display = show ? "block" : "none";
}

/* ---------- HISTORY MANAGEMENT ---------- */

function initHistorySidebar() {
  const toggleButton = document.getElementById('toggleHistory');
  const historyContent = document.getElementById('historyContent');
  const toggleIcon = document.getElementById('toggleIcon');
  
  if (toggleButton && historyContent) {
    toggleButton.addEventListener('click', function() {
      const isCollapsed = historyContent.style.display === 'none';
      historyContent.style.display = isCollapsed ? 'block' : 'none';
      toggleIcon.textContent = isCollapsed ? '▼' : '▲';
    });
  }
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
    
    const headers = {
      "Accept": "application/json",
    };
    
    if (sessionId) {
      headers["X-Session-ID"] = sessionId;
    }
    
    const response = await fetch(`${baseUrl}/history`, {
      method: "GET",
      headers: headers,
      mode: "cors",
      credentials: "omit",
    });
    
    if (!response.ok) {
      throw new Error(`Failed to load history: ${response.status}`);
    }
    
    const data = await response.json();
    
    // Store session ID if provided
    if (data.session_id) {
      sessionId = data.session_id;
      localStorage.setItem('session_id', sessionId);
    }
    
    historyLoading.style.display = 'none';
    
    if (!data.predictions || data.predictions.length === 0) {
      historyEmpty.style.display = 'block';
      return;
    }
    
    // Display predictions
    data.predictions.forEach(prediction => {
      const card = createHistoryCard(prediction);
      historyList.appendChild(card);
    });
    
  } catch (error) {
    logger.error("Failed to load history:", error);
    historyLoading.style.display = 'none';
    historyEmpty.style.display = 'block';
    historyEmpty.innerHTML = '<p>Failed to load history.</p>';
  }
}

function createHistoryCard(prediction) {
  const card = document.createElement('div');
  card.className = 'history-card';
  card.dataset.predictionId = prediction.id;
  
  const inputData = typeof prediction.input_data === 'string' 
    ? JSON.parse(prediction.input_data) 
    : prediction.input_data;
  const result = typeof prediction.prediction_result === 'string'
    ? JSON.parse(prediction.prediction_result)
    : prediction.prediction_result;
  
  const text = inputData.text || '';
  const sentiment = result.sentiment || 'Unknown';
  const confidence = result.confidence || 0;
  const isFavorite = prediction.is_favorite || false;
  
  // Format timestamp
  const timestamp = new Date(prediction.timestamp);
  const dateStr = timestamp.toLocaleDateString();
  const timeStr = timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  
  const emoji = sentiment === "Positive" ? "😊" : "😔";
  const sentimentClass = sentiment === "Positive" ? "positive" : "negative";
  
  card.innerHTML = `
    <div class="history-card-header">
      <div class="history-card-sentiment ${sentimentClass}">
        <span class="history-emoji">${emoji}</span>
        <span class="history-sentiment-text">${sentiment}</span>
      </div>
      <button class="favorite-btn ${isFavorite ? 'favorited' : ''}" 
              data-prediction-id="${prediction.id}" 
              aria-label="${isFavorite ? 'Remove from favorites' : 'Add to favorites'}">
        ${isFavorite ? '⭐' : '☆'}
      </button>
    </div>
    <div class="history-card-text">${truncateText(text, 100)}</div>
    <div class="history-card-footer">
      <span class="history-confidence">Confidence: ${(confidence * 100).toFixed(1)}%</span>
      <span class="history-timestamp">${dateStr} ${timeStr}</span>
    </div>
  `;
  
  // Add favorite button event listener
  const favoriteBtn = card.querySelector('.favorite-btn');
  if (favoriteBtn) {
    favoriteBtn.addEventListener('click', async function(e) {
      e.stopPropagation();
      await toggleFavorite(prediction.id, favoriteBtn);
    });
  }
  
  return card;
}

function truncateText(text, maxLength) {
  if (text.length <= maxLength) return text;
  return text.substring(0, maxLength) + '...';
}

async function toggleFavorite(predictionId, buttonElement) {
  try {
    const headers = {
      "Content-Type": "application/json",
      "Accept": "application/json",
    };
    
    if (sessionId) {
      headers["X-Session-ID"] = sessionId;
    }
    
    const response = await fetch(`${baseUrl}/favorite/${predictionId}`, {
      method: "POST",
      headers: headers,
      mode: "cors",
      credentials: "omit",
    });
    
    if (!response.ok) {
      throw new Error(`Failed to toggle favorite: ${response.status}`);
    }
    
    const data = await response.json();
    
    // Update button state
    if (buttonElement) {
      if (data.is_favorite) {
        buttonElement.classList.add('favorited');
        buttonElement.textContent = '⭐';
        buttonElement.setAttribute('aria-label', 'Remove from favorites');
      } else {
        buttonElement.classList.remove('favorited');
        buttonElement.textContent = '☆';
        buttonElement.setAttribute('aria-label', 'Add to favorites');
      }
    }
    
    logger.debug("Favorite toggled:", data);
    
  } catch (error) {
    logger.error("Failed to toggle favorite:", error);
    alert('Failed to update favorite status. Please try again.');
  }
}
// 3. Global Text-to-Speech Function
window.speakResult = function(sentiment, confidence) {
    if ('speechSynthesis' in window) {
        window.speechSynthesis.cancel();

        const text = `The sentiment is ${sentiment}. I am ${confidence} percent sure.`;
        const utterance = new SpeechSynthesisUtterance(text);
        
        const voices = window.speechSynthesis.getVoices();
        const preferredVoice = voices.find(v => v.name.includes("Google") && v.lang.includes("en")) || voices[0];
        if (preferredVoice) utterance.voice = preferredVoice;

        utterance.rate = 1; 
        utterance.pitch = 1; 
        
        window.speechSynthesis.speak(utterance);
    } else {
        alert("Sorry, your browser doesn't support text-to-speech.");
    }
};

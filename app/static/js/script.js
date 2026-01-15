const logger = window.log;

logger.setLevel(
  window.location.hostname === "localhost" ? "debug" : "warn"
);

// Session management
let sessionId = localStorage.getItem('session_id') || null;

document.addEventListener("DOMContentLoaded", function () {
  const textInput = document.getElementById("text");
  const charCounter = document.getElementById("charCounter");
  const maxLength = 500;

  if (!textInput || !charCounter) return;

  function updateCharCounter() {
    const currentLength = textInput.value.length;
    const remaining = maxLength - currentLength;

    charCounter.textContent = `${currentLength} / ${maxLength} characters (${remaining} remaining)`;

    charCounter.classList.remove("warning", "error");

    if (remaining <= 50 && remaining > 0) {
      charCounter.classList.add("warning");
    }

    if (remaining <= 0) {
      charCounter.classList.add("error");
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


    if (!text) {
      showResponse("Error: Please enter some text!", true);
      return;
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
    }
  });


/* ---------- UI HELPERS ---------- */
function showResponse(html, isError, isPositive = null) {
  const responseElement = document.getElementById("response");
  responseElement.innerHTML = html;
  responseElement.className = "fade-in";


  if (isError) {
    responseElement.classList.add("error");
  } else if (isPositive !== null) {
    responseElement.classList.add(isPositive ? "success" : "error");
  }


  responseElement.style.display = "block";
  responseElement.classList.add("show");
}


function hideResponse() {
  const responseElement = document.getElementById("response");
  responseElement.style.display = "none";
  responseElement.className = "";
}


function showLoading(show) {
  let loadingElement = document.querySelector(".loading");


  if (!loadingElement) {
    loadingElement = document.createElement("div");
    loadingElement.className = "loading";
    loadingElement.innerHTML = `
      <div class="spinner"></div>
      <p>Analyzing sentiment...</p>
    `;
    document.getElementById("sentimentForm").after(loadingElement);
  }


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

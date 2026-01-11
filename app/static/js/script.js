document.addEventListener("DOMContentLoaded", function() {
    const textInput = document.getElementById("text");
    const charCount = document.getElementById("charCount");
    const charCounter = document.querySelector(".character-counter");
    const maxLength = 1000;

    // Update character count
    function updateCharCount() {
        const length = textInput.value.length;
        charCount.textContent = length;
        
        // Update counter styling
        charCounter.classList.remove("warning", "error");
        if (length > maxLength * 0.9) {
            charCounter.classList.add("warning");
        }
        if (length > maxLength) {
            charCounter.classList.add("error");
        }
    }

    // Add input event listener for character counting
    textInput.addEventListener("input", updateCharCount);
    
    // Initialize character count
    updateCharCount();
});

document.getElementById("sentimentForm").addEventListener("submit", async function(event) {
    event.preventDefault();  // Prevent form from reloading the page

    const text = document.getElementById("text").value.trim();
    const responseElement = document.getElementById("response");

    if (!text) {
        showResponse("Error: Please enter some text!", true);
        return;
    }

    // Check character limit
    if (text.length > 1000) {
        showResponse("Error: Text exceeds the maximum limit of 1000 characters. Please shorten your text.", true);
        return;
    }

    // Show loading animation
    showLoading(true);
    hideResponse();

    try {
        const url = `${baseUrl}/predict`;
        console.log('Sending request to:', url);
        console.log('Request data:', { text: text });

        // Use baseUrl for API calls
        const response = await fetch(url, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                "Accept": "application/json"
            },
            mode: "cors",
            credentials: "omit",
            body: JSON.stringify({ text: text })
        });

        console.log('Response status:', response.status);
        console.log('Response headers:', Object.fromEntries(response.headers.entries()));
        
        // Get the response text first
        const responseText = await response.text();
        console.log('Raw response:', responseText);
        
        let responseData;
        try {
            responseData = JSON.parse(responseText);
            console.log('Parsed response data:', responseData);
        } catch (e) {
            console.error('Failed to parse JSON:', e);
            console.error('Raw response that failed to parse:', responseText);
            throw new Error('Invalid response from server');
        }

        if (!response.ok) {
            throw new Error(responseData.error || `Server error: ${response.status}`);
        }
        
        if (!responseData.sentiment || !responseData.confidence) {
            console.error('Invalid response format:', responseData);
            throw new Error('Invalid response format from server');
        }
        
        // Create result HTML with emoji
        const emoji = responseData.sentiment === 'Positive' ? '😊' : '😔';
        const resultHTML = `
            <div class="emoji">${emoji}</div>
            <div class="result-content">
                <h3>Analysis Result</h3>
                <p>Sentiment: <strong>${responseData.sentiment}</strong></p>
                <p>Confidence: <strong>${(responseData.confidence * 100).toFixed(1)}%</strong></p>
            </div>
        `;
        
        showResponse(resultHTML, false, responseData.sentiment === 'Positive');
    } catch (error) {
        console.error('Error:', error);
        showResponse(`Error: ${error.message}`, true);
    } finally {
        showLoading(false);
    }
});

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
    const loadingElement = document.querySelector(".loading");
    if (!loadingElement) {
        // Create loading element if it doesn't exist
        const loading = document.createElement("div");
        loading.className = "loading";
        loading.innerHTML = `
            <div class="spinner"></div>
            <p>Analyzing sentiment...</p>
        `;
        document.getElementById("sentimentForm").after(loading);
    }
    document.querySelector(".loading").style.display = show ? "block" : "none";
}

// old one 
// fetch("/predict", {
//     fetch("http://127.0.0.1:5000/predict", {
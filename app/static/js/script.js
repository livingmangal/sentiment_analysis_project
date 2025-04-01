document.getElementById("sentimentForm").addEventListener("submit", async function(event) {
    event.preventDefault();  // Prevent form from reloading the page

    const text = document.getElementById("text").value.trim();
    const responseElement = document.getElementById("response");

    if (!text) {
        showResponse("Error: Please enter some text!", true);
        return;
    }

    // Show loading animation
    showLoading(true);
    hideResponse();

    try {
        const response = await fetch("/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ text: text })
        });

        if (!response.ok) {
            throw new Error(`Server error: ${response.status}`);
        }

        const data = await response.json();
        
        // Create result HTML with emoji
        const emoji = data.sentiment === 'Positive' ? '😊' : '😔';
        const resultHTML = `
            <div class="emoji">${emoji}</div>
            <div class="result-content">
                <h3>Analysis Result</h3>
                <p>Sentiment: <strong>${data.sentiment}</strong></p>
                <p>Confidence: <strong>${(data.confidence * 100).toFixed(1)}%</strong></p>
            </div>
        `;
        
        showResponse(resultHTML, false, data.sentiment === 'Positive');
    } catch (error) {
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
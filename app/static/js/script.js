// Wait for DOM to be fully loaded
document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById("sentimentForm");
    const textInput = document.getElementById("text");
    const responseElement = document.getElementById("response");
    const loadingSpinner = document.getElementById("loading-spinner");

    // Check if elements exist
    if (!form || !textInput || !responseElement || !loadingSpinner) {
        console.error("Required elements not found!");
        return;
    }

    form.addEventListener("submit", async function(event) {
        event.preventDefault();  // Prevent form from submitting normally
        console.log("Form submitted");

        const text = textInput.value.trim();
        console.log("Input text:", text);

        if (!text) {
            showError("Please enter some text to analyze");
            return;
        }

        // Show loading spinner and hide previous results
        loadingSpinner.style.display = "block";
        responseElement.style.display = "none";

        try {
            console.log("Sending request to server...");
            // Make sure we're sending to the /predict endpoint
            const response = await fetch("/predict", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                    "Accept": "application/json"
                },
                body: JSON.stringify({ text: text })
            });

            console.log("Response status:", response.status);
            
            if (!response.ok) {
                const errorData = await response.json().catch(() => ({}));
                throw new Error(errorData.error || `Server error: ${response.status}`);
            }

            const data = await response.json();
            console.log("Server response:", data);
            
            if (!data.sentiment || typeof data.confidence === 'undefined') {
                throw new Error("Invalid response format from server");
            }

            // Create result HTML
            const resultHTML = `
                <h2>Analysis Result</h2>
                <div class="sentiment-display">
                    <div class="sentiment-icon">
                        <span>${data.sentiment === 'Positive' ? '😊' : '😔'}</span>
                    </div>
                    <div class="sentiment-details">
                        <p>Sentiment: <strong>${data.sentiment}</strong></p>
                        <p>Confidence: <strong>${(data.confidence * 100).toFixed(1)}%</strong></p>
                    </div>
                </div>
            `;
            
            responseElement.innerHTML = resultHTML;
            responseElement.style.display = "block";
            responseElement.style.borderLeft = `4px solid ${data.sentiment === 'Positive' ? 'var(--success-color)' : 'var(--danger-color)'}`;
        } catch (error) {
            console.error("Error:", error);
            showError(error.message);
        } finally {
            loadingSpinner.style.display = "none";
        }
    });
});

function showError(message) {
    console.log("Showing error:", message);
    const responseElement = document.getElementById("response");
    if (!responseElement) {
        console.error("Response element not found!");
        return;
    }

    responseElement.innerHTML = `
        <h2 style="color: var(--danger-color)">Error</h2>
        <p>${message}</p>
    `;
    responseElement.style.display = "block";
    responseElement.style.borderLeft = "4px solid var(--danger-color)";
    
    const loadingSpinner = document.getElementById("loading-spinner");
    if (loadingSpinner) {
        loadingSpinner.style.display = "none";
    }
}

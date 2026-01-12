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
const form = document.getElementById("sentimentForm");
const textInput = document.getElementById("text");
const submitButton = form.querySelector("button[type='submit']");
const responseElement = document.getElementById("response");

/* ---------- REAL-TIME VALIDATION ---------- */
function isValidInput(value) {
	return value.trim().length > 0;
}

function updateUIState() {
	const valid = isValidInput(textInput.value);

	submitButton.disabled = !valid;

	if (!valid && textInput.value.length > 0) {
		showResponse("Please enter meaningful text (not just spaces).", true);
	} else {
		hideResponse();
	}
}

textInput.addEventListener("input", updateUIState);

/* ---------- FORM SUBMISSION ---------- */
form.addEventListener("submit", async function (event) {
	event.preventDefault();

	const text = textInput.value.trim();

	if (!isValidInput(text)) {
		showResponse("Error: Input cannot be empty or whitespace only.", true);
		return;
	}

	showLoading(true);
	hideResponse();

	try {
		const response = await fetch(`${baseUrl}/predict`, {
			method: "POST",
			headers: {
				"Content-Type": "application/json",
				Accept: "application/json",
			},
			body: JSON.stringify({ text }),
		});

		const data = await response.json();

		if (!response.ok) {
			throw new Error(data.error || "Server error");
		}

		const emoji = data.sentiment === "Positive" ? "😊" : "😔";

		showResponse(
			`
            <div class="emoji">${emoji}</div>
            <div class="result-content">
                <h3>Analysis Result</h3>
                <p>Sentiment: <strong>${data.sentiment}</strong></p>
                <p>Confidence: <strong>${(data.confidence * 100).toFixed(1)}%</strong></p>
            </div>
            `,
			false,
			data.sentiment === "Positive"
		);
	} catch (error) {
		showResponse(`Error: ${error.message}`, true);
	} finally {
		showLoading(false);
	}
});

/* ---------- UI HELPERS ---------- */
function showResponse(html, isError, isPositive = null) {
	responseElement.innerHTML = html;
	responseElement.className = "fade-in show";

	if (isError) {
		responseElement.classList.add("error");
	} else if (isPositive !== null) {
		responseElement.classList.add(isPositive ? "success" : "error");
	}

	responseElement.style.display = "block";
}

function hideResponse() {
	responseElement.style.display = "none";
	responseElement.className = "";
}

function showLoading(show) {
    // --- 1. Existing Spinner Logic ---
    let loading = document.querySelector(".loading");

    if (!loading) {
        loading = document.createElement("div");
        loading.className = "loading";
        loading.innerHTML = `
            <div class="spinner"></div>
            <p>Analyzing sentiment...</p>
        `;
        form.after(loading);
    }

    loading.style.display = show ? "block" : "none";

    // --- 2. New Button Loading State Logic ---
    
    // Disable the button based on the 'show' flag
    submitButton.disabled = show;

    // Select the text and icon spans inside the button
    const btnText = submitButton.querySelector(".button-text");
    const btnIcon = submitButton.querySelector(".button-icon");

    if (show) {
        // Change text and icon to indicate loading
        if (btnText) btnText.textContent = "Analyzing...";
        if (btnIcon) btnIcon.textContent = "⏳";
    } else {
        // Restore original text and icon
        if (btnText) btnText.textContent = "Analyze Sentiment";
        if (btnIcon) btnIcon.textContent = "🔍";
        
        // BUG FIX: Do NOT call updateUIState() here.
        // Instead, manually check validity so we don't accidentally hide the result.
        submitButton.disabled = !isValidInput(textInput.value);
    }
}
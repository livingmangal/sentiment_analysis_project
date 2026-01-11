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
}
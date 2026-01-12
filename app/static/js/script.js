const logger = window.log;

logger.setLevel(
  window.location.hostname === "localhost" ? "debug" : "warn"
);

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


      const response = await fetch(url, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Accept: "application/json",
        },
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

document.addEventListener("DOMContentLoaded", () => {
  // Prompt + interpret
  const promptInput = document.getElementById("prompt-input");
  const interpretBtn = document.getElementById("interpret-btn");
  const resultsContainer = document.getElementById("results-container");

  // Patch target toggle
  const patchToggle = document.getElementById("patch-target-toggle");
  const patchToggleVal = document.getElementById("patch-target-value");
  patchToggleVal.textContent = patchToggle.checked ? "Same Layer" : "0";
  patchToggle.addEventListener("change", () => {
    patchToggleVal.textContent = patchToggle.checked ? "Same Layer" : "0";
  });

  // The snippet-based slider logic
  const rangeMin = document.querySelector(".range-min");
  const rangeMax = document.querySelector(".range-max");
  const progress = document.querySelector(".slider .progress");
  const startValSpan = document.querySelector(".start-value");
  const endValSpan = document.querySelector(".end-value");

  function updateSlider(e) {
    let minVal = parseInt(rangeMin.value, 10);
    let maxVal = parseInt(rangeMax.value, 10);

    if (maxVal < minVal) {
      if (e.target.classList.contains("range-min")) {
        rangeMin.value = maxVal;
        minVal = maxVal;
      } else {
        rangeMax.value = minVal;
        maxVal = minVal;
      }
    }

    // Update the progress bar
    progress.style.left = (minVal / 39) * 100 + "%";
    progress.style.right = (100 - (maxVal / 39) * 100) + "%";

    // Update numeric text
    startValSpan.textContent = minVal;
    endValSpan.textContent = maxVal;
  }

  [rangeMin, rangeMax].forEach(input => {
    input.addEventListener("input", updateSlider);
  });
  updateSlider({ target: rangeMin });

  //--- Mode Toggle Logic (onlyBest / showAll)
  let interpretationsOnlyBest = [];
  let interpretationsAll = [];
  let currentMode = "onlyBest";

  const modeToggleContainer = document.getElementById("mode-toggle-container");
  const modeToggle = document.getElementById("mode-toggle");
  const modeValueSpan = document.getElementById("mode-value");

  modeToggleContainer.style.display = "none"; // hide until data is loaded

  modeToggle.addEventListener("change", () => {
    if (modeToggle.checked) {
      currentMode = "showAll";
      modeValueSpan.textContent = "Show All";
      displayInterpretations(interpretationsAll);
    } else {
      currentMode = "onlyBest";
      modeValueSpan.textContent = "Show Only Best";
      displayInterpretations(interpretationsOnlyBest);
    }
  });

  // Interpret => fetch from /api/interpret
  interpretBtn.addEventListener("click", async () => {
    const prompt = promptInput.value.trim();
    if (!prompt) {
      alert("Please enter a prompt.");
      return;
    }
    const minVal = parseInt(rangeMin.value, 10);
    const maxVal = parseInt(rangeMax.value, 10);
    const patchTarget = patchToggle.checked ? "Same Layer" : "0";

    resultsContainer.innerHTML = "<p>Loading...</p>";

    try {
      const resp = await fetch("/api/interpret", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          prompt,
          layerStart: minVal,
          layerEnd: maxVal,
          patchTarget
        })
      });
      if (!resp.ok) {
        throw new Error("Server responded with an error!");
      }
      const data = await resp.json();

      // data.onlyBest => array, data.showAll => array
      interpretationsOnlyBest = data.onlyBest || [];
      interpretationsAll = data.showAll || [];

      // Default to onlyBest
      modeToggle.checked = false;
      currentMode = "onlyBest";
      modeValueSpan.textContent = "Show Only Best";

      // Show the mode toggle container
      modeToggleContainer.style.display = "flex";

      // Render onlyBest by default
      displayInterpretations(interpretationsOnlyBest);

    } catch (err) {
      resultsContainer.innerHTML = `<p class="error">Error: ${err.message}</p>`;
    }
  });

  function displayInterpretations(interpretations) {
    resultsContainer.innerHTML = "";

    if (!interpretations || interpretations.length === 0) {
      resultsContainer.innerHTML = "<p>No tokens to display.</p>";
      return;
    }

    const tokensLine = document.createElement("div");
    tokensLine.className = "tokens-line";
    const detailsContainer = document.createElement("div");

    interpretations.forEach((tokenInfo) => {
      const tokenBadge = document.createElement("span");
      tokenBadge.className = "token-badge";
      tokenBadge.textContent = tokenInfo.token;
      tokensLine.appendChild(tokenBadge);

      const tokenDetails = document.createElement("div");
      tokenDetails.className = "token-details";

      const heading = document.createElement("h4");
      heading.textContent = `Interpretations for token: "${tokenInfo.token}"`;
      tokenDetails.appendChild(heading);

      // We'll have 4 columns in the order:
      // 1) Layer
      // 2) Residual Pre MLP
      // 3) MLP Output
      // 4) Hidden State
      const table = document.createElement("table");
      table.className = "layer-table";

      const thead = document.createElement("thead");
      const headerRow = document.createElement("tr");

      const layerTh = document.createElement("th");
      layerTh.textContent = "Layer";

      const residualTh = document.createElement("th");
      residualTh.textContent = "Residual Pre MLP";

      const mlpTh = document.createElement("th");
      mlpTh.textContent = "MLP Output";

      const hiddenTh = document.createElement("th");
      hiddenTh.textContent = "Hidden State";

      headerRow.appendChild(layerTh);
      headerRow.appendChild(residualTh);
      headerRow.appendChild(mlpTh);
      headerRow.appendChild(hiddenTh);

      thead.appendChild(headerRow);
      table.appendChild(thead);

      const tbody = document.createElement("tbody");
      (tokenInfo.layers || []).forEach((layerObj) => {
        const row = document.createElement("tr");

        // Column 1: Layer
        const layerTd = document.createElement("td");
        layerTd.textContent = layerObj.layer_name;

        // Column 2: Residual + Amp
        const residualTd = document.createElement("td");
        const rText = layerObj.residual_pre_mlp_interpretation;
        const rAmp = layerObj.residual_pre_mlp_amp || 0;
        residualTd.textContent = `${rText} (Amp=${rAmp})`;

        // Column 3: MLP + Amp
        const mlpTd = document.createElement("td");
        const mText = layerObj.mlp_output_interpretation;
        const mAmp = layerObj.mlp_output_amp || 0;
        mlpTd.textContent = `${mText} (Amp=${mAmp})`;

        // Column 4: Hidden + Amp (last)
        const hiddenTd = document.createElement("td");
        const hText = layerObj.hidden_state_interpretation;
        const hAmp = layerObj.hidden_state_amp || 0;
        hiddenTd.textContent = `${hText} (Amp=${hAmp})`;

        row.appendChild(layerTd);
        row.appendChild(residualTd);
        row.appendChild(mlpTd);
        row.appendChild(hiddenTd);
        tbody.appendChild(row);
      });
      table.appendChild(tbody);

      tokenDetails.appendChild(table);
      detailsContainer.appendChild(tokenDetails);

      // Toggle on click
      tokenBadge.addEventListener("click", () => {
        tokenDetails.classList.toggle("show");
      });
    });

    resultsContainer.appendChild(tokensLine);
    resultsContainer.appendChild(detailsContainer);
  }
});

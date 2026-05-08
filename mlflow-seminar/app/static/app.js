const fileInput = document.querySelector("#fileInput");
const originalImage = document.querySelector("#originalImage");
const overlayImage = document.querySelector("#overlayImage");
const maskImage = document.querySelector("#maskImage");
const confidenceValue = document.querySelector("#confidenceValue");
const histogram = document.querySelector("#histogram");

function renderHistogram(values) {
  const rows = Object.entries(values)
    .sort((a, b) => b[1].share - a[1].share)
    .slice(0, 8)
    .map(([name, data]) => {
      const percent = Math.round(data.share * 1000) / 10;
      return `
        <div class="hist-row">
          <span>${name}</span>
          <div class="bar"><i style="width: ${percent}%"></i></div>
          <strong>${percent}%</strong>
        </div>
      `;
    });
  histogram.innerHTML = rows.join("");
}

fileInput.addEventListener("change", async () => {
  const file = fileInput.files[0];
  if (!file) return;

  originalImage.src = URL.createObjectURL(file);
  overlayImage.classList.add("loading");
  maskImage.classList.add("loading");
  confidenceValue.textContent = "processing";
  histogram.innerHTML = "";

  const form = new FormData();
  form.append("file", file);

  const response = await fetch("/api/predict", {
    method: "POST",
    body: form,
  });

  if (!response.ok) {
    confidenceValue.textContent = "failed";
    overlayImage.classList.remove("loading");
    maskImage.classList.remove("loading");
    throw new Error(await response.text());
  }

  const result = await response.json();
  overlayImage.src = result.overlay;
  maskImage.src = result.mask;
  overlayImage.classList.remove("loading");
  maskImage.classList.remove("loading");
  confidenceValue.textContent = result.mean_confidence;
  renderHistogram(result.class_histogram);
});

(() => {
  const colors = { text: "#66777e", grid: "#e8eef0" };

  function drawChart(canvas) {
    let config;
    try {
      config = JSON.parse(canvas.dataset.chart);
    } catch (error) {
      console.error("Could not read chart data.", error);
      return;
    }
    const context = canvas.getContext("2d");
    if (!context || !config.labels || !config.datasets) return;
    const ratio = window.devicePixelRatio || 1;
    const width = canvas.clientWidth;
    const height = canvas.clientHeight;
    if (!width || !height) return;
    canvas.width = Math.floor(width * ratio);
    canvas.height = Math.floor(height * ratio);
    context.scale(ratio, ratio);
    context.clearRect(0, 0, width, height);

    const left = 42;
    const right = 12;
    const top = 14;
    const bottom = 48;
    const plotWidth = width - left - right;
    const plotHeight = height - top - bottom;
    const allValues = config.datasets.flatMap((dataset) => dataset.data);
    const maxValue = Math.max(1, ...allValues);
    const tickCount = 4;
    context.font = "11px Segoe UI, Arial, sans-serif";
    context.textBaseline = "middle";

    for (let tick = 0; tick <= tickCount; tick += 1) {
      const value = (maxValue * tick) / tickCount;
      const y = top + plotHeight - (plotHeight * tick) / tickCount;
      context.strokeStyle = colors.grid;
      context.beginPath();
      context.moveTo(left, y);
      context.lineTo(width - right, y);
      context.stroke();
      context.fillStyle = colors.text;
      context.textAlign = "right";
      context.fillText(Number.isInteger(value) ? String(value) : value.toFixed(1), left - 8, y);
    }

    const groupWidth = plotWidth / config.labels.length;
    const seriesWidth = Math.min(24, (groupWidth * 0.68) / config.datasets.length);
    config.datasets.forEach((dataset, seriesIndex) => {
      dataset.data.forEach((value, index) => {
        const barHeight = (value / maxValue) * plotHeight;
        const offset = (seriesIndex - (config.datasets.length - 1) / 2) * seriesWidth;
        const x = left + groupWidth * (index + 0.5) + offset - seriesWidth / 2;
        const y = top + plotHeight - barHeight;
        context.fillStyle = dataset.color || "#176b87";
        context.beginPath();
        context.roundRect(x, y, Math.max(3, seriesWidth - 3), barHeight, 3);
        context.fill();
      });
    });

    config.labels.forEach((label, index) => {
      const x = left + groupWidth * (index + 0.5);
      context.fillStyle = colors.text;
      context.textAlign = "center";
      context.textBaseline = "top";
      context.fillText(label.length > 14 ? `${label.slice(0, 12)}…` : label, x, top + plotHeight + 9);
    });

    let legendX = left;
    const legendY = height - 12;
    config.datasets.forEach((dataset) => {
      context.fillStyle = dataset.color || "#176b87";
      context.fillRect(legendX, legendY - 7, 9, 9);
      context.fillStyle = colors.text;
      context.textAlign = "left";
      context.textBaseline = "middle";
      context.fillText(dataset.label, legendX + 14, legendY - 2);
      legendX += Math.min(175, dataset.label.length * 7 + 30);
    });
  }

  function renderAll() {
    document.querySelectorAll("canvas[data-chart]").forEach(drawChart);
  }

  window.addEventListener("DOMContentLoaded", renderAll);
  window.addEventListener("resize", renderAll);
})();

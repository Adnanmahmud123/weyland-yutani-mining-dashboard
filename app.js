
const $ = id => document.getElementById(id);

let rows = [];
let series = [];
let chart = null;
let findings = [];
let stats = [];
let sourceLabel = "No data";

const fmt = n =>
  Number.isFinite(n)
    ? n.toLocaleString(undefined, { maximumFractionDigits: 2 })
    : "—";

const mean = a =>
  a.length ? a.reduce((s, x) => s + x, 0) / a.length : NaN;

const sd = a => {
  if (a.length < 2) return 0;
  const m = mean(a);
  return Math.sqrt(
    a.reduce((s, x) => s + (x - m) ** 2, 0) / (a.length - 1)
  );
};

const quantile = (a, q) => {
  if (!a.length) return NaN;
  const sorted = [...a].sort((x, y) => x - y);
  const p = (sorted.length - 1) * q;
  const i = Math.floor(p);
  const f = p - i;
  return sorted[i + 1] === undefined
    ? sorted[i]
    : sorted[i] + f * (sorted[i + 1] - sorted[i]);
};

const median = a => quantile(a, 0.5);

function normCdf(x) {
  const t = 1 / (1 + 0.2316419 * Math.abs(x));
  const d = 0.3989423 * Math.exp(-x * x / 2);
  const p =
    1 -
    d *
      t *
      (0.3193815 +
        t *
          (-0.3565638 +
            t * (1.781478 + t * (-1.821256 + t * 1.330274))));
  return x >= 0 ? p : 1 - p;
}

function normInv(p) {
  let lo = -8;
  let hi = 8;

  for (let i = 0; i < 80; i++) {
    const mid = (lo + hi) / 2;
    if (normCdf(mid) < p) lo = mid;
    else hi = mid;
  }

  return (lo + hi) / 2;
}

function logGamma(z) {
  const c = [
    676.5203681218851,
    -1259.1392167224028,
    771.3234287776531,
    -176.6150291621406,
    12.507343278686905,
    -0.13857109526572012,
    9.984369578019572e-6,
    1.5056327351493116e-7
  ];

  if (z < 0.5) {
    return Math.log(Math.PI) -
      Math.log(Math.sin(Math.PI * z)) -
      logGamma(1 - z);
  }

  z -= 1;
  let x = 0.9999999999998099;

  for (let i = 0; i < c.length; i++) {
    x += c[i] / (z + i + 1);
  }

  const t = z + c.length - 0.5;

  return (
    0.5 * Math.log(2 * Math.PI) +
    (z + 0.5) * Math.log(t) -
    t +
    Math.log(x)
  );
}

function betaCF(a, b, x) {
  const MAX = 200;
  const EPS = 3e-14;
  const FPMIN = 1e-300;

  const qab = a + b;
  const qap = a + 1;
  const qam = a - 1;

  let c = 1;
  let d = 1 - (qab * x) / qap;

  if (Math.abs(d) < FPMIN) d = FPMIN;

  d = 1 / d;
  let h = d;

  for (let m = 1; m <= MAX; m++) {
    const m2 = 2 * m;

    let aa =
      (m * (b - m) * x) /
      ((qam + m2) * (a + m2));

    d = 1 + aa * d;
    if (Math.abs(d) < FPMIN) d = FPMIN;

    c = 1 + aa / c;
    if (Math.abs(c) < FPMIN) c = FPMIN;

    d = 1 / d;
    h *= d * c;

    aa =
      (-(a + m) * (qab + m) * x) /
      ((a + m2) * (qap + m2));

    d = 1 + aa * d;
    if (Math.abs(d) < FPMIN) d = FPMIN;

    c = 1 + aa / c;
    if (Math.abs(c) < FPMIN) c = FPMIN;

    d = 1 / d;

    const del = d * c;
    h *= del;

    if (Math.abs(del - 1) < EPS) break;
  }

  return h;
}

function regBeta(x, a, b) {
  if (x <= 0) return 0;
  if (x >= 1) return 1;

  const bt = Math.exp(
    logGamma(a + b) -
      logGamma(a) -
      logGamma(b) +
      a * Math.log(x) +
      b * Math.log1p(-x)
  );

  return x < (a + 1) / (a + b + 2)
    ? (bt * betaCF(a, b, x)) / a
    : 1 - (bt * betaCF(b, a, 1 - x)) / b;
}

function studentTCdf(t, df) {
  const x = df / (df + t * t);
  const ib = regBeta(x, df / 2, 0.5);

  return t >= 0 ? 1 - 0.5 * ib : 0.5 * ib;
}

function tCritical(df, tailProbability) {
  let lo = 0;
  let hi = 100;

  for (let i = 0; i < 100; i++) {
    const mid = (lo + hi) / 2;

    if (1 - studentTCdf(mid, df) > tailProbability) {
      lo = mid;
    } else {
      hi = mid;
    }
  }

  return (lo + hi) / 2;
}

// Grubbs uses a two-tailed critical probability.
function tCriticalApprox(df, alpha) {
  const safeDf = Math.max(1, df);
  const tailProbability = Math.max(
    1e-10,
    Math.min(0.499999, alpha / 2)
  );

  return tCritical(safeDf, tailProbability);
}

function grubbs(values, alpha) {
  let a = [...values];
  let indexes = values.map((_, i) => i);
  const out = [];

  if (a.length < 3) return out;

  for (let k = 0; k < Math.min(5, values.length - 2); k++) {
    const m = mean(a);
    const s = sd(a);

    if (!s) break;

    let j = 0;

    for (let i = 1; i < a.length; i++) {
      if (Math.abs(a[i] - m) > Math.abs(a[j] - m)) {
        j = i;
      }
    }

    const G = Math.abs(a[j] - m) / s;
    const n = a.length;
    const t = tCriticalApprox(n - 2, alpha / n);

    const Gcrit =
      ((n - 1) / Math.sqrt(n)) *
      Math.sqrt((t * t) / (n - 2 + t * t));

    if (G > Gcrit) {
      out.push({
        index: indexes[j],
        value: a[j],
        score: G,
        reason:
          `Grubbs G=${G.toFixed(2)} > critical ${Gcrit.toFixed(2)}`
      });

      a.splice(j, 1);
      indexes.splice(j, 1);
    } else {
      break;
    }
  }

  return out;
}

function analyze(vals, dates, name, p) {
  const a = vals.map(Number);
  const m = mean(a);
  const s = sd(a);
  const med = median(a);
  const q1 = quantile(a, 0.25);
  const q3 = quantile(a, 0.75);
  const iqr = q3 - q1;
  const lo = q1 - p.k * iqr;
  const hi = q3 + p.k * iqr;

  const out = [];

  a.forEach((v, i) => {
    if (v < lo || v > hi) {
      out.push({
        index: i,
        value: v,
        test: "IQR",
        reason: `${fmt(v)} outside [${fmt(lo)}, ${fmt(hi)}]`
      });
    }

    const z = s ? Math.abs((v - m) / s) : 0;

    if (z > p.z) {
      out.push({
        index: i,
        value: v,
        test: "Z-score",
        reason: `|z|=${z.toFixed(2)} > ${p.z}`
      });
    }

    if (i >= p.w - 1) {
      const window = a.slice(i - p.w + 1, i + 1);
      const avg = mean(window);
      const dist = avg
        ? (Math.abs(v - avg) / Math.abs(avg)) * 100
        : 0;

      if (dist > p.p) {
        out.push({
          index: i,
          value: v,
          test: "Moving average",
          reason:
            `${dist.toFixed(1)}% from ${p.w}-day moving average (${fmt(avg)})`
        });
      }
    }
  });

  grubbs(a, p.alpha).forEach(g => {
    out.push({
      index: g.index,
      value: g.value,
      test: "Grubbs",
      reason: g.reason
    });
  });

  return {
    name,
    mean: m,
    sd: s,
    median: med,
    iqr,
    values: a,
    dates,
    findings: out.map(f => ({
      ...f,
      name,
      date: dates[f.index],
      direction: f.value >= m ? "Spike" : "Drop"
    }))
  };
}

function params() {
  return {
    k: Math.max(0.1, +$("iqrK").value || 1.5),
    z: Math.max(0.5, +$("zThreshold").value || 2.5),
    w: Math.max(2, Math.floor(+$("maWindow").value || 7)),
    p: Math.max(1, +$("maPercent").value || 30),
    alpha: +$("grubbsAlpha").value || 0.05
  };
}

function parseDate(v) {
  if (v instanceof Date && !isNaN(v)) return v;

  if (typeof v === "number" && v > 20000 && v < 80000) {
    const d = XLSX.SSF.parse_date_code(v);
    return d ? new Date(d.y, d.m - 1, d.d) : null;
  }

  const s = String(v ?? "").trim();
  if (!s) return null;

  // Handle ISO dates without timezone conversion.
  const iso = s.match(/^(\d{4})-(\d{1,2})-(\d{1,2})$/);

  if (iso) {
    return new Date(+iso[1], +iso[2] - 1, +iso[3]);
  }

  const d = new Date(s);
  return isNaN(d) ? null : d;
}

function dateLabel(d) {
  if (!(d instanceof Date)) return String(d ?? "");

  const year = d.getFullYear();
  const month = String(d.getMonth() + 1).padStart(2, "0");
  const day = String(d.getDate()).padStart(2, "0");

  return `${year}-${month}-${day}`;
}

// FIX: Total Output is optional.
// If missing, it is calculated from the individual mine columns.
function normalizeData(raw) {
  if (!raw || raw.length < 2) {
    throw new Error(
      "CSV is empty. Please publish the Data sheet as CSV."
    );
  }

  const headers = raw[0].map(v =>
    String(v ?? "").replace(/^\uFEFF/, "").trim()
  );

  let dateIdx = headers.findIndex(h => /^date$/i.test(h));

  if (dateIdx < 0) dateIdx = 0;

  let totalIdx = headers.findIndex(h =>
    /^total\s*output$/i.test(h)
  );

  if (totalIdx < 0) {
    totalIdx = headers.findIndex(h => /^total$/i.test(h));
  }

  let mineColumns = headers
    .map((header, index) => ({ header, index }))
    .filter(c =>
      /^mine\s*\d+/i.test(c.header) &&
      c.index !== dateIdx &&
      c.index !== totalIdx
    );

  // Support differently named mine columns as a fallback.
  if (mineColumns.length === 0) {
    mineColumns = headers
      .map((header, index) => ({ header, index }))
      .filter(c =>
        c.index !== dateIdx &&
        c.index !== totalIdx &&
        c.header !== "" &&
        !/^(day|weekday)$/i.test(c.header)
      );
  }

  if (mineColumns.length === 0 && totalIdx < 0) {
    throw new Error(
      "No mine output columns found. Check the Data sheet header row."
    );
  }

  function toNumber(value) {
    if (
      value === null ||
      value === undefined ||
      String(value).trim() === ""
    ) {
      return null;
    }

    const n = Number(String(value).replace(/,/g, "").trim());

    return Number.isFinite(n) ? n : null;
  }

  const parsed = [];

  for (const row of raw.slice(1)) {
    const date = parseDate(row[dateIdx]);

    if (!date) continue;

    const item = { Date: date };
    let sum = 0;
    let validCount = 0;

    for (const column of mineColumns) {
      const value = toNumber(row[column.index]);

      if (value !== null) {
        item[column.header] = value;
        sum += value;
        validCount++;
      }
    }

    let total =
      totalIdx >= 0 ? toNumber(row[totalIdx]) : null;

    if (total === null && validCount > 0) {
      total = sum;
    }

    if (total === null) continue;

    item["Total Output"] = total;
    parsed.push(item);
  }

  parsed.sort((a, b) => a.Date - b.Date);

  if (parsed.length === 0) {
    throw new Error(
      "No valid rows found. Check that the Data sheet contains dates and numeric mine output values."
    );
  }

  return parsed;
}

function setData(parsed, label) {
  if (!parsed || parsed.length === 0) {
    throw new Error("No valid data was loaded.");
  }

  rows = parsed;
  sourceLabel = label;

  series = [
    ...new Set(
      rows.flatMap(row =>
        Object.keys(row).filter(key => key !== "Date")
      )
    )
  ];

  if (!series.includes("Total Output")) {
    throw new Error("Could not calculate Total Output.");
  }

  $("seriesSelect").innerHTML = series
    .map(name => `<option value="${name}">${name}</option>`)
    .join("");

  $("seriesSelect").value = "Total Output";

  $("sourceStatus").textContent = `${rows.length} days loaded`;
  $("sourceStatus").className = "pill ok";

  $("dateRange").textContent =
    `${dateLabel(rows[0].Date)} → ${dateLabel(rows.at(-1).Date)}`;

  $("dayCount").textContent =
    `${rows.length} daily records • ${series.filter(s => s !== "Total Output").length} mines detected`;

  render();
}

function compute() {
  if (!rows.length || !series.length) return;

  const p = params();

  stats = series.map(name => {
    const validRows = rows.filter(r => Number.isFinite(r[name]));

    return analyze(
      validRows.map(r => r[name]),
      validRows.map(r => r.Date),
      name,
      p
    );
  });

  findings = stats
    .flatMap(s => s.findings)
    .sort((a, b) => a.date - b.date);

  $("meanTotal").textContent = fmt(
    stats.find(s => s.name === "Total Output")?.mean
  );

  $("sdTotal").textContent = fmt(
    stats.find(s => s.name === "Total Output")?.sd
  );

  $("anomalyCount").textContent = findings.length;
  $("anomalyPill").textContent = `${findings.length} findings`;

  renderStats();
  renderAnomalies();
  renderChart();
}

function render() {
  compute();
}

function renderStats() {
  $("statsTable").querySelector("tbody").innerHTML = stats
    .map(s => `
      <tr>
        <td><b>${s.name}</b></td>
        <td>${fmt(s.mean)}</td>
        <td>${fmt(s.sd)}</td>
        <td>${fmt(s.median)}</td>
        <td>${fmt(s.iqr)}</td>
        ${["IQR", "Z-score", "Moving average", "Grubbs"]
          .map(test => `
            <td class="${s.findings.some(f => f.test === test) ? "flag" : ""}">
              ${s.findings.filter(f => f.test === test).length}
            </td>
          `).join("")}
      </tr>
    `).join("");
}

function renderAnomalies() {
  const fs = findings.slice(0, 500);

  $("anomalyTable").querySelector("tbody").innerHTML = fs.length
    ? fs.map(f => `
        <tr>
          <td>${dateLabel(f.date)}</td>
          <td>${f.name}</td>
          <td>${fmt(f.value)}</td>
          <td>${f.test}</td>
          <td class="${f.direction === "Spike" ? "direction-up" : "direction-down"}">
            ${f.direction === "Spike" ? "▲" : "▼"} ${f.direction}
          </td>
          <td>${f.reason}</td>
        </tr>
      `).join("")
    : `<tr><td colspan="6" class="empty">
         No anomalies detected with current parameters.
       </td></tr>`;
}

function polyFit(x, y, degree) {
  degree = Math.min(degree, x.length - 1);

  const n = degree + 1;
  const A = Array.from(
    { length: n },
    () => Array(n + 1).fill(0)
  );

  for (let r = 0; r < n; r++) {
    for (let c = 0; c < n; c++) {
      A[r][c] = x.reduce(
        (sum, xi) => sum + xi ** (r + c),
        0
      );
    }

    A[r][n] = x.reduce(
      (sum, xi, i) => sum + y[i] * xi ** r,
      0
    );
  }

  for (let i = 0; i < n; i++) {
    let max = i;

    for (let j = i + 1; j < n; j++) {
      if (Math.abs(A[j][i]) > Math.abs(A[max][i])) {
        max = j;
      }
    }

    [A[i], A[max]] = [A[max], A[i]];

    const div = A[i][i];

    if (Math.abs(div) < 1e-12) {
      return x.map(() => NaN);
    }

    for (let c = i; c <= n; c++) {
      A[i][c] /= div;
    }

    for (let r = 0; r < n; r++) {
      if (r === i) continue;

      const f = A[r][i];

      for (let c = i; c <= n; c++) {
        A[r][c] -= f * A[i][c];
      }
    }
  }

  const coef = A.map(r => r[n]);

  return x.map(xi =>
    coef.reduce((sum, c, i) => sum + c * xi ** i, 0)
  );
}

function renderChart() {
  if (!rows.length || !window.Chart) return;

  const name = $("seriesSelect").value || "Total Output";

  const validRows = rows.filter(r => Number.isFinite(r[name]));
  const vals = validRows.map(r => r[name]);
  const dates = validRows.map(r => dateLabel(r.Date));

  const fs = findings.filter(f => f.name === name);
  const bad = new Set(fs.map(f => dateLabel(f.date)));

  const degree = +$("trendDegree").value;
  const type = $("chartType").value;

  const trend = polyFit(vals.map((_, i) => i), vals, degree);
  const datasets = [];

  if (type === "stacked") {
    const mineSeries = series.filter(s => s !== "Total Output");

    mineSeries.forEach((s, j) => {
      datasets.push({
        type: "bar",
        label: s,
        data: rows.map(r => r[s] ?? 0),
        backgroundColor: `hsla(${(j * 37 + 90) % 360},65%,62%,.65)`,
        stack: "mines"
      });
    });
  } else {
    datasets.push({
      type: type === "bar" ? "bar" : "line",
      label: name,
      data: vals,
      borderColor: "#c8f169",
      backgroundColor: type === "bar"
        ? "rgba(200,241,105,.7)"
        : "rgba(200,241,105,.08)",
      borderWidth: 2,
      fill: type === "line",
      tension: 0.22,
      pointRadius: vals.map((_, i) => bad.has(dates[i]) ? 5 : 1.5),
      pointBackgroundColor: vals.map((_, i) =>
        bad.has(dates[i]) ? "#ff6f70" : "#c8f169"
      ),
      pointBorderColor: vals.map((_, i) =>
        bad.has(dates[i]) ? "#ff6f70" : "#c8f169"
      )
    });

    datasets.push({
      type: "line",
      label: `Polynomial trend (degree ${degree})`,
      data: trend,
      borderColor: "#6ac7ed",
      borderDash: [6, 5],
      pointRadius: 0,
      borderWidth: 2,
      fill: false
    });
  }

  if (chart) chart.destroy();

  chart = new Chart($("outputChart"), {
    type: type === "bar" || type === "stacked" ? "bar" : "line",
    data: {
      labels: type === "stacked"
        ? rows.map(r => dateLabel(r.Date))
        : dates,
      datasets
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: {
        mode: "index",
        intersect: false
      },
      plugins: {
        legend: {
          labels: {
            color: "#a9b8c7",
            usePointStyle: true,
            boxWidth: 7
          }
        },
        tooltip: {
          callbacks: {
            label: c => `${c.dataset.label}: ${fmt(c.parsed.y)}`
          }
        }
      },
      scales: {
        x: {
          stacked: type === "stacked",
          ticks: {
            color: "#8192a2",
            maxTicksLimit: 12
          },
          grid: {
            color: "rgba(120,145,165,.09)"
          }
        },
        y: {
          beginAtZero: true,
          stacked: type === "stacked",
          ticks: {
            color: "#8192a2"
          },
          grid: {
            color: "rgba(120,145,165,.12)"
          }
        }
      }
    }
  });
}

async function loadFile(file) {
  const ext = file.name.split(".").pop().toLowerCase();
  let raw;

  if (ext === "csv") {
    raw = parseCSV(await file.text());
  } else {
    const wb = XLSX.read(await file.arrayBuffer(), {
      type: "array",
      cellDates: true
    });

    const sheetName =
      wb.SheetNames.find(n => /^data$/i.test(n)) ||
      wb.SheetNames.find(n => /data/i.test(n));

    if (!sheetName) {
      throw new Error("No Data sheet found in workbook.");
    }

    raw = XLSX.utils.sheet_to_json(wb.Sheets[sheetName], {
      header: 1,
      defval: ""
    });
  }

  setData(normalizeData(raw), file.name);
}

function parseCSV(text) {
  const lines = text
    .replace(/^\uFEFF/, "")
    .split(/\r?\n/)
    .filter(line => line.trim());

  return lines.map(line => {
    const result = [];
    let value = "";
    let quoted = false;

    for (let i = 0; i < line.length; i++) {
      const char = line[i];

      if (char === '"' && line[i + 1] === '"' && quoted) {
        value += '"';
        i++;
      } else if (char === '"') {
        quoted = !quoted;
      } else if (char === "," && !quoted) {
        result.push(value);
        value = "";
      } else {
        value += char;
      }
    }

    result.push(value);
    return result;
  });
}

$("fileInput").addEventListener("change", async event => {
  try {
    const file = event.target.files[0];
    if (file) await loadFile(file);
  } catch (error) {
    alert(error.message);
  }
});

$("loadUrl").addEventListener("click", async () => {
  const url = $("csvUrl").value.trim();

  if (!url) {
    alert("Paste a published CSV URL first.");
    return;
  }

  try {
    const response = await fetch(url);

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }

    const text = await response.text();

    if (/^\s*<!doctype html|^\s*<html/i.test(text)) {
      throw new Error(
        "The URL returned an HTML page instead of CSV. Publish the Data sheet as CSV."
      );
    }

    setData(normalizeData(parseCSV(text)), url);
  } catch (error) {
    alert(
      "Could not load URL. Publish the Data sheet as CSV and confirm it is publicly accessible. Browser CORS may block some URLs. Details: " +
      error.message
    );
  }
});

$("demoBtn").addEventListener("click", () => {
  const start = new Date(2026, 0, 1);

  const raw = [[
    "Date",
    "Mine 1",
    "Mine 2",
    "Mine 3",
    "Mine 4",
    "Mine 5",
    "Total Output"
  ]];

  for (let i = 0; i < 90; i++) {
    const values = Array.from(
      { length: 5 },
      (_, j) =>
        Math.max(
          0,
          50 +
            18 * Math.sin(i / 5 + j) +
            Math.random() * 30 +
            i * 0.45
        )
    );

    if (i === 38) values[2] *= 3;
    if (i === 70) values[0] *= 0.2;

    raw.push([
      new Date(start.getTime() + i * 86400000),
      ...values,
      values.reduce((a, b) => a + b, 0)
    ]);
  }

  setData(normalizeData(raw), "DEMO DATA (synthetic)");
});

$("refreshBtn").addEventListener("click", compute);

[
  "iqrK",
  "zThreshold",
  "maWindow",
  "maPercent",
  "grubbsAlpha",
  "seriesSelect",
  "chartType",
  "trendDegree"
].forEach(id => {
  $(id).addEventListener("change", compute);
});

$("pdfBtn").addEventListener("click", () => {
  if (!rows.length) {
    alert("Load the generator workbook or CSV first.");
    return;
  }

  if (!window.jspdf || !window.jspdf.jsPDF) {
    alert("PDF library did not load. Check your internet connection.");
    return;
  }

  const { jsPDF } = window.jspdf;

  const doc = new jsPDF({
    orientation: "landscape",
    unit: "mm",
    format: "a4"
  });

  const total = stats.find(s => s.name === "Total Output");

  doc.setFillColor(13, 19, 27);
  doc.rect(0, 0, 297, 210, "F");

  doc.setTextColor(200, 241, 105);
  doc.setFont("helvetica", "bold");
  doc.setFontSize(20);
  doc.text("WEYLAND-YUTANI", 15, 18);

  doc.setTextColor(235, 242, 248);
  doc.setFontSize(15);
  doc.text("Mining Operations - Statistical Report", 15, 28);

  doc.setTextColor(145, 163, 181);
  doc.setFontSize(9);
  doc.text(`Source: ${sourceLabel}`, 15, 36);
  doc.text(
    `Period: ${dateLabel(rows[0].Date)} to ${dateLabel(rows.at(-1).Date)} | ${rows.length} days`,
    15,
    42
  );

  doc.autoTable({
    startY: 49,
    head: [[
      "Series",
      "Mean",
      "Std. dev.",
      "Median",
      "IQR",
      "IQR flags",
      "Z flags",
      "MA flags",
      "Grubbs flags"
    ]],
    body: stats.map(s => [
      s.name,
      fmt(s.mean),
      fmt(s.sd),
      fmt(s.median),
      fmt(s.iqr),
      ...["IQR", "Z-score", "Moving average", "Grubbs"].map(
        test => s.findings.filter(f => f.test === test).length
      )
    ]),
    theme: "grid",
    styles: {
      fontSize: 8,
      textColor: [225, 235, 242],
      fillColor: [17, 25, 35],
      lineColor: [45, 61, 75]
    },
    headStyles: {
      fillColor: [35, 50, 62],
      textColor: [200, 241, 105]
    },
    alternateRowStyles: {
      fillColor: [21, 31, 43]
    }
  });

  const y = doc.lastAutoTable.finalY + 10;

  doc.setTextColor(200, 241, 105);
  doc.setFontSize(12);
  doc.text("Total output - overview", 15, y);

  doc.setTextColor(220, 230, 239);
  doc.setFontSize(9);

  doc.text(
    `Mean ${fmt(total?.mean)} | Standard deviation ${fmt(total?.sd)} | Median ${fmt(total?.median)} | IQR ${fmt(total?.iqr)} | Findings ${findings.length}`,
    15,
    y + 7
  );

  try {
    const image = $("outputChart").toDataURL("image/png", 1);

    doc.addPage();
    doc.setFillColor(13, 19, 27);
    doc.rect(0, 0, 297, 210, "F");

    doc.setTextColor(200, 241, 105);
    doc.setFontSize(15);
    doc.text("Output chart and trendline", 15, 18);
    doc.addImage(image, "PNG", 15, 25, 265, 125);
  } catch (error) {
    console.warn("Could not add chart image to PDF:", error);
  }

  doc.addPage();
  doc.setFillColor(13, 19, 27);
  doc.rect(0, 0, 297, 210, "F");

  doc.setTextColor(200, 241, 105);
  doc.setFontSize(15);
  doc.text("Anomaly details - spikes and drops", 15, 18);

  const body = findings.map(f => [
    dateLabel(f.date),
    f.name,
    fmt(f.value),
    f.test,
    f.direction,
    f.reason
  ]);

  doc.autoTable({
    startY: 25,
    head: [[
      "Date",
      "Series",
      "Output",
      "Test",
      "Direction",
      "Explanation"
    ]],
    body: body.length
      ? body
      : [["—", "—", "—", "—", "—", "No anomalies detected."]],
    theme: "grid",
    styles: {
      fontSize: 7,
      overflow: "linebreak",
      cellWidth: "wrap",
      textColor: [225, 235, 242],
      fillColor: [17, 25, 35],
      lineColor: [45, 61, 75]
    },
    headStyles: {
      fillColor: [35, 50, 62],
      textColor: [200, 241, 105]
    },
    columnStyles: {
      5: { cellWidth: 85 }
    },
    margin: { left: 12, right: 12 },
    didDrawPage: () => {
      doc.setTextColor(120, 140, 158);
      doc.setFontSize(8);
      doc.text(
        "WEYLAND-YUTANI | Operations Intelligence",
        12,
        204
      );
    }
  });

  doc.save("Weyland-Yutani-Mining-Analytics-Report.pdf");
});

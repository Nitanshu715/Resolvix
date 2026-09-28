/**
 * RESOLVIX Platform Client Engine
 */

let currentCountry = 'US';
let baselineChartInstance = null;
let recallChartInstance = null;

// Initialize Lucide Icons & Theme
document.addEventListener('DOMContentLoaded', () => {
  if (window.lucide) {
    lucide.createIcons();
  }
  initTheme();
  initAuditCharts();
  initDropZone();
});

// ----------------- Theme Controller -----------------
function initTheme() {
  if (localStorage.theme === 'light') {
    document.documentElement.classList.remove('dark');
  } else {
    document.documentElement.classList.add('dark');
  }
}

function toggleDarkMode() {
  if (document.documentElement.classList.contains('dark')) {
    document.documentElement.classList.remove('dark');
    localStorage.theme = 'light';
  } else {
    document.documentElement.classList.add('dark');
    localStorage.theme = 'dark';
  }
  lucide.createIcons();
  if (baselineChartInstance) baselineChartInstance.destroy();
  if (recallChartInstance) recallChartInstance.destroy();
  initAuditCharts();
}

// ----------------- Navigation Tabs -----------------
function switchTab(tabId) {
  document.querySelectorAll('.view-panel').forEach(el => el.classList.add('hidden'));
  document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.classList.remove('bg-white', 'dark:bg-slate-800', 'text-slate-900', 'dark:text-white', 'shadow-sm');
    btn.classList.add('text-slate-500', 'dark:text-slate-400');
  });

  const activePanel = document.getElementById(`view-${tabId}`);
  const activeBtn = document.getElementById(`tab-btn-${tabId}`);

  if (activePanel) activePanel.classList.remove('hidden');
  if (activeBtn) {
    activeBtn.classList.add('bg-white', 'dark:bg-slate-800', 'text-slate-900', 'dark:text-white', 'shadow-sm');
    activeBtn.classList.remove('text-slate-500', 'dark:text-slate-400');
  }

  lucide.createIcons();
}

// ----------------- Country Selector -----------------
function setCountry(c) {
  currentCountry = c;
  document.querySelectorAll('.country-btn').forEach(b => {
    b.classList.remove('border-emerald-500', 'bg-emerald-50', 'dark:bg-emerald-950/40', 'text-emerald-600', 'dark:text-emerald-400');
    b.classList.add('border-slate-200', 'dark:border-slate-800', 'text-slate-600', 'dark:text-slate-400');
  });

  const btn = document.getElementById(`btn-country-${c}`);
  if (btn) {
    btn.classList.remove('border-slate-200', 'dark:border-slate-800', 'text-slate-600', 'dark:text-slate-400');
    btn.classList.add('border-emerald-500', 'bg-emerald-50', 'dark:bg-emerald-950/40', 'text-emerald-600', 'dark:text-emerald-400');
  }
}

// ----------------- Sample Query Loader -----------------
const SAMPLES = {
  US: {
    name: "Certified Elite Rain Gutters Co.",
    address: "1525 Peoria St, Aurora, CO 80010"
  },
  India: {
    name: "Sri Anand Construction & Trading Private Limited",
    address: "Plot 42, Sector 18, Gurugram, Haryana 122001"
  },
  France: {
    name: "Societe Nouvelle Boulangerie Patisserie SAS",
    address: "14 Rue de la Republique, 75011 Paris, France"
  }
};

function loadSampleQuery() {
  const sample = SAMPLES[currentCountry] || SAMPLES.US;
  document.getElementById('query-name').value = sample.name;
  document.getElementById('query-address').value = sample.address;
  updateKeyPreview(sample.name, sample.address);
  showToast("Loaded sample query entity");
}

function updateKeyPreview(name, address) {
  const normName = name.toLowerCase().replace(/[^a-z0-9]/g, '');
  const addrMatch = address.match(/\d+/);
  const addrNum = addrMatch ? addrMatch[0] : '';
  const streetPart = address.toLowerCase().replace(/[^a-z0-9\s]/g, '').split(/\s+/).find(w => w.length > 3) || 'st';

  document.getElementById('key-comp').innerText = normName.substring(0, 20) || 'none';
  document.getElementById('key-addr').innerText = addrNum ? `${addrNum}_${streetPart}` : 'none';
}

// ----------------- Live Entity Resolution -----------------
async function executeResolution() {
  const name = document.getElementById('query-name').value.trim();
  const address = document.getElementById('query-address').value.trim();
  const threshold = parseInt(document.getElementById('query-threshold').value);

  if (!name) {
    showToast("Please provide a business name", true);
    return;
  }

  updateKeyPreview(name, address);

  const btn = document.getElementById('btn-resolve');
  const originalHtml = btn.innerHTML;
  btn.innerHTML = `<svg class="animate-spin -ml-1 mr-2 h-4 w-4 text-white" fill="none" viewBox="0 0 24 24"><circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle><path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg> Resolving Candidates...`;
  btn.disabled = true;

  try {
    const res = await fetch('/api/resolve', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        country: currentCountry,
        business_name: name,
        business_address: address,
        threshold: threshold
      })
    });

    if (!res.ok) throw new Error("Resolution failed");
    const data = await res.json();
    renderResolutionResults(data);
  } catch (err) {
    showToast(err.message, true);
  } finally {
    btn.innerHTML = originalHtml;
    btn.disabled = false;
    lucide.createIcons();
  }
}

function renderResolutionResults(data) {
  // Update Latency Badge
  const latBadge = document.getElementById('latency-badge');
  latBadge.classList.remove('hidden');
  document.getElementById('latency-val').innerText = `${data.latency_ms} ms`;

  // Status Summary Card
  const statusIcon = document.getElementById('status-icon');
  const statusTitle = document.getElementById('status-title');
  const statusSub = document.getElementById('status-subtitle');

  if (data.is_singleton) {
    statusIcon.className = "w-12 h-12 rounded-xl bg-amber-500/10 text-amber-500 flex items-center justify-center";
    statusIcon.innerHTML = `<i data-lucide="help-circle" class="w-6 h-6"></i>`;
    statusTitle.innerText = "Single Entity (No Confident Matches)";
    statusSub.innerText = `Candidate pool retrieved ${data.candidates_count} records, but none met threshold ${data.threshold || 74}.`;
  } else {
    statusIcon.className = "w-12 h-12 rounded-xl bg-emerald-500/10 text-emerald-500 flex items-center justify-center";
    statusIcon.innerHTML = `<i data-lucide="check-check" class="w-6 h-6"></i>`;
    statusTitle.innerText = `Successfully Resolved (${data.matches_count} Linked Target${data.matches_count > 1 ? 's' : ''})`;
    statusSub.innerText = `Candidate generation retrieved ${data.candidates_count} candidates; ${data.matches_count} confirmed matches.`;
  }

  // Pill
  document.getElementById('candidate-count-pill').innerText = `${data.candidates_count} retrieved (${data.matches_count} linked)`;

  // Render Candidate Cards
  const container = document.getElementById('candidates-container');
  if (data.candidates.length === 0) {
    container.innerHTML = `
      <div class="glass-card rounded-2xl p-8 text-center text-slate-400">
        <p class="text-sm">No candidate matches generated from index.</p>
      </div>
    `;
    return;
  }

  container.innerHTML = data.candidates.map((c, idx) => `
    <div class="glass-card rounded-2xl p-4 border ${c.is_match ? 'border-emerald-500/40 bg-emerald-50/20 dark:bg-emerald-950/20' : 'border-slate-200 dark:border-slate-800'} space-y-3 transition-all hover:scale-[1.005]">
      <div class="flex items-start justify-between">
        <div class="flex items-center space-x-2">
          <span class="px-2 py-0.5 rounded-md font-mono text-[10px] font-bold ${c.source === 'Source 2' ? 'bg-sky-500/10 text-sky-500' : 'bg-indigo-500/10 text-indigo-500'} border border-current/20">${c.source}</span>
          <span class="font-mono text-xs font-semibold text-slate-500">${c.entity_id}</span>
          ${c.is_match ? '<span class="px-2 py-0.5 rounded-full text-[10px] font-bold bg-emerald-500 text-white">MATCH</span>' : '<span class="px-2 py-0.5 rounded-full text-[10px] font-mono text-slate-400 bg-slate-100 dark:bg-slate-800">REJECTED</span>'}
        </div>
        <div class="text-right">
          <span class="text-[10px] font-mono uppercase text-slate-400">Score</span>
          <span class="font-mono text-sm font-extrabold ${c.is_match ? 'text-emerald-500' : 'text-slate-400'} ml-1">${c.similarity_score.toFixed(1)}</span>
        </div>
      </div>

      <div>
        <p class="text-sm font-bold text-slate-900 dark:text-white">${escapeHtml(c.business_name)}</p>
        <p class="text-xs text-slate-500 dark:text-slate-400 flex items-center space-x-1 mt-0.5">
          <i data-lucide="map-pin" class="w-3 h-3 text-slate-400"></i>
          <span>${escapeHtml(c.business_address)}</span>
        </p>
      </div>

      <div class="flex flex-wrap gap-1.5 pt-2 border-t border-slate-100 dark:border-slate-800/60">
        ${c.reasons.map(r => `
          <span class="text-[10px] px-2 py-0.5 rounded-md bg-slate-100 dark:bg-slate-800/80 text-slate-500 dark:text-slate-400 font-mono">${escapeHtml(r)}</span>
        `).join('')}
      </div>
    </div>
  `).join('');

  lucide.createIcons();
}

// ----------------- Drop Zone & Batch Job -----------------
let selectedBatchFile = null;

function initDropZone() {
  const dropZone = document.getElementById('drop-zone');
  const fileInput = document.getElementById('batch-file-input');

  dropZone.addEventListener('click', () => fileInput.click());
  fileInput.addEventListener('change', (e) => {
    if (e.target.files.length) {
      selectedBatchFile = e.target.files[0];
      document.getElementById('file-name-label').innerText = `${selectedBatchFile.name} (${(selectedBatchFile.size/1024).toFixed(1)} KB)`;
    }
  });

  dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('border-emerald-500');
  });

  dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('border-emerald-500');
  });

  dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('border-emerald-500');
    if (e.dataTransfer.files.length) {
      selectedBatchFile = e.dataTransfer.files[0];
      document.getElementById('file-name-label').innerText = `${selectedBatchFile.name} (${(selectedBatchFile.size/1024).toFixed(1)} KB)`;
    }
  });
}

async function startBatchJob() {
  if (!selectedBatchFile) {
    showToast("Please choose a TSV or CSV file first", true);
    return;
  }

  const country = document.getElementById('batch-country').value;
  const thresh = document.getElementById('batch-threshold').value;

  const formData = new FormData();
  formData.append('file', selectedBatchFile);
  formData.append('country', country);
  formData.append('threshold', thresh);

  const btn = document.getElementById('btn-start-batch');
  btn.disabled = true;

  try {
    const res = await fetch('/api/batch-upload', {
      method: 'POST',
      body: formData
    });
    if (!res.ok) throw new Error("Batch submission failed");
    const data = await res.json();
    showToast(`Batch job ${data.job_id} launched!`);
    pollBatchJob(data.job_id);
  } catch (err) {
    showToast(err.message, true);
    btn.disabled = false;
  }
}

function pollBatchJob(jobId) {
  const badge = document.getElementById('batch-status-badge');
  const bar = document.getElementById('batch-progress-bar');
  const pct = document.getElementById('batch-progress-pct');
  const procCnt = document.getElementById('batch-processed-cnt');
  const matchCnt = document.getElementById('batch-matched-cnt');
  const singCnt = document.getElementById('batch-singletons-cnt');
  const dlBox = document.getElementById('batch-downloads');

  badge.className = "px-2.5 py-0.5 rounded-full font-mono text-[11px] font-semibold bg-emerald-500/20 text-emerald-500 animate-pulse";
  badge.innerText = "STREAMING RESOLUTION...";

  const interval = setInterval(async () => {
    try {
      const res = await fetch(`/api/batch-status/${jobId}`);
      if (!res.ok) return;
      const job = await res.json();

      procCnt.innerText = job.processed.toLocaleString();
      matchCnt.innerText = job.matched.toLocaleString();
      singCnt.innerText = job.singletons.toLocaleString();

      if (job.total > 0) {
        const p = Math.round((job.processed / job.total) * 100);
        bar.style.width = `${p}%`;
        pct.innerText = `${p}%`;
      }

      if (job.status === 'completed') {
        clearInterval(interval);
        badge.className = "px-2.5 py-0.5 rounded-full font-mono text-[11px] font-semibold bg-emerald-500 text-white";
        badge.innerText = `COMPLETED (${job.duration}s)`;
        bar.style.width = '100%';
        pct.innerText = '100%';
        dlBox.classList.remove('hidden');

        document.getElementById('btn-dl-matches').href = `/api/batch-download/${jobId}/matches`;
        document.getElementById('btn-dl-cands').href = `/api/batch-download/${jobId}/candidates`;
        document.getElementById('btn-start-batch').disabled = false;
        showToast("Batch processing finished!");
      } else if (job.status === 'failed') {
        clearInterval(interval);
        badge.className = "px-2.5 py-0.5 rounded-full font-mono text-[11px] font-semibold bg-red-500 text-white";
        badge.innerText = "FAILED";
        showToast(job.error || "Batch processing encountered an error", true);
        document.getElementById('btn-start-batch').disabled = false;
      }
    } catch (e) {
      console.error(e);
    }
  }, 1000);
}

// ----------------- Audit Charts -----------------
function initAuditCharts() {
  const isDark = document.documentElement.classList.contains('dark');
  const gridColor = isDark ? 'rgba(255, 255, 255, 0.05)' : 'rgba(0, 0, 0, 0.05)';
  const textColor = isDark ? '#94a3b8' : '#64748b';

  // Baseline Comparison Chart
  const ctxBaseline = document.getElementById('baselineChart');
  if (ctxBaseline) {
    baselineChartInstance = new Chart(ctxBaseline, {
      type: 'bar',
      data: {
        labels: ['Baseline A (Singletons)', 'Baseline B (Exact Name)', 'Baseline C (Simple Fuzzy)', 'Baseline D (RESOLVIX Pipeline)'],
        datasets: [{
          label: 'Macro-F0.5 Score',
          data: [0.0516, 0.1796, 0.6579, 0.6853],
          backgroundColor: [
            'rgba(148, 163, 184, 0.4)',
            'rgba(148, 163, 184, 0.6)',
            'rgba(20, 184, 166, 0.7)',
            'rgba(34, 197, 94, 0.85)'
          ],
          borderRadius: 8
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false }
        },
        scales: {
          y: {
            min: 0,
            max: 1.0,
            grid: { color: gridColor },
            ticks: { color: textColor, font: { family: '"JetBrains Mono"' } }
          },
          x: {
            grid: { display: false },
            ticks: { color: textColor, font: { family: '"Plus Jakarta Sans"', size: 10 } }
          }
        }
      }
    });
  }

  // Recall Chart
  const ctxRecall = document.getElementById('recallChart');
  if (ctxRecall) {
    recallChartInstance = new Chart(ctxRecall, {
      type: 'line',
      data: {
        labels: ['Recall@1', 'Recall@5', 'Recall@10', 'Recall@25', 'Complete Recall'],
        datasets: [
          {
            label: 'US Full Corpus (6.18M)',
            data: [43.33, 74.96, 78.03, 81.13, 56.99],
            borderColor: '#22c55e',
            backgroundColor: 'rgba(34, 197, 94, 0.1)',
            tension: 0.3,
            fill: true
          },
          {
            label: 'India Full Corpus (4.13M)',
            data: [29.51, 52.38, 56.72, 62.13, 36.42],
            borderColor: '#14b8a6',
            backgroundColor: 'rgba(20, 184, 166, 0.05)',
            tension: 0.3,
            fill: true
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            position: 'top',
            labels: { color: textColor, font: { family: '"Plus Jakarta Sans"', size: 11 } }
          }
        },
        scales: {
          y: {
            min: 0,
            max: 100,
            grid: { color: gridColor },
            ticks: {
              callback: v => `${v}%`,
              color: textColor,
              font: { family: '"JetBrains Mono"' }
            }
          },
          x: {
            grid: { color: gridColor },
            ticks: { color: textColor, font: { family: '"JetBrains Mono"', size: 10 } }
          }
        }
      }
    });
  }
}

// ----------------- Toast Alerts -----------------
function showToast(msg, isError = false) {
  const toast = document.getElementById('toast');
  const toastMsg = document.getElementById('toast-msg');
  const toastIcon = document.getElementById('toast-icon');

  toastMsg.innerText = msg;
  toastIcon.className = isError ? "text-rose-500" : "text-emerald-500";
  toastIcon.innerHTML = isError ? `<i data-lucide="alert-circle" class="w-5 h-5"></i>` : `<i data-lucide="check-circle" class="w-5 h-5"></i>`;

  lucide.createIcons();

  toast.classList.remove('translate-y-20', 'opacity-0');
  setTimeout(() => {
    toast.classList.add('translate-y-20', 'opacity-0');
  }, 3500);
}

function escapeHtml(str) {
  if (!str) return '';
  return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;").replace(/'/g, "&#039;");
}

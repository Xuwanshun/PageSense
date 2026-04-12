const METRIC_PATHS = {
  mrr: "mrr",
  recall3: "recall_at_k.3",
  ndcg3: "ndcg_at_k.3",
  p1_tol: "precision_at_1_fuzzy",
  p2_tol: "precision_at_2_fuzzy",
  attr_recall_tol: "citation_recall_fuzzy_attribution_top_2",
};

const FALLBACK_EXAMPLE_QUESTIONS = [
  "What are the four AI RMF Core functions?",
  "What goal does the AI RMF have according to the executive summary?",
  "What does the MAP function cover in the AI RMF core?",
  "What are the GOVERN categories and subcategories?",
  "How are stakeholder expectations connected to technical requirements in the NASA Systems Engineering Handbook?",
  "How does the NASA systems engineering handbook distinguish technical and programmatic risk?",
];

async function init() {
  const payload = await fetchJson("/api/showcase-data");

  // If the server is unreachable, disable the form entirely.
  if (!payload) {
    renderLoadError();
    renderExampleQuestions(FALLBACK_EXAMPLE_QUESTIONS);
    renderCorpusPreviews([]);
    setupAskForm(false);
    return;
  }

  // Render corpus info and example questions regardless of eval data.
  renderStats(payload.stats || {});
  renderCorpusPreviews(payload.corpus_previews || []);
  renderExampleQuestions(payload.example_questions || FALLBACK_EXAMPLE_QUESTIONS);

  // Render eval metrics only when an evaluation run exists.
  if (payload.summary && payload.summary.retrieval) {
    const summary = payload.summary;
    const retrieval = summary.retrieval;
    const topN = retrieval.attribution_top_n ?? 2;
    METRIC_PATHS.attr_recall_tol = `citation_recall_fuzzy_attribution_top_${topN}`;

    renderRunHeader(payload.latest_run, summary.questions, topN);
    renderMetrics(retrieval);
    renderJudge(payload.judge?.summary || summary.llm_judge || null);
    renderPredictionExamples(payload.prediction_examples || []);
  }

  // The ask form works whenever the server is up — no eval run required.
  setupAskForm(true);
}

async function fetchJson(path) {
  try {
    const response = await fetch(path, { cache: "no-store" });
    if (!response.ok) {
      return null;
    }
    return await response.json();
  } catch (error) {
    console.error(`Failed to fetch ${path}`, error);
    return null;
  }
}

function renderLoadError() {
  const runNode = document.getElementById("run-id");
  if (runNode) {
    runNode.textContent = "Could not load showcase API data. Start server: python showcase/server.py";
  }

  const status = document.getElementById("ask-status");
  if (status) {
    status.textContent = "Live API unavailable.";
  }
}

function renderRunHeader(runId, questionCount, topN) {
  const runNode = document.getElementById("run-id");
  if (!runNode) {
    return;
  }
  runNode.textContent = `Run: ${runId || "unknown"} | Questions: ${questionCount ?? "--"} | Attribution Top-N: ${topN}`;
}

function renderStats(stats) {
  const container = document.getElementById("stats-list");
  if (!container) {
    return;
  }

  const docs = Array.isArray(stats.documents) ? stats.documents : [];
  const chunks = typeof stats.chunks === "number" ? stats.chunks : 0;
  const regionCounts = stats.region_counts || {};

  const lines = [
    `<li><strong>${docs.length}</strong> processed documents</li>`,
    `<li><strong>${chunks}</strong> indexed chunks</li>`,
    `<li><strong>${regionCounts.table ?? 0}</strong> table regions</li>`,
    `<li><strong>${regionCounts.figure ?? 0}</strong> figure regions</li>`,
  ];
  container.innerHTML = lines.join("");
}

function safeUrl(raw) {
  const value = String(raw || "").trim();
  return value.startsWith("/") ? value : "";
}

function renderCorpusPreviews(previews) {
  const container = document.getElementById("corpus-previews");
  if (!container) {
    return;
  }

  const rows = Array.isArray(previews) ? previews : [];
  if (rows.length === 0) {
    container.innerHTML = '<p class="corpus-loading">No corpus previews available yet.</p>';
    return;
  }

  container.innerHTML = rows
    .slice(0, 3)
    .map((doc) => {
      const title = escapeHtml(doc.title || doc.source_filename || "Document");
      const desc = escapeHtml(doc.description || "");
      const facts = escapeHtml(doc.facts || "");
      const pdfUrl = safeUrl(doc.pdf_url);
      const coverUrl = safeUrl(doc.page_preview_url);

      const thumbHtml = coverUrl ? `<img src="${coverUrl}" alt="${title} preview" loading="lazy" />` : "";
      const metaHtml = `<div class="doc-preview__meta">
            <p class="doc-preview__title">${title}</p>
            ${desc ? `<p class="doc-preview__desc">${desc}</p>` : ""}
            ${facts ? `<p class="doc-preview__facts">${facts}</p>` : ""}
          </div>`;

      const wrapperStart = pdfUrl ? `<a class="doc-preview__link" href="${pdfUrl}" target="_blank" rel="noopener">` : '<div class="doc-preview__link">';
      const wrapperEnd = pdfUrl ? "</a>" : "</div>";

      return `<article class="doc-preview">
          ${wrapperStart}
            <div class="doc-preview__thumb">${thumbHtml}</div>
            ${metaHtml}
          ${wrapperEnd}
        </article>`;
    })
    .join("");
}

function renderMetrics(retrieval) {
  const cards = document.querySelectorAll(".metric");
  cards.forEach((card) => {
    const metricId = card.getAttribute("data-key");
    const path = METRIC_PATHS[metricId];
    const node = card.querySelector(".metric-value");
    if (!node || !path) {
      return;
    }

    const value = getNestedValue(retrieval, path);
    if (typeof value !== "number") {
      node.textContent = "--";
      return;
    }
    animateValue(node, value);
  });
}

function renderJudge(summary) {
  const container = document.getElementById("judge-list");
  if (!container) {
    return;
  }

  if (!summary) {
    container.innerHTML = "<p>No judge data found in latest run.</p>";
    return;
  }

  const rows = [
    ["Correctness", summary.answer_correctness_mean],
    ["Faithfulness", summary.faithfulness_groundedness_mean],
    ["Citation Attribution", summary.citation_attribution_mean],
    ["Completeness", summary.answer_completeness_mean],
    ["Overall", summary.overall_mean],
    ["Hallucination Rate", summary.hallucination_rate],
  ];

  container.innerHTML = rows
    .map(([label, value]) => {
      const numeric = typeof value === "number" ? value.toFixed(label.includes("Rate") ? 3 : 2) : "--";
      return `<div class="judge-item"><strong>${escapeHtml(label)}</strong><br />${numeric}</div>`;
    })
    .join("");
}

function renderPredictionExamples(predictions) {
  const container = document.getElementById("qa-list");
  if (!container) {
    return;
  }

  if (!Array.isArray(predictions) || predictions.length === 0) {
    container.innerHTML = "<p>No prediction examples available.</p>";
    return;
  }

  container.innerHTML = predictions
    .slice(0, 3)
    .map((item, idx) => {
      const question = escapeHtml(item.question || "");
      const answer = escapeHtml(item.predicted_answer || "");
      const cites = Array.isArray(item.sources)
        ? item.sources
            .slice(0, 3)
            .map((s) => `${s.source_filename || "unknown"} p.${s.page_number ?? "?"}`)
            .join(" | ")
        : "No citations";

      return `
        <article class="qa-item">
          <h3>Example ${idx + 1}: ${question}</h3>
          <p>${answer}</p>
          <p class="qa-cites">${escapeHtml(cites)}</p>
        </article>
      `;
    })
    .join("");
}

function setupAskForm(enabled) {
  const form = document.getElementById("ask-form");
  const input = document.getElementById("question-input");
  const status = document.getElementById("ask-status");
  const exampleButtons = document.getElementById("example-buttons");

  if (!form || !input || !status) {
    return;
  }

  if (exampleButtons) {
    exampleButtons.addEventListener("click", (event) => {
      const button = event.target.closest(".example-btn");
      if (!button) {
        return;
      }
      const question = String(button.dataset.question || "").trim();
      if (!question) {
        return;
      }
      input.value = question;
      if (enabled) {
        form.requestSubmit();
      } else {
        status.textContent = "Live query is unavailable. Run: python showcase/server.py";
      }
    });
  }

  if (!enabled) {
    status.textContent = "Live query is unavailable. Run: python showcase/server.py";
    return;
  }

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const question = String(input.value || "").trim();
    if (!question) {
      status.textContent = "Enter a question first.";
      return;
    }

    status.textContent = "Querying pipeline...";
    const result = await postJson("/api/ask", { question, top_k: 4 });
    if (!result || result.error) {
      status.textContent = result?.detail || result?.error || "Query failed.";
      return;
    }

    status.textContent = `Completed in ${result.latency_ms ?? "?"} ms`;
    renderAskResult(result);
  });
}

async function postJson(path, payload) {
  try {
    const response = await fetch(path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    return await response.json();
  } catch (error) {
    console.error("POST failed", error);
    return null;
  }
}

function renderAskResult(result) {
  const container = document.getElementById("ask-result");
  if (!container) {
    return;
  }

  const sources = Array.isArray(result.sources) ? result.sources : [];
  const sourceHtml = sources
    .map((src) => {
      const line = `${src.source_filename || "unknown"} | p.${src.page_number ?? "?"} | score ${src.score ?? "?"}`;
      return `<div class="source-item">${escapeHtml(line)}</div>`;
    })
    .join("");

  const routerSummary = result.router
    ? `table_agent=${Boolean(result.router.use_table_agent)} | figure_agent=${Boolean(result.router.use_figure_agent)}`
    : "router=n/a";

  const specialistSummary = Array.isArray(result.specialists) && result.specialists.length > 0
    ? result.specialists.map((s) => `${s.agent_name}: ${s.output}`).join(" || ")
    : "No specialist branch used";

  container.innerHTML = `
    <h3>Answer</h3>
    <p class="answer-text">${escapeHtml(result.answer || "")}</p>
    <p class="answer-meta">${escapeHtml(routerSummary)}</p>
    <p class="answer-meta">${escapeHtml(specialistSummary)}</p>
    <div class="source-list">${sourceHtml || '<p class="empty">No sources returned.</p>'}</div>
  `;
}

function renderExampleQuestions(rawQuestions) {
  const container = document.getElementById("example-buttons");
  if (!container) {
    return;
  }

  const seen = new Set();
  const questions = [];
  const candidates = Array.isArray(rawQuestions) && rawQuestions.length > 0 ? rawQuestions : FALLBACK_EXAMPLE_QUESTIONS;
  for (const candidate of candidates) {
    const question = String(candidate || "").trim();
    if (!question) {
      continue;
    }
    const key = question.toLowerCase();
    if (seen.has(key)) {
      continue;
    }
    seen.add(key);
    questions.push(question);
    if (questions.length >= 6) {
      break;
    }
  }

  container.innerHTML = "";
  if (questions.length === 0) {
    container.innerHTML = "<p class='example-loading'>No example questions found.</p>";
    return;
  }

  for (const question of questions) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "example-btn";
    button.dataset.question = question;
    button.textContent = question;
    container.appendChild(button);
  }
}

function getNestedValue(obj, path) {
  const parts = path.split(".");
  let cursor = obj;
  for (const part of parts) {
    if (cursor == null || !(part in cursor)) {
      return null;
    }
    cursor = cursor[part];
  }
  return cursor;
}

function animateValue(node, target) {
  const final = Number(target.toFixed(4));
  const start = performance.now();
  const duration = 650;

  function tick(now) {
    const progress = Math.min((now - start) / duration, 1);
    const eased = 1 - Math.pow(1 - progress, 3);
    node.textContent = (final * eased).toFixed(4);
    if (progress < 1) {
      requestAnimationFrame(tick);
    }
  }

  requestAnimationFrame(tick);
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

void init();

/* Showcase client for the banking-triage API.
   Vanilla JS on purpose: one container serves the page and the API, no CDN,
   no build step. All API text is inserted with textContent, never innerHTML. */
(() => {
  "use strict";

  const ROUTES = {
    faq: { label: "Curated FAQ", tag: "tag-green", short: "FAQ", caption: "answered from reviewed copy" },
    exchange_rate: { label: "ECB reference rate", tag: "tag-cyan", short: "rate", caption: "answered with a dated rate" },
    clarify: { label: "Clarification", tag: "tag-amber", short: "ask", caption: "asked instead of guessing" },
    escalate: { label: "Human handoff", tag: "tag-red", short: "handoff", caption: "handed to a person" },
  };

  const REASONS = {
    classifier:
      "Confidence and margin cleared both thresholds, and curated copy exists for this intent.",
    explicit_currency_pair:
      "An explicit pair plus an amount outranked the classifier. Complaint and app-how-to cues were checked first.",
    follow_up_currency_pair:
      "This reply completed the previous clarification, so it was not re-classified.",
    currency_pair_missing:
      "A rate question needs two currencies. The workflow asked instead of inventing the missing side.",
    low_confidence:
      "Confidence fell below the escalation floor, so the workflow asked rather than guessing.",
    ambiguous:
      "The prediction did not clear both thresholds (confidence and top-1/top-2 margin), so the candidates are surfaced instead of asserting one.",
    protected_intent:
      "Fraud and dispute intents never auto-answer, regardless of confidence.",
    no_curated_answer:
      "The intent was confident but has no reviewed copy yet. It was handed off instead of improvising policy.",
    unsupported_pair:
      "The rate provider does not publish one of these currencies. The failure became a visible clarification.",
    tool_unavailable:
      "The rate tool could not be reached. The turn degraded into a handoff with the error recorded.",
    empty_text: "The message contained no usable text.",
    fallback: "The workflow fell back to a safe handoff.",
  };

  const EXAMPLES = [
    { text: "I would like to change my pin number please", kicker: "FAQ" },
    { text: "convert 250 EUR to CHF", kicker: "quote" },
    { text: "what's the exchange rate?", kicker: "clarify" },
    { text: "I was charged the wrong rate on a 500 USD to EUR payment", kicker: "guardrail" },
    { text: "I lost my card, please block it before someone uses it", kicker: "guardrail" },
    { text: "can I use apple pay", kicker: "no copy" },
  ];

  const ROUTE_NODES = new Set(["faq", "exchange", "clarify", "escalate"]);

  const state = {
    sessionId: newSessionId(),
    busy: false,
    messages: 0,
    intents: [],
    filter: "all",
    query: "",
    thresholds: { auto_route_threshold: 0.54, min_margin: 0.1 },
  };

  const $ = (id) => document.getElementById(id);

  // --------------------------------------------------------------- helpers

  function newSessionId() {
    const raw =
      typeof crypto !== "undefined" && crypto.randomUUID
        ? crypto.randomUUID()
        : `${Date.now().toString(16)}${Math.random().toString(16).slice(2)}`;
    return raw.replaceAll("-", "").slice(0, 12);
  }

  function el(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  }

  function setText(id, value) {
    const node = $(id);
    if (node) node.textContent = value;
  }

  function pct(value, digits = 1) {
    return typeof value === "number" ? `${(value * 100).toFixed(digits)}%` : "—";
  }

  function ms(value) {
    return typeof value === "number" ? `${value.toFixed(1)} ms` : "—";
  }

  function count(value) {
    return typeof value === "number" ? value.toLocaleString("en-US") : "—";
  }

  function money(value) {
    return typeof value === "number"
      ? value.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 })
      : "—";
  }

  function displayIntent(name) {
    if (!name) return "unknown";
    const words = name.replaceAll("_", " ").replace(/\s+/g, " ").trim();
    return words.charAt(0).toUpperCase() + words.slice(1);
  }

  function routeMeta(route) {
    return ROUTES[route] || { label: route || "Unknown", tag: "tag-muted", short: "?", caption: "" };
  }

  function routeTag(route, extra) {
    const meta = routeMeta(route);
    return el("span", `tag ${meta.tag}${extra ? ` ${extra}` : ""}`, route);
  }

  function reasonText(reason) {
    return REASONS[reason] || "The router recorded this reason in the response.";
  }

  async function fetchJSON(url, options) {
    const response = await fetch(url, options);
    if (!response.ok) {
      let message = `${response.status} ${response.statusText}`;
      try {
        const problem = await response.json();
        if (problem && (problem.detail || problem.title)) {
          message = problem.detail || problem.title;
        }
      } catch {
        /* keep the status line */
      }
      throw new Error(message);
    }
    return response.json();
  }

  // ------------------------------------------------------------ page boot

  async function loadHealth() {
    const pill = $("health-pill");
    const text = $("health-text");
    try {
      const health = await fetchJSON("/healthz");
      pill.classList.remove("is-bad");
      pill.classList.add("is-ok");
      text.textContent = `ok · ${health.model_version || "model loaded"}`;
    } catch {
      pill.classList.remove("is-ok");
      pill.classList.add("is-bad");
      text.textContent = "degraded";
    }
  }

  async function loadModel() {
    try {
      const model = await fetchJSON("/model");
      setText("model-version", model.model_version);
      setText("model-trained", model.trained_on);
      setText("model-vectorizer", model.vectorizer);
      setText("footer-model", `${model.model_version} · ${model.n_intents} intents`);
      if (model.thresholds) {
        if (typeof model.thresholds.auto_route_threshold === "number") {
          state.thresholds.auto_route_threshold = model.thresholds.auto_route_threshold;
        }
        if (typeof model.thresholds.min_margin === "number") {
          state.thresholds.min_margin = model.thresholds.min_margin;
        }
      }
    } catch {
      setText("model-version", "unavailable");
    }
  }

  function applyMetrics(data) {
    const workflow = data.workflow;
    const policy = data.classifier_policy;
    const probes = data.probes;
    const model = data.model;
    const latency = workflow.latency;

    if (model && model.thresholds) {
      state.thresholds.auto_route_threshold = model.thresholds.confidence;
      state.thresholds.min_margin = model.thresholds.margin;
    }

    setText("kpi-coverage", pct(workflow.coverage));
    setText("kpi-precision", pct(workflow.precision));
    setText("kpi-probes", `${probes.passed}/${probes.total}`);
    setText("kpi-latency", ms(latency.p50));

    setText("out-auto", `${count(workflow.auto_answered)} · ${pct(workflow.coverage)}`);
    setText("out-correct", `${count(workflow.correct)} · ${pct(workflow.correct_rate)}`);
    setText("out-clarify", `${count(workflow.clarified)} · ${pct(workflow.clarified / workflow.n)}`);
    setText("out-nocopy", `${count(workflow.no_copy)} · ${pct(workflow.no_copy / workflow.n)}`);
    setText("out-protected", `${count(workflow.protected)} · ${pct(workflow.protected / workflow.n)}`);
    setText("out-precision", pct(workflow.precision));

    setText("ceiling-coverage", pct(policy.coverage));
    setText("ceiling-precision", pct(policy.precision));
    const meter = $("ceiling-meter");
    if (meter) meter.style.width = `${Math.min(100, policy.coverage * 100)}%`;
    setText("ceiling-f1", model.test_macro_f1.toFixed(4));
    setText("ceiling-acc", model.test_accuracy.toFixed(4));

    setText("probe-pill", `${probes.passed} / ${probes.total} passed`);
    setText("probe-routes", pct(probes.route_accuracy, 0));
    setText("probe-calls", `${probes.tool_correct} / ${probes.tool_expected}`);
    setText("probe-precision", probes.tool_precision.toFixed(2));
    setText("probe-recall", probes.tool_recall.toFixed(2));

    setText("lat-p50", ms(latency.p50));
    setText("lat-p95", ms(latency.p95));
    setText("lat-p99", ms(latency.p99));
    setText("safe-predicted", count(data.safety.predicted_never_auto));
    setText("safe-gold", count(data.safety.gold_never_auto));

    setText("caveat-ceiling", pct(policy.coverage));
    setText("caveat-precision", pct(policy.precision));
    setText("caveat-workflow", pct(workflow.coverage));
  }

  async function loadMetrics() {
    try {
      applyMetrics(await fetchJSON("/static/metrics.json"));
    } catch {
      /* the page still works without the static evidence block */
    }
  }

  // ------------------------------------------------------------ chat demo

  function scrollChat() {
    const log = $("chat-log");
    log.scrollTop = log.scrollHeight;
  }

  function addMessage(role, text, data) {
    const empty = $("chat-empty");
    if (empty) empty.remove();

    const row = el("div", `msg msg-${role}`);
    row.appendChild(el("span", "msg-avatar", role === "user" ? "U" : role === "error" ? "!" : "◆"));

    const body = el("div", "msg-body");
    body.appendChild(el("div", "msg-bubble", text));

    if (role === "assistant" && data) {
      const meta = el("div", "msg-meta");
      meta.appendChild(el("span", `tag ${routeMeta(data.route).tag}`, data.route));
      meta.appendChild(el("span", "", displayIntent(data.intent)));
      meta.appendChild(el("span", "", pct(data.confidence)));
      meta.appendChild(el("span", "", ms(data.latency_ms)));

      const link = el("button", "trace-link", "trace →");
      link.type = "button";
      link.addEventListener("click", () => {
        renderTrace(data);
        if (window.matchMedia("(max-width: 980px)").matches) {
          $("trace").scrollIntoView({ behavior: "smooth", block: "start" });
        }
      });
      meta.appendChild(link);
      body.appendChild(meta);
    }

    row.appendChild(body);
    $("chat-log").appendChild(row);
    scrollChat();
  }

  function addTyping() {
    const empty = $("chat-empty");
    if (empty) empty.remove();

    const row = el("div", "msg msg-assistant");
    row.appendChild(el("span", "msg-avatar", "◆"));
    const body = el("div", "msg-body");
    const bubble = el("div", "msg-bubble");
    const dots = el("span", "typing");
    for (let i = 0; i < 3; i += 1) dots.appendChild(el("i"));
    bubble.appendChild(dots);
    body.appendChild(bubble);
    row.appendChild(body);
    $("chat-log").appendChild(row);
    scrollChat();
    return row;
  }

  function setBusy(busy) {
    state.busy = busy;
    $("send").disabled = busy;
    $("chat-dot").classList.toggle("is-busy", busy);
  }

  async function sendMessage(text) {
    if (state.busy || !text.trim()) return;
    setBusy(true);
    addMessage("user", text.trim());
    const typing = addTyping();

    try {
      const data = await fetchJSON("/triage", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ text: text.trim(), session_id: state.sessionId }),
      });
      typing.remove();
      state.messages += 1;
      addMessage("assistant", data.answer, data);
      renderTrace(data);
    } catch (error) {
      typing.remove();
      addMessage("error", `The workflow could not answer: ${error.message}`);
    } finally {
      setBusy(false);
      $("chat-input").focus();
    }
  }

  // --------------------------------------------------------- trace render

  function pipeline(steps, route) {
    const done = new Set(turnSteps(steps));
    const act = route === "exchange_rate" ? "exchange" : route;
    const nodes = [
      { label: "classify", done: done.has("triage") },
      { label: "decide", done: true },
      { label: "act", done: done.has(act), hint: routeMeta(route).short },
      { label: "compose", done: done.has("compose") },
    ];

    const wrap = el("div", "pipeline");
    nodes.forEach((node, index) => {
      const step = el("div", `pipeline-step${node.done ? " is-done" : ""}`);
      step.appendChild(el("span", "pipeline-num", String(index + 1).padStart(2, "0")));
      step.appendChild(el("span", "pipeline-name", node.hint ? `${node.label} · ${node.hint}` : node.label));
      if (node.hint) step.title = `act: ${node.hint}`;
      wrap.appendChild(step);
      if (index < nodes.length - 1) wrap.appendChild(el("span", "pipeline-arrow", "→"));
    });
    return wrap;
  }

  function turnSteps(steps) {
    // The checkpointer accumulates steps across a session; a turn-level trace
    // starts at the most recent "triage" node.
    const all = steps || [];
    const start = all.lastIndexOf("triage");
    return start >= 0 ? all.slice(start) : all;
  }

  function section(label, extra) {
    const wrap = el("div", "trace-section");
    const head = el("div", "trace-label");
    head.appendChild(el("span", "", label));
    if (extra) head.appendChild(el("span", "", extra));
    wrap.appendChild(head);
    return wrap;
  }

  function classificationSection(data) {
    const wrap = section("classification", "top-3");
    const line = el("div", "intent-line");
    line.appendChild(el("span", "intent-name", data.intent));
    line.appendChild(el("span", "intent-display", displayIntent(data.intent)));
    line.appendChild(el("span", "intent-conf", pct(data.confidence)));
    wrap.appendChild(line);

    const threshold = state.thresholds.auto_route_threshold;
    const bar = el("div", `bar${data.confidence < threshold ? " is-warm" : ""}`);
    const fill = el("i", "bar-fill");
    fill.style.width = `${Math.max(1.5, data.confidence * 100)}%`;
    bar.appendChild(fill);
    const marker = el("span", "bar-threshold");
    marker.style.left = `${threshold * 100}%`;
    marker.title = `auto-answer threshold ${pct(threshold)}`;
    bar.appendChild(marker);
    wrap.appendChild(bar);

    const topAlt = data.alternatives && data.alternatives.length ? data.alternatives[0].score : 0;
    const margin = Math.max(0, data.confidence - topAlt);
    const caption = el("div", "bar-caption");
    caption.appendChild(
      el(
        "span",
        "",
        `margin ${margin.toFixed(2)} · floor ${state.thresholds.min_margin.toFixed(2)} · threshold ${pct(threshold)}`,
      ),
    );
    caption.appendChild(el("span", "", "confidence"));
    wrap.appendChild(caption);

    if (data.alternatives && data.alternatives.length) {
      const list = el("div", "alt-list");
      data.alternatives.forEach((alt) => {
        const row = el("div", "alt-row");
        row.appendChild(el("span", "alt-name", alt.intent));
        const track = el("span", "alt-track");
        const inner = el("i");
        inner.style.width = `${Math.max(2, alt.score * 100)}%`;
        track.appendChild(inner);
        row.appendChild(track);
        row.appendChild(el("span", "alt-score", pct(alt.score)));
        list.appendChild(row);
      });
      wrap.appendChild(list);
    }
    return wrap;
  }

  function decisionSection(data) {
    const wrap = section("decision");
    const route = routeMeta(data.route);
    const tags = el("div", "trace-route");
    tags.appendChild(routeTag(data.route));
    tags.appendChild(el("span", "trace-reason", data.route_reason || "no reason recorded"));
    const pill = el(
      "span",
      `pill ${data.decision === "auto_answer" ? "pill-green" : data.decision === "clarify" ? "pill-amber" : "pill-red"}`,
      data.decision,
    );
    tags.appendChild(pill);
    wrap.appendChild(tags);
    wrap.appendChild(el("p", "reason-text", reasonText(data.route_reason)));

    const kv = el("div", "kv-grid");
    kv.appendChild(kvItem("outcome", route.caption || route.label));
    kv.appendChild(kvItem("needs human", data.needs_human ? "yes" : "no"));
    const steps = kvItem("steps executed", turnSteps(data.steps).join(" → ") || "—");
    steps.classList.add("kv-wide");
    kv.appendChild(steps);
    wrap.appendChild(kv);
    return wrap;
  }

  function kvItem(label, value) {
    const item = el("div", "kv");
    item.appendChild(el("span", "", label));
    item.appendChild(el("b", "", value));
    return item;
  }

  function actionSection(data) {
    const wrap = section("action");
    let hasContent = false;

    if (data.currency_query) {
      hasContent = true;
      const kv = el("div", "kv-grid");
      const pair = `${data.currency_query.base} → ${data.currency_query.quote}`;
      kv.appendChild(kvItem("currency pair", pair));
      kv.appendChild(kvItem("amount", money(data.currency_query.amount)));
      wrap.appendChild(kv);
    }

    if (data.sources && data.sources.length) {
      hasContent = true;
      const list = el("div", "source-list");
      data.sources.forEach((source) => {
        const item = el("div", "source-item");
        item.appendChild(el("span", "source-kind", source.kind));
        const isURL = typeof source.ref === "string" && source.ref.startsWith("http");
        if (isURL) {
          const link = el("a", "source-ref", source.ref);
          link.href = source.ref;
          link.target = "_blank";
          link.rel = "noopener";
          item.appendChild(link);
        } else {
          item.appendChild(el("span", "source-ref", source.ref || "—"));
        }
        list.appendChild(item);
      });
      wrap.appendChild(list);
    }

    if (!hasContent) {
      wrap.appendChild(el("p", "empty-note", "No tool call and no external source: this turn asserts nothing."));
    }

    if (data.errors && data.errors.length) {
      const box = el("div", "error-box");
      box.appendChild(el("b", "", "degraded"));
      data.errors.forEach((error) => box.appendChild(el("div", "", error)));
      wrap.appendChild(box);
    }
    return wrap;
  }

  function metaSection(data) {
    const wrap = section("response");
    const grid = el("div", "trace-meter");
    const items = [
      ["latency", ms(data.latency_ms)],
      ["model", data.model_version || "—"],
      ["request", data.request_id || "—"],
    ];
    items.forEach(([label, value]) => {
      const cell = el("div");
      cell.appendChild(el("span", "", label));
      cell.appendChild(el("b", "", value));
      grid.appendChild(cell);
    });
    wrap.appendChild(grid);
    return wrap;
  }

  function renderTrace(data) {
    const trace = $("trace");
    trace.replaceChildren();
    setText("trace-turn", `turn ${state.messages || 1}`);

    const tags = el("div", "trace-route");
    tags.appendChild(routeTag(data.route, "trace-route-tag"));
    tags.appendChild(el("span", "trace-reason", routeMeta(data.route).label));
    if (data.route_reason === "follow_up_currency_pair") {
      tags.appendChild(el("span", "pill pill-cyan", "follow-up"));
    }
    trace.appendChild(tags);

    trace.appendChild(pipeline(data.steps, data.route));
    trace.appendChild(classificationSection(data));
    trace.appendChild(decisionSection(data));
    trace.appendChild(actionSection(data));
    trace.appendChild(metaSection(data));
  }

  // --------------------------------------------------------- intents table

  const FILTERS = [
    { id: "all", label: "All" },
    { id: "curated", label: "Curated copy" },
    { id: "protected", label: "Human-only" },
    { id: "uncovered", label: "No copy" },
  ];

  function matchesFilter(intent) {
    if (state.filter === "curated") return intent.faq_covered;
    if (state.filter === "protected") return !intent.auto_answerable;
    if (state.filter === "uncovered") return !intent.faq_covered;
    return true;
  }

  function renderFilters() {
    const wrap = $("intent-filters");
    wrap.replaceChildren();
    FILTERS.forEach((filter) => {
      const total =
        filter.id === "all"
          ? state.intents.length
          : state.intents.filter((intent) => {
              if (filter.id === "curated") return intent.faq_covered;
              if (filter.id === "protected") return !intent.auto_answerable;
              return !intent.faq_covered;
            }).length;
      const button = el("button", `chip${state.filter === filter.id ? " is-active" : ""}`, `${filter.label} ${total}`);
      button.type = "button";
      button.addEventListener("click", () => {
        state.filter = filter.id;
        renderFilters();
        renderIntents();
      });
      wrap.appendChild(button);
    });
  }

  function renderIntents() {
    const body = $("intent-body");
    if (!body) return;
    body.replaceChildren();

    const query = state.query.trim().toLowerCase();
    const rows = state.intents.filter((intent) => {
      if (!matchesFilter(intent)) return false;
      if (!query) return true;
      return (
        intent.name.toLowerCase().includes(query) ||
        intent.display.toLowerCase().includes(query) ||
        String(intent.id) === query
      );
    });

    if (!rows.length) {
      const row = el("tr");
      const cell = el("td", "table-empty", "No intents match this filter.");
      cell.colSpan = 4;
      row.appendChild(cell);
      body.appendChild(row);
    } else {
      rows.forEach((intent) => {
        const row = el("tr");
        row.appendChild(el("td", "intent-id", String(intent.id)));
        const nameCell = el("td");
        nameCell.appendChild(el("code", "intent-label", intent.name));
        nameCell.appendChild(el("span", "intent-display", intent.display));
        row.appendChild(nameCell);

        const copyCell = el("td");
        copyCell.appendChild(
          intent.faq_covered
            ? el("span", "badge badge-yes", "reviewed")
            : el("span", "badge badge-no", "none"),
        );
        row.appendChild(copyCell);

        const autoCell = el("td");
        autoCell.appendChild(
          intent.auto_answerable
            ? el("span", "badge badge-yes", "allowed")
            : el("span", "badge badge-warn", "never"),
        );
        row.appendChild(autoCell);
        body.appendChild(row);
      });
    }

    setText("intent-count", `${rows.length} of ${state.intents.length || 77} intents`);
  }

  async function loadIntents() {
    try {
      state.intents = await fetchJSON("/intents");
      renderFilters();
      renderIntents();
    } catch {
      setText("intent-count", "unavailable");
    }
  }

  // -------------------------------------------------------------- wire up

  function renderExamples() {
    const wrap = $("example-chips");
    if (!wrap) return;
    EXAMPLES.forEach((example) => {
      const button = el("button", "chip");
      button.type = "button";
      button.appendChild(el("span", "chip-kicker", example.kicker));
      button.appendChild(el("span", "", example.text));
      button.addEventListener("click", () => {
        $("chat-input").value = example.text;
        autoGrow();
        sendMessage(example.text);
      });
      wrap.appendChild(button);
    });
  }

  function autoGrow() {
    const input = $("chat-input");
    input.style.height = "auto";
    input.style.height = `${Math.min(input.scrollHeight, 140)}px`;
  }

  function resetConversation() {
    state.sessionId = newSessionId();
    state.messages = 0;
    $("session-label").textContent = state.sessionId.slice(0, 8);
    $("chat-log").replaceChildren();

    const empty = el("div", "chat-empty");
    empty.id = "chat-empty";
    empty.appendChild(el("p", "chat-empty-title", "Start with an example"));
    const chips = el("div", "chips");
    chips.id = "example-chips";
    empty.appendChild(chips);
    empty.appendChild(el("p", "hint", "Or type your own message below."));
    $("chat-log").appendChild(empty);
    renderExamples();

    setText("trace-turn", "no turn yet");
    $("trace").replaceChildren();
    const traceEmpty = el("div", "trace-empty");
    traceEmpty.appendChild(
      el(
        "p",
        "",
        "Send a message to see the classifier, the route decision, the action, and the answer's provenance.",
      ),
    );
    $("trace").appendChild(traceEmpty);
    $("chat-input").focus();
  }

  function wire() {
    $("session-label").textContent = state.sessionId.slice(0, 8);
    renderExamples();

    // The sample must work on whatever host serves the page, not just localhost.
    const origin = window.location.origin.startsWith("http") ? window.location.origin : "http://127.0.0.1:8000";
    setText(
      "curl-code",
      `curl -s ${origin}/triage \\\n  -H 'content-type: application/json' \\\n  -d '{"text": "convert 250 EUR to CHF"}'`,
    );

    $("chat-form").addEventListener("submit", (event) => {
      event.preventDefault();
      const input = $("chat-input");
      const text = input.value;
      if (!text.trim()) return;
      input.value = "";
      autoGrow();
      sendMessage(text);
    });

    $("chat-input").addEventListener("input", autoGrow);
    $("chat-input").addEventListener("keydown", (event) => {
      if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        $("chat-form").requestSubmit();
      }
    });

    $("new-chat").addEventListener("click", resetConversation);

    $("intent-search").addEventListener("input", (event) => {
      state.query = event.target.value;
      renderIntents();
    });

    $("copy-curl").addEventListener("click", async (event) => {
      const button = event.currentTarget;
      const text = $("curl-code").textContent;
      try {
        await navigator.clipboard.writeText(text);
        button.textContent = "Copied";
      } catch {
        button.textContent = "Copy failed";
      }
      window.setTimeout(() => {
        button.textContent = "Copy";
      }, 1600);
    });
  }

  document.addEventListener("DOMContentLoaded", () => {
    wire();
    loadHealth();
    loadMetrics();
    loadModel();
    loadIntents();
  });
})();

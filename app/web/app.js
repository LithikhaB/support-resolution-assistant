"use strict";
let conversation = null;
let busy = false;
const byId = id => document.getElementById(id);
const el = (tag, text, className) => {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (className) node.className = className;
  return node;
};
async function send(request, updatedIssue = null) {
  if (busy) return;
  busy = true;
  const status = byId("status");
  status.hidden = false;
  status.className = "";
  status.textContent = "Finding a useful next step… The first response can take a little longer.";
  document.querySelectorAll("button").forEach(button => button.disabled = true);
  try {
    const response = await fetch("/api/v1/conversation", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(request)});
    const result = await response.json();
    if (!response.ok) {
      const detail = result.detail;
      const retry = response.headers?.get?.("Retry-After");
      const message = typeof detail === "string" ? detail : Array.isArray(detail) ? detail.map(item => item.msg.replace(/^Value error, /, "")).join("; ") : "Unable to process this request. Please try again.";
      throw new Error(message + (retry ? ` Retry in ${retry} seconds.` : ""));
    }
    conversation = {...request, turns: request.turns.map(turn => ({...turn, message: turn.message.replace(/(password|passcode|OTP|verification code)\s*(?:is\s+|[:=]\s*)[^\s,;]+/gi, "$1= [REDACTED]" )}))};
    if (result.conversation_id) {
      conversation.conversation_id = result.conversation_id;
      conversation.revision = result.revision;
    }
    byId("intake").hidden = true;
    byId("new-ticket").hidden = false;
    byId("issues").replaceChildren(...result.issues.map(renderIssue));
    status.hidden = true;
    const target = byId(`response-${updatedIssue || result.issues[0].issue_id}`);
    target?.focus({preventScroll: true});
    target?.scrollIntoView({behavior: "smooth", block: "start"});
  } catch (error) {
    status.className = "error";
    status.textContent = error.message || "Unable to connect. Check that the server is running.";
  } finally {
    busy = false;
    document.querySelectorAll("button").forEach(button => button.disabled = false);
  }
}
function sourcesPanel(resolution, issueId) {
  const details = el("details", undefined, "sources");
  const history = (resolution.historical_cases || []).slice(0, 3);
  details.append(el("summary", `Source records (${resolution.sources.length + history.length})`));
  details.open = true;
  if (resolution.sources.length) details.append(el("h3", "Knowledge-base articles"));
  for (const source of resolution.sources) {
    const article = el("div", undefined, "source");
    article.id = `source-${issueId}-${source.citation_id}`;
    article.append(el("strong", `[${source.citation_id}] ${source.title}`));
    article.append(el("small", `${source.doc_id} · AI-authored synthetic · ${source.authority}`));
    const quotes = el("details");
    quotes.append(el("summary", "Inspect exact source evidence"));
    for (const quote of source.quotes) quotes.append(el("p", `${quote.field === "plan_step" ? `Step ${quote.step_id}` : quote.field}: ${quote.text}`));
    article.append(quotes);
    details.append(article);
  }
  if (history.length) details.append(el("h3", "Similar resolved tickets"));
  for (const item of history) {
    const article = el("div", undefined, "source");
    article.id = `source-${issueId}-${item.citation_id}`;
    const relationship = item.relationship === "similar_category" ? "Category comparison (different incident)" : "Similar simulated resolved case";
    article.append(el("strong", `[${item.citation_id}] ${relationship}: ${item.title}`), el("small", `Ticket ID: ${item.doc_id} · ${item.outcome_status || "simulated_resolved"}`));
    const outcome = el("details");
    outcome.append(el("summary", "Inspect historical resolution"), el("p", item.resolution));
    article.append(outcome);
    details.append(article);
  }
  details.append(el("small", "These are synthetic support procedures for this demonstration."));
  return details;
}
function citedText(node, text, issueId) {
  const refs = [...new Set([...text.matchAll(/\[([ST]\d+)\]/g)].map(match => match[1]))];
  node.append(el("span", text.replace(/\[([ST]\d+)\]/g, "").replace(/[ \t]+/g, " ").replace(/\s+([.,;:!?])/g, "$1").trim(), "instruction-text"));
  if (!refs.length) return;
  const citations = el("div", undefined, "step-citations");
  citations.append(el("span", "Evidence", "citation-label"));
  for (const ref of refs) {
    const link = el("a", `[${ref}] · ${ref.startsWith("S") ? "KB" : "Ticket"}`, "citation");
    link.href = `#source-${issueId}-${ref}`;
    link.title = `Inspect ${ref.startsWith("S") ? "knowledge-base article" : "historical resolved ticket"} ${ref}`;
    link.onclick = event => {
      event?.preventDefault?.();
      const source = byId(`source-${issueId}-${ref}`);
      if (!source) return;
      let parent = source.parentElement;
      while (parent) { if (parent.tagName === "DETAILS") parent.open = true; parent = parent.parentElement; }
      const quotes = source.querySelector?.("details");
      if (quotes) quotes.open = true;
      source.scrollIntoView?.({behavior: "smooth", block: "nearest"});
    };
    citations.append(link);
  }
  node.append(citations);
}
function fieldChip(label, value, evidence, fallback = "No quoted evidence; provisional model prediction.") {
  const chip = el("details", undefined, "chip");
  const why = evidence?.length ? evidence.map(item => `“${item.text}”`).join("; ") : fallback;
  const summary = el("summary", `${label}: ${value}`);
  summary.title = why;
  chip.append(summary, el("small", `Why: ${why}`));
  return chip;
}
function quickAnswers(question) {
  // A connection staying online does not establish normal wired throughput.
  // Keep slow-speed answers as quoted text rather than send a false connection fact.
  if (/wired device also slow/i.test(question)) return [];
  if (/Ethernet|wired/i.test(question) && !/Without using Ethernet/i.test(question)) return [
    ["Wired also drops", {wired_connection: "failing"}],
    ["Wired stays connected", {wired_connection: "working"}],
    ["No wired test available", {wired_connection: "unavailable"}]
  ];
  if (/one device|every wireless|all your Wi-Fi/i.test(question)) return [
    ["One wireless device", {wireless_devices: "one"}], ["All wireless devices", {wireless_devices: "all"}]
  ];
  if (/pending or settled/i.test(question) && !/Which charge/i.test(question)) return [
    ["Payment pending", {billing_status: "pending"}], ["Payment settled", {billing_status: "settled"}]
  ];
  if (/completely unavailable or intermittent/i.test(question)) return [
    ["Complete loss", {impact: "complete_loss"}], ["Intermittent", {impact: "intermittent"}]
  ];
  return [];
}
function renderIssue(issue) {
  const resolution = issue.resolution;
  const panel = el("article", undefined, "panel");
  const decision = resolution.decision;
  const header = el("div", undefined, "decision");
  const unsupported = resolution.analysis.scope_status === "unsupported";
  header.append(el("h2", `Issue ${issue.issue_id} · ${unsupported ? "irrelevant request" : decision.action.replaceAll("_", " ")}`),
    el("span", `${decision.priority} priority`, `badge ${decision.priority}`),
    el("span", `Target: ${decision.target.replaceAll("_", " ")}`, "badge"));
  header.title = decision.reasons.join("; ");
  panel.append(header, el("small", "Agent review required · advisory only; no handoff created"), el("h3", "Customer complaint"), el("p", issue.complaint, "message"));
  if (unsupported) {
    panel.append(el("h2", resolution.customer_plan.title),
      el("p", resolution.customer_plan.summary), el("small", resolution.customer_plan.note));
    return panel;
  }
  for (const turn of conversation.turns.filter(turn => turn.issue_id === issue.issue_id)) panel.append(el("p", turn.message, "reply"));
  const response = el("section", undefined, "response");
  response.id = `response-${issue.issue_id}`;
  response.tabIndex = -1;
  const plan = resolution.language_plan || resolution.customer_plan;
  const analysis = resolution.analysis;
  const products = analysis.products.map(item => item.product.replaceAll("_", " ")).join(", ") || "not identified";
  const fields = el("div", undefined, "fields");
  const category = analysis.category || `Provisional: ${analysis.candidates.slice(0, 2).map(item => item.category.replaceAll("_", " ")).join(" / ") || "unclassified"}`;
  fields.append(fieldChip("Category", category.replaceAll("_", " "), analysis.category_evidence, "Category is an advisory model estimate; review the reported symptoms."),
    fieldChip("Product", products, analysis.products, "No affected service or product has been identified yet."),
    fieldChip("Severity", analysis.severity.value, analysis.severity.evidence, analysis.severity.rule === "category_default" ? "Estimated priority from the accepted category; no quoted impact evidence was available." : "Customer impact is unclear; record which services and people are affected."),
    fieldChip("Sentiment", ["unknown", "neutral"].includes(analysis.sentiment.value) ? "Neutral" : analysis.sentiment.value,
      analysis.sentiment.value === "unknown" ? [] : analysis.sentiment.evidence, analysis.sentiment.value === "unknown" || analysis.sentiment.rule === "neutral_default"
        ? "Neutral is the default; no explicit emotional tone was detected." : analysis.sentiment.rule));
  const attempts = analysis.actions.filter(item => item.status === "attempted");
  const displayedAttempts = attempts.filter((action, index) => !attempts.some((other, otherIndex) =>
    otherIndex !== index && other.text.toLowerCase().includes(action.text.toLowerCase()) &&
    (other.text.length > action.text.length || otherIndex < index)));
  for (const action of displayedAttempts) fields.append(fieldChip("Already tried", action.text, [action]));
  response.append(el("h2", "1. Complaint understanding"), fields, el("h2", "2. Recommended resolution"), el("strong", plan.title, "plan-stage"), el("p", resolution.language_summary || plan.summary));
  const trust = el("div", undefined, "trust");
  const validation = el("span", `Citation validation: ${resolution.validation.status}`, `badge ${resolution.validation.status}`);
  validation.title = resolution.validation.scope + " " + resolution.validation.issues.join("; ");
  const generatedSteps = resolution.language_plan && JSON.stringify(resolution.language_plan.steps) !== JSON.stringify(resolution.customer_plan.steps);
  const languageBadge = resolution.language_status === "generated_for_review"
    ? `${generatedSteps ? "LLM-assisted plan · protected repair steps" : "LLM introduction · local steps"} · ${resolution.language_provider} · ${resolution.language_model}`
    : `Local deterministic plan · ${resolution.language_status}`;
  trust.append(validation, el("span", languageBadge, "badge"), el("span", "AI-authored synthetic data", "badge"));
  if (resolution.evidence_reuse === "exact_cache_revalidated") trust.append(el("span", "24-hour source cache - revalidated", "badge"));
  if (resolution.evidence_reuse === "reviewed_history_revalidated") trust.append(el("span", "Reviewed recent case - revalidated", "badge"));
  if (resolution.evidence_reuse === "semantic_hit_revalidated") trust.append(el("span", "Similar source selection reused · revalidated", "badge"));
  response.append(trust);
  if (plan.steps.length) {
    const steps = el("ol", undefined, "resolution-steps");
    for (const step of plan.steps) {
      const item = el("li");
      citedText(item, step, issue.issue_id);
      steps.append(item);
    }
    response.append(steps);
  }
  if (plan.note) {
    const note = el("div", undefined, "plan-note");
    note.append(el("strong", "Restrictions and escalation"));
    citedText(note, plan.note, issue.issue_id);
    response.append(note);
  }
  const evidence = el("aside", undefined, "evidence-column");
  evidence.append(el("h2", "3. Evidence & citations"));
  const legend = el("div", undefined, "citation-legend");
  legend.append(el("strong", "What the citations mean"),
    el("p", "S1, S2… = knowledge-base articles (KB)."),
    el("p", "T1, T2… = historical resolved tickets. Demo outcomes are simulated."));
  evidence.append(legend, sourcesPanel(resolution, issue.issue_id));
  if (!resolution.sources.length) evidence.append(el("p", "No KB article was retained for this response; record the missing details.", "muted"));
  const layout = el("div", undefined, "triage-grid");
  layout.append(response, evidence);
  panel.append(layout);
  const form = el("form", undefined, "followup");
  const questions = resolution.clarification_questions;
  for (const question of questions) form.append(el("p", question, "question"));
  const label = el("label", questions.length ? "Record the customer's answer" : "Add an observation or follow-up");
  const input = el("textarea");
  input.id = `answer-${issue.issue_id}`;
  input.required = true;
  input.maxLength = 1000;
  input.rows = 3;
  input.placeholder = questions.length ? "Type your answer here…" : "What else would you like to add?";
  label.htmlFor = input.id;
  const submit = el("button", "Send reply");
  submit.type = "submit";
  for (const [answer, observations] of quickAnswers(questions[0] || "")) {
    const button = el("button", answer, "secondary quick-answer");
    button.type = "button";
    button.onclick = () => {
      if (conversation.turns.length >= 8) { input.setCustomValidity("Eight replies reached. Start a new ticket with a summary."); input.reportValidity(); return; }
      send({...conversation, turns: [...conversation.turns, {issue_id: issue.issue_id, message: answer, observations}]}, issue.issue_id);
    };
    form.append(button);
  }
  form.append(label, input, submit);
  form.onsubmit = event => {
    event.preventDefault();
    const message = input.value.trim();
    if (conversation.turns.length >= 8) {
      input.setCustomValidity("This case has reached eight replies. Start a new case with a summary of the current issue.");
      input.reportValidity();
      return;
    }
    if (!message) { input.setCustomValidity("Please enter your answer."); input.reportValidity(); return; }
    const last = conversation.turns.at(-1);
    if (last?.issue_id === issue.issue_id && last.message === message) {
      input.setCustomValidity("This answer is already included above. Add a new detail or correction.");
      input.reportValidity();
      return;
    }
    send({...conversation, turns: [...conversation.turns, {issue_id: issue.issue_id, message}]}, issue.issue_id);
  };
  input.oninput = () => input.setCustomValidity("");
  panel.append(form);
  return panel;
}
byId("create-form").onsubmit = event => {
  event.preventDefault();
  send({query: byId("complaint").value.trim(), turns: [], max_sources: 2});
};
let historyItems = [];
let selectedHistory = new Set();
let historyBusy = false;
let historyReturnFocus = null;

function setHistoryOpen(open) {
  const drawer = byId("history-drawer");
  if (!drawer) return;
  drawer.hidden = !open;
  byId("history-backdrop").hidden = !open;
  byId("recent-conversations").setAttribute?.("aria-expanded", String(open));
  const workspace = byId("workspace");
  if (workspace) workspace.inert = open;
  if (typeof document.body !== "undefined") document.body.style.overflow = open ? "hidden" : "";
  if (open) { historyReturnFocus = document.activeElement; byId("close-history").focus(); }
  else { byId("history-confirmation").hidden = true; historyReturnFocus?.focus?.(); }
}
function startNewComplaint() {
  conversation = null;
  try { if (typeof localStorage !== "undefined") localStorage.removeItem("support_conversation_id"); } catch { /* Storage can be disabled. */ }
  byId("issues").replaceChildren();
  byId("intake").hidden = false;
  byId("new-ticket").hidden = false;
  byId("status").hidden = true;
  byId("complaint").value = "";
  setHistoryOpen(false);
  if (typeof window !== "undefined" && window.location.hash.startsWith("#source-")) window.history.replaceState(null, "", window.location.pathname + window.location.search);
  byId("complaint").focus();
}
byId("new-ticket").onclick = startNewComplaint;

function updateHistorySelection() {
  const button = byId("delete-selected");
  button.textContent = `Delete selected (${selectedHistory.size})`;
  button.disabled = !selectedHistory.size || historyBusy;
  const all = byId("history-select-all");
  all.checked = historyItems.length > 0 && selectedHistory.size === historyItems.length;
  all.indeterminate = selectedHistory.size > 0 && !all.checked;
  all.disabled = !historyItems.length || historyBusy;
}
function renderHistoryList() {
  const panel = byId("recent-list");
  panel.replaceChildren();
  if (!historyItems.length) panel.append(el("p", "No saved conversations yet. Your new complaint starts here.", "muted"));
  for (const item of historyItems) {
    const row = el("div", undefined, "history-row");
    const label = el("label", undefined, "history-checkbox");
    const checkbox = el("input");
    checkbox.type = "checkbox"; checkbox.checked = selectedHistory.has(item.conversation_id);
    checkbox.setAttribute?.("aria-label", `Select conversation: ${item.query}`);
    checkbox.onchange = () => {
      if (checkbox.checked) selectedHistory.add(item.conversation_id); else selectedHistory.delete(item.conversation_id);
      byId("history-confirmation").hidden = true;
      updateHistorySelection();
    };
    label.append(checkbox);
    const content = el("div");
    const open = el("button", item.query, "history-open"); open.type = "button";
    open.onclick = () => restoreConversation(item.conversation_id);
    const date = new Date(item.updated_at);
    content.append(open, el("small", `${Number.isNaN(date.getTime()) ? "Saved conversation" : date.toLocaleString()} · revision ${item.revision}`));
    row.append(label, content); panel.append(row);
  }
  updateHistorySelection();
}
async function loadRecentConversations() {
  const status = byId("history-status"); status.textContent = "Loading saved conversations…";
  byId("recent-list").replaceChildren();
  try {
    const response = await fetch("/api/v1/conversations"); const result = await response.json();
    if (!response.ok) throw new Error(typeof result.detail === "string" ? result.detail : "History unavailable.");
    historyItems = result.conversations;
    selectedHistory = new Set();
    byId("history-confirmation").hidden = true;
    status.textContent = "Select conversations to delete, or open a complaint to re-check it against current evidence.";
    renderHistoryList();
    return true;
  } catch (error) { status.textContent = error.message; historyItems = []; selectedHistory = new Set(); updateHistorySelection(); return false; }
}
async function restoreConversation(id) {
  if (busy || historyBusy) return;
  historyBusy = true;
  byId("history-status").textContent = "Opening conversation…";
  try {
    const response = await fetch(`/api/v1/conversations/${id}`);
    const saved = await response.json();
    if (!response.ok) throw new Error(typeof saved.detail === "string" ? saved.detail : "Saved conversation unavailable.");
    setHistoryOpen(false);
    await send({...saved.request, conversation_id:saved.conversation_id, revision:saved.revision});
  } catch (error) { byId("history-status").textContent = error.message; }
  finally { historyBusy = false; updateHistorySelection(); }
}
async function deleteSelectedConversations() {
  if (historyBusy || !selectedHistory.size) return;
  historyBusy = true;
  byId("confirm-delete").disabled = true;
  byId("cancel-delete").disabled = true;
  updateHistorySelection();
  const targets = [...selectedHistory];
  let deleted = 0;
  try {
    for (const id of targets) {
      const response = await fetch(`/api/v1/conversations/${id}`, {method:"DELETE"});
      if (!response.ok) throw new Error("Some conversations could not be deleted. Reload the list and try again.");
      deleted++;
      historyItems = historyItems.filter(item => item.conversation_id !== id);
      selectedHistory.delete(id);
      if (conversation?.conversation_id === id) { startNewComplaint(); setHistoryOpen(true); }
    }
    const refreshed = await loadRecentConversations();
    byId("history-status").textContent = `Deleted ${deleted} conversation${deleted === 1 ? "" : "s"}.${refreshed ? "" : " History could not be refreshed; reopen the drawer to retry."}`;
  } catch (error) { byId("history-status").textContent = `${deleted} deleted. ${error.message}`; }
  finally {
    historyBusy = false;
    byId("history-confirmation").hidden = true;
    byId("confirm-delete").disabled = false;
    byId("cancel-delete").disabled = false;
    renderHistoryList();
  }
}
if (byId("recent-conversations")) {
  byId("recent-conversations").onclick = async () => { setHistoryOpen(true); await loadRecentConversations(); };
  byId("close-history").onclick = () => setHistoryOpen(false);
  byId("history-backdrop").onclick = () => setHistoryOpen(false);
  byId("history-select-all").onchange = () => {
    selectedHistory = byId("history-select-all").checked ? new Set(historyItems.map(item => item.conversation_id)) : new Set();
    byId("history-confirmation").hidden = true;
    renderHistoryList();
  };
  byId("delete-selected").onclick = () => {
    byId("history-confirmation-text").textContent = `Delete ${selectedHistory.size} selected conversation${selectedHistory.size === 1 ? "" : "s"} and linked reviewed outcomes? This cannot be undone.`;
    byId("history-confirmation").hidden = false;
    byId("cancel-delete").focus();
  };
  byId("cancel-delete").onclick = () => { byId("history-confirmation").hidden = true; byId("delete-selected").focus(); };
  byId("confirm-delete").onclick = deleteSelectedConversations;
  document.addEventListener?.("keydown", event => {
    if (byId("history-drawer").hidden) return;
    if (event.key === "Escape") { setHistoryOpen(false); return; }
    if (event.key !== "Tab") return;
    const nodes = [...byId("history-drawer").querySelectorAll("button:not(:disabled),input:not(:disabled),[tabindex='0']")].filter(node => !node.closest("[hidden]"));
    const first = nodes[0], last = nodes.at(-1);
    if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last?.focus(); }
    else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first?.focus(); }
  });
}
startNewComplaint();
if (typeof window !== "undefined") window.addEventListener("pageshow", event => { if (event.persisted) startNewComplaint(); });

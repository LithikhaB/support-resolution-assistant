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
      if (typeof localStorage !== "undefined") localStorage.setItem("support_conversation_id", result.conversation_id);
    }
    byId("intake").hidden = true;
    byId("new-ticket").hidden = false;
    byId("issues").replaceChildren(...result.issues.map(renderIssue));
    status.hidden = true;
    const target = byId(`response-${updatedIssue || result.issues[0].issue_id}`);
    target.focus({preventScroll: true});
    target.scrollIntoView({behavior: "smooth", block: "start"});
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
  details.append(el("summary", `Sources (${resolution.sources.length + history.length})`));
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
  let start = 0;
  for (const match of text.matchAll(/\[([ST]\d+)\]/g)) {
    node.append(el("span", text.slice(start, match.index)));
    const link = el("a", match[0], "citation");
    link.href = `#source-${issueId}-${match[1]}`;
    link.title = `Inspect source ${match[1]}`;
    link.onclick = () => {
      const source = byId(`source-${issueId}-${match[1]}`);
      const panel = source?.closest?.("details");
      if (panel) panel.open = true;
    };
    node.append(link);
    start = match.index + match[0].length;
  }
  node.append(el("span", text.slice(start)));
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
  fields.append(fieldChip("Category", category.replaceAll("_", " "), analysis.category_evidence),
    fieldChip("Product", products, analysis.products),
    fieldChip("Severity", analysis.severity.value, analysis.severity.evidence, analysis.severity.rule),
    fieldChip("Sentiment", analysis.sentiment.value, analysis.sentiment.evidence, analysis.sentiment.rule));
  for (const action of analysis.actions.filter(item => item.status === "attempted")) fields.append(fieldChip("Already tried", action.text, [action]));
  response.append(fields);
  response.append(el("h2", plan.title), el("p", resolution.language_summary || plan.summary));
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
    const steps = el("ol");
    for (const step of plan.steps) {
      const item = el("li");
      citedText(item, step, issue.issue_id);
      steps.append(item);
    }
    response.append(steps);
  }
  if (plan.note) response.append(el("small", plan.note));
  panel.append(response);
  if (resolution.sources.length) panel.append(sourcesPanel(resolution, issue.issue_id));
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
  if (conversation.conversation_id && resolution.validation.status === "passed" && resolution.sources.length) {
    const review = el("details");
    review.append(el("summary", "Record a reviewed simulated resolution"));
    const reviewForm = el("form");
    const note = el("textarea");
    note.id = `outcome-${issue.issue_id}`;
    note.required = true; note.minLength = 20; note.maxLength = 1000;
    const noteLabel = el("label", "Record the observed result and the checks reviewed (at least 20 characters)");
    noteLabel.htmlFor = note.id;
    const confirmLabel = el("label", "I reviewed these conditional steps and am recording a simulated demo outcome.");
    const confirm = el("input"); confirm.type = "checkbox"; confirm.required = true;
    confirmLabel.append(confirm);
    const save = el("button", "Save reviewed outcome"); save.type = "submit";
    const feedback = el("p"); feedback.setAttribute?.("role", "status");
    reviewForm.append(noteLabel, note, confirmLabel, save, feedback);
    reviewForm.onsubmit = async event => {
      event.preventDefault();
      if (!confirm.checked) return;
      save.disabled = true;
      try {
        const response = await fetch(`/api/v1/conversations/${conversation.conversation_id}/review`, {method:"POST", headers:{"Content-Type":"application/json"},body:JSON.stringify({issue_id:issue.issue_id,revision:conversation.revision,outcome_note:note.value.trim(),confirmation:"simulated_resolution_reviewed"})});
        const result = await response.json();
        if (!response.ok) throw new Error(typeof result.detail === "string" ? result.detail : "Review could not be saved.");
        feedback.textContent = "Reviewed simulated history saved for this browser. It does not confirm another customer's cause.";
      } catch (error) { feedback.textContent = error.message; }
      finally { save.disabled = false; }
    };
    review.append(reviewForm); panel.append(review);
  }
  return panel;
}
byId("create-form").onsubmit = event => {
  event.preventDefault();
  send({query: byId("complaint").value.trim(), turns: [], max_sources: 2});
};
byId("new-ticket").onclick = () => {
  conversation = null;
  if (typeof localStorage !== "undefined") localStorage.removeItem("support_conversation_id");
  byId("issues").replaceChildren();
  byId("intake").hidden = false;
  byId("new-ticket").hidden = true;
  byId("status").hidden = true;
  byId("complaint").value = "";
  byId("complaint").focus();
};

async function restoreConversation(id) {
  try {
    const response = await fetch(`/api/v1/conversations/${id}`);
    const saved = await response.json();
    if (!response.ok) throw new Error(saved.detail || "Saved conversation unavailable.");
    byId("recent-list").hidden = true;
    await send({...saved.request, conversation_id:saved.conversation_id, revision:saved.revision});
  } catch (error) { byId("status").hidden = false; byId("status").textContent = error.message; }
}
if (byId("recent-conversations")) {
  byId("recent-conversations").onclick = async () => {
    const panel = byId("recent-list"); panel.hidden = false;
    try {
      const response = await fetch("/api/v1/conversations"); const result = await response.json();
      if (!response.ok) throw new Error(result.detail || "History unavailable.");
      panel.replaceChildren(el("h2", "Recent conversations · this browser"));
      if (!result.conversations.length) panel.append(el("p", "No saved conversations yet."));
      for (const item of result.conversations) {
        const row = el("div");
        const open = el("button", item.query, "secondary"); open.type = "button";
        open.onclick = () => restoreConversation(item.conversation_id);
        const remove = el("button", "Delete", "secondary"); remove.type = "button";
        remove.onclick = async () => {
          const response = await fetch(`/api/v1/conversations/${item.conversation_id}`, {method:"DELETE"});
          if (response.ok) { if (typeof localStorage !== "undefined" && localStorage.getItem("support_conversation_id") === item.conversation_id) localStorage.removeItem("support_conversation_id"); byId("recent-conversations").onclick(); }
          else { byId("status").hidden=false; byId("status").textContent="Could not delete the saved conversation."; }
        };
        row.append(open, remove); panel.append(row);
      }
    } catch (error) { panel.replaceChildren(el("p", error.message)); }
  };
  if (typeof localStorage !== "undefined") {
    const active = localStorage.getItem("support_conversation_id");
    if (active && /^[a-f0-9-]{36}$/i.test(active)) restoreConversation(active);
  }
}

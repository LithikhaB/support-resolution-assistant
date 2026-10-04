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
      throw new Error(typeof detail === "string" ? detail : Array.isArray(detail) ? detail.map(item => item.msg.replace(/^Value error, /, "")).join("; ") : "Unable to process this request. Please try again.");
    }
    conversation = {...request, turns: request.turns.map(turn => ({...turn, message: turn.message.replace(/(password|passcode|OTP|verification code)\s*(?:is\s+|[:=]\s*)[^\s,;]+/gi, "$1= [REDACTED]" )}))};
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
function sourcesPanel(resolution) {
  const details = el("details", undefined, "sources");
  const history = (resolution.historical_cases || []).slice(0, 3);
  details.append(el("summary", `Sources (${resolution.sources.length + history.length})`));
  for (const source of resolution.sources) {
    const article = el("div", undefined, "source");
    article.append(el("strong", `[${source.citation_id}] ${source.title}`));
    article.append(el("small", `${source.doc_id} · AI-authored synthetic · ${source.authority}`));
    const quotes = el("details");
    quotes.append(el("summary", "Inspect exact source evidence"));
    for (const quote of source.quotes) quotes.append(el("p", `${quote.field}: ${quote.text}`));
    article.append(quotes);
    details.append(article);
  }
  for (const item of history) {
    const article = el("div", undefined, "source");
    const relationship = item.relationship === "similar_category" ? "Category comparison (different incident)" : "Similar simulated resolved case";
    article.append(el("strong", `[${item.citation_id}] ${relationship}: ${item.title}`), el("p", item.resolution));
    details.append(article);
  }
  details.append(el("small", "These are synthetic support procedures for this demonstration."));
  return details;
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
  header.append(el("h2", `Issue ${issue.issue_id} · ${decision.action.replaceAll("_", " ")}`),
    el("span", `${decision.priority} priority`, `badge ${decision.priority}`),
    el("span", `Target: ${decision.target.replaceAll("_", " ")}`, "badge"));
  header.title = decision.reasons.join("; ");
  panel.append(header, el("small", "Agent review required · advisory only; no handoff created"), el("h3", "Customer complaint"), el("p", issue.complaint, "message"));
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
    ? `${generatedSteps ? "LLM plan" : "LLM introduction · local steps"} · ${resolution.language_provider} · ${resolution.language_model}`
    : `Local deterministic plan · ${resolution.language_status}`;
  trust.append(validation, el("span", languageBadge, "badge"), el("span", "AI-authored synthetic data", "badge"));
  response.append(trust);
  if (plan.steps.length) {
    const steps = el("ol");
    for (const step of plan.steps) steps.append(el("li", step));
    response.append(steps);
  }
  if (plan.note) response.append(el("small", plan.note));
  panel.append(response);
  if (resolution.sources.length) panel.append(sourcesPanel(resolution));
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
byId("new-ticket").onclick = () => {
  conversation = null;
  byId("issues").replaceChildren();
  byId("intake").hidden = false;
  byId("new-ticket").hidden = true;
  byId("status").hidden = true;
  byId("complaint").value = "";
  byId("complaint").focus();
};

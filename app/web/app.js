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
  const history = (resolution.historical_cases || []).filter(item => (resolution.language_summary || "").includes(`[${item.citation_id}]`));
  details.append(el("summary", `Sources (${resolution.sources.length + history.length})`));
  for (const source of resolution.sources) {
    const article = el("div", undefined, "source");
    article.append(el("strong", `[${source.citation_id}] ${source.title}`));
    for (const quote of source.quotes.filter(quote => quote.field !== "scope")) article.append(el("p", quote.text));
    details.append(article);
  }
  for (const item of history) {
    const article = el("div", undefined, "source");
    article.append(el("strong", `[${item.citation_id}] ${item.title}`), el("p", item.resolution));
    details.append(article);
  }
  details.append(el("small", "These are synthetic support procedures for this demonstration."));
  return details;
}
function renderIssue(issue) {
  const resolution = issue.resolution;
  const panel = el("article", undefined, "panel");
  panel.append(el("h2", "Your complaint"), el("p", issue.complaint, "message"));
  for (const turn of conversation.turns.filter(turn => turn.issue_id === issue.issue_id)) panel.append(el("p", turn.message, "reply"));
  const response = el("section", undefined, "response");
  response.id = `response-${issue.issue_id}`;
  response.tabIndex = -1;
  const plan = resolution.language_plan || resolution.customer_plan;
  const analysis = resolution.analysis;
  const products = analysis.products.map(item => item.product.replaceAll("_", " ")).join(", ") || "not identified";
  response.append(el("p", `Category: ${(analysis.category || "needs more detail").replaceAll("_", " ")} · Product: ${products} · Severity: ${analysis.severity.value} · Sentiment: ${analysis.sentiment.value}`, "muted"));
  response.append(el("h2", plan.title), el("p", resolution.language_summary || plan.summary));
  if (plan.steps.length) {
    const steps = el("ol");
    for (const step of plan.steps) steps.append(el("li", step));
    response.append(steps);
  }
  if (resolution.acknowledged_actions.length) response.append(el("p", `Already tried: ${resolution.acknowledged_actions.join("; ")}`, "muted"));
  if (plan.note) response.append(el("small", plan.note));
  panel.append(response);
  if (resolution.sources.length) panel.append(sourcesPanel(resolution));
  const form = el("form", undefined, "followup");
  const questions = resolution.clarification_questions;
  for (const question of questions) form.append(el("p", question, "question"));
  const label = el("label", questions.length ? "Your answer" : "Add a detail or ask a follow-up");
  const input = el("textarea");
  input.id = `answer-${issue.issue_id}`;
  input.required = true;
  input.maxLength = 1000;
  input.rows = 3;
  input.placeholder = questions.length ? "Type your answer here…" : "What else would you like to add?";
  label.htmlFor = input.id;
  const submit = el("button", "Send reply");
  submit.type = "submit";
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
  send({query: byId("complaint").value.trim(), turns: [], max_sources: 3});
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

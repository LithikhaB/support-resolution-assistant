"use strict";

let activeCase = null;
let busy = false;
const byId = id => document.getElementById(id);
const pretty = value => value.replaceAll("_", " ");
const el = (tag, text, className) => {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (className) node.className = className;
  return node;
};

async function api(path, body) {
  const response = await fetch(path, body === undefined ? {} : {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body)});
  const result = await response.json();
  if (!response.ok) {
    const detail = result.detail;
    throw new Error(typeof detail === "string" ? detail : Array.isArray(detail) ? detail.map(item => item.msg).join("; ") : "Request failed. Please try again.");
  }
  return result;
}

async function task(message, operation, success = "Ticket saved locally.") {
  if (busy) return;
  busy = true;
  const status = byId("status");
  status.hidden = false;
  status.className = "";
  status.textContent = message;
  document.querySelectorAll("button").forEach(button => button.disabled = true);
  try {
    await operation();
    status.textContent = success;
  } catch (error) {
    status.className = "error";
    status.textContent = error.message || "Unable to connect. Check that the server is running.";
  } finally {
    busy = false;
    document.querySelectorAll("button").forEach(button => button.disabled = false);
  }
}

async function refreshRecent() {
  const cases = await api("/api/v1/cases");
  const list = byId("recent");
  list.replaceChildren();
  if (!cases.length) list.append(el("p", "No saved cases yet."));
  for (const item of cases) {
    const button = el("button", item.title, activeCase?.id === item.id ? "active" : "");
    button.type = "button";
    button.onclick = () => task("Loading saved case…", async () => showCase(await api(`/api/v1/cases/${item.id}`)));
    list.append(button);
  }
}

function field(form, title, name, choices) {
  const wrapper = el("div");
  const label = el("label", title);
  const input = choices ? el("select") : el("input");
  input.name = name;
  input.id = `${form.dataset.key}-${name}`;
  label.htmlFor = input.id;
  if (choices) {
    for (const value of ["", ...choices]) {
      const option = el("option", value ? pretty(value) : "Not specified");
      option.value = value;
      input.append(option);
    }
  } else input.maxLength = 100;
  wrapper.append(label, input);
  return wrapper;
}

function evidencePanel(resolution) {
  const panel = el("div", undefined, "evidence");
  panel.append(el("h3", "Source evidence"), el("p", "Citation checks verify the copied text, not the diagnosis.", "source-note"));
  if (!resolution.sources.length) panel.append(el("p", "No applicable procedure was selected."));
  for (const source of resolution.sources) {
    const details = el("details");
    details.append(el("summary", `[${source.citation_id}] ${source.title}`), el("p", `${source.doc_id} · Fictional provider procedure`, "source-note"));
    for (const quote of source.quotes) details.append(el("p", `${pretty(quote.field)}: ${quote.text}`));
    panel.append(details);
  }
  return panel;
}

function followupForm(issue) {
  const form = el("form", undefined, "answer-panel");
  form.dataset.key = `reply-${issue.issue_id}`;
  form.append(el("h3", issue.resolution.clarification_questions.length ? "Answer the clarification questions" : "Add a detail or correction"));
  const questions = el("ol", undefined, "questions");
  for (const question of issue.resolution.clarification_questions) questions.append(el("li", question));
  if (questions.childElementCount) form.append(questions);
  else form.append(el("p", "No customer questions are pending. Add new information if needed.", "source-note"));
  const label = el("label", `Your answer for issue ${issue.issue_id}`);
  const message = el("textarea");
  message.id = `${form.dataset.key}-message`;
  label.htmlFor = message.id;
  message.rows = 3;
  message.maxLength = 1000;
  message.placeholder = "Answer the questions above here. You can answer them together.";
  form.append(label, message);
  const details = el("details");
  details.append(el("summary", "Or record specific observations (optional)"));
  const fields = el("div", undefined, "fields");
  const products = new Set(issue.resolution.analysis.products.map(p => p.product));
  if (!issue.resolution.analysis.reported_facts.some(f => f.name === "equipment_condition" && ["damaged", "water_exposed"].includes(f.value)) && ["broadband", "home_wifi", "router"].some(p => products.has(p))) {
    fields.append(field(form, "Ethernet connection", "wired_connection", ["working", "failing"]), field(form, "Wireless devices affected", "wireless_devices", ["one", "all"]));
  }
  fields.append(field(form, "Current impact", "impact", ["complete_loss", "intermittent", "working"]));
  if (products.has("billing")) fields.append(field(form, "Payment status", "billing_status", ["pending", "settled"]), field(form, "Affected charge", "charge"));
  if (products.has("mobile")) fields.append(field(form, "Affected mobile services", "mobile_services", ["calls", "texts", "data", "several"]));
  if (products.has("iptv")) fields.append(field(form, "TV symptom", "tv_symptom", ["no_picture", "error", "buffering"]));
  for (const [name, title] of [["area", "Area"], ["started", "When it started"]]) fields.append(field(form, title, name));
  details.append(fields);
  const submit = el("button", "Send answer & update response", "primary");
  submit.type = "submit";
  form.append(details, submit);
  form.onsubmit = event => {
    event.preventDefault();
    const observations = Object.fromEntries([...new FormData(form)].filter(([,value]) => value.trim()));
    if (!message.value.trim() && !Object.keys(observations).length) {message.setCustomValidity("Add a reply or observation."); message.reportValidity(); return;}
    task("Updating the resolution with your answer…", async () => showCase(await api(`/api/v1/cases/${activeCase.id}/followups`, {expected_revision: activeCase.revision, turn: {issue_id: issue.issue_id, message: message.value, observations}}), issue.issue_id), "Response updated. Your answer is saved below.");
  };
  form.oninput = () => message.setCustomValidity("");
  return form;
}

function reviewForm(issue, draft) {
  const form = el("form", undefined, "review-form subsection");
  form.dataset.key = `review-${issue.issue_id}`;
  form.append(el("h3", "Agent decision"));
  const fields = el("div", undefined, "fields");
  fields.append(field(form, "Review decision", "action", ["accept", "edit", "reject"]), field(form, "Reported outcome", "outcome", ["pending", "resolved", "unresolved", "escalated"]));
  fields.querySelector('[name="action"]').required = true;
  fields.querySelector('[name="outcome"]').value = "pending";
  form.append(fields);
  for (const [name, title] of [["notes", "Review notes (required for edit or reject)"], ["outcome_notes", "Outcome evidence (required for a reported outcome)"]]) {
    const label = el("label", title);
    const input = el("textarea");
    input.rows = 2; input.name = name; input.maxLength = 2000; input.id = `${form.dataset.key}-${name}`; label.htmlFor = input.id;
    form.append(label, input);
  }
  form.append(el("small", "Accept preserves the original draft. Edits are saved separately and are not automatically citation-validated."));
  const save = el("button", "Save review", "primary"); save.type = "submit"; form.append(el("br"), save);
  form.onsubmit = event => {
    event.preventDefault();
    const values = Object.fromEntries(new FormData(form));
    if (values.action === "accept" && draft.value !== issue.resolution.draft) {
      byId("status").hidden = false; byId("status").className = "error"; byId("status").textContent = "The draft was edited. Choose Edit to save those changes, or restore the original before accepting."; return;
    }
    if (values.action === "edit") values.edited_draft = draft.value;
    task("Saving agent review…", async () => showCase(await api(`/api/v1/cases/${activeCase.id}/reviews`, {...values, expected_revision: activeCase.revision, issue_id: issue.issue_id})));
  };
  return form;
}

function renderIssue(issue, updatedIssue) {
  const resolution = issue.resolution;
  const panel = el("article", undefined, "panel issue-panel");
  const top = el("div", undefined, "issue-top");
  top.append(el("h2", `Issue ${issue.issue_id}`));
  top.append(el("span", `${resolution.decision.priority} priority`, "badge"));
  panel.append(top);
  const complaint = el("details", undefined, "complaint-detail");
  complaint.append(el("summary", "Original complaint"), el("p", issue.complaint));
  panel.append(complaint);
  const updated = updatedIssue === issue.issue_id;
  const response = el("section", undefined, `response-panel${updated ? " response-updated" : ""}`);
  response.id = `response-${issue.issue_id}`;
  response.tabIndex = -1;
  response.append(el("div", updated ? "UPDATED RESPONSE" : "RESOLUTION PLAN", "eyebrow"));
  const plan = resolution.language_plan || resolution.customer_plan;
  response.append(el("h3", plan?.title || "Resolution for agent review"));
  if (plan) {
    response.append(el("p", resolution.language_summary || plan.summary, "response-text"));
    const steps = el("ol", undefined, "resolution-steps");
    for (const step of plan.steps) steps.append(el("li", step));
    response.append(steps);
    if (plan.note) response.append(el("p", plan.note, "source-note"));
  } else response.append(el("p", "This saved ticket uses an earlier response format. Add current details to regenerate a resolution plan, or inspect its original draft in Agent tools.", "response-text"));
  if (resolution.acknowledged_actions.length) response.append(el("p", `Already attempted: ${resolution.acknowledged_actions.join("; ")}`, "attempts"));
  if (resolution.contact_status === "unverified") response.append(el("p", "The customer asked to speak with support. Arrange a human follow-up through the support desk; this demo does not provide telephone numbers.", "notice"));
  panel.append(response);
  if (resolution.language_status === "fallback" || resolution.analysis.language_method === "rules_fallback") response.append(el("p", "Language assistance was unavailable for part of this request. Review the evidence-controlled plan and extracted observations.", "source-note"));
  const reply = followupForm(issue);
  if (resolution.clarification_questions.length) panel.append(reply);
  else {
    const extra = el("details", undefined, "optional-reply");
    extra.append(el("summary", "Add information or correct this ticket"), reply);
    panel.append(extra);
  }
  const turns = activeCase.request.turns.filter(t => t.issue_id === issue.issue_id);
  if (turns.length) {
    const history = el("details", undefined, "answer-history");
    history.append(el("summary", `Your previous answers (${turns.length})`));
    for (const turn of turns) {
      const entry = el("div", undefined, "answer-entry");
      if (turn.message) entry.append(el("p", turn.message));
      for (const [key, value] of Object.entries(turn.observations)) if (value) entry.append(el("p", `${pretty(key)}: ${value}`));
      history.append(entry);
    }
    panel.append(history);
  }
  const tools = el("details", undefined, "review-tools");
  tools.append(el("summary", "Agent tools · evidence, draft and review"));
  tools.append(el("p", `Next action: ${pretty(resolution.decision.action)} · Category: ${resolution.analysis.category ? pretty(resolution.analysis.category) : "uncertain"} · Citation check: ${resolution.validation.status}`, "source-note"));
  if (resolution.language_status) tools.append(el("p", `Language draft: ${pretty(resolution.language_status)}${resolution.language_provider ? ` · ${pretty(resolution.language_provider)}` : ""} · Wording check: ${pretty(resolution.faithfulness_status || "not_run")}`, "source-note"));
  const latest = [...activeCase.reviews].reverse().find(r => r.generation === activeCase.generation && r.review.issue_id === issue.issue_id);
  if (latest) tools.append(el("p", `Latest review: ${pretty(latest.review.action)} · Outcome: ${latest.review.outcome}`, "notice"));
  const columns = el("div", undefined, "columns");
  if (resolution.language_draft) {
    const generated = el("details");
    generated.append(el("summary", "Language draft · review wording before use"), el("p", resolution.language_draft, "response-text"));
    tools.append(generated);
  }
  for (const historical of resolution.historical_cases || []) {
    const entry = el("details");
    entry.append(el("summary", `[${historical.citation_id}] ${historical.title} · ${pretty(historical.outcome_status)}`), el("p", historical.resolution));
    tools.append(entry);
  }
  const left = el("div");
  const label = el("label", "Internal draft for agent review");
  const draft = el("textarea", undefined, "draft");
  draft.id = `draft-${issue.issue_id}`;
  label.htmlFor = draft.id;
  draft.maxLength = 16000;
  draft.value = latest?.review.action === "edit" ? latest.reviewed_text : resolution.draft;
  const restore = el("button", "Restore original draft", "secondary");
  restore.type = "button";
  restore.onclick = () => { draft.value = resolution.draft; };
  const download = el("a", "Download handoff", "secondary download");
  download.href = `/api/v1/cases/${activeCase.id}/handoff/${issue.issue_id}`;
  download.download = `handoff-${activeCase.id}-${issue.issue_id}.json`;
  const review = reviewForm(issue, draft);
  if (resolution.language_draft) {
    const useGenerated = el("button", "Review language draft", "secondary");
    useGenerated.type = "button";
    useGenerated.onclick = () => {
      draft.value = resolution.language_draft;
      review.querySelector('[name="action"]').value = "edit";
      draft.focus();
    };
    left.append(useGenerated);
  }
  left.append(label, draft, restore, review, download);
  columns.append(left, evidencePanel(resolution));
  tools.append(columns);
  panel.append(tools);
  return panel;
}

async function showCase(record, updatedIssue = null) {
  activeCase = record;
  byId("empty").hidden = true; byId("case").hidden = false;
  byId("case-title").textContent = `Case ${record.id.slice(0, 8)} · ${record.response.issues.length} issue${record.response.issues.length === 1 ? "" : "s"}`;
  byId("case-meta").textContent = `Revision ${record.revision} · Updated ${new Date(record.updated_at).toLocaleString()}`;
  byId("issues").replaceChildren(...record.response.issues.map(issue => renderIssue(issue, updatedIssue)));
  const history = byId("history"); history.replaceChildren();
  if (!record.reviews.length) history.append(el("p", "No reviews recorded yet.", "audit-entry"));
  for (const event of [...record.reviews].reverse()) {
    const entry = el("div", undefined, "audit-entry");
    entry.append(el("strong", `Issue ${event.review.issue_id} · ${pretty(event.review.action)} · ${event.generation === record.generation ? "Current draft" : "Earlier draft"}`), el("p", `${new Date(event.created_at).toLocaleString()} · Outcome: ${event.review.outcome}`), el("p", event.review.notes), el("p", event.review.outcome_notes));
    if (event.reviewed_text) {const detail = el("details"); detail.append(el("summary", "Reviewed text"), el("p", event.reviewed_text)); entry.append(detail);}
    history.append(entry);
  }
  await refreshRecent();
  if (updatedIssue !== null) {
    const response = byId(`response-${updatedIssue}`);
    response.focus({preventScroll: true});
    response.scrollIntoView({behavior: "smooth", block: "start"});
  } else byId("case").scrollIntoView({behavior: "smooth", block: "start"});
}

byId("create-form").onsubmit = event => {event.preventDefault(); task("Analyzing complaint… The first request loads the local models and can take a minute.", async () => showCase(await api("/api/v1/cases", {query: byId("complaint").value})));};
byId("reload").onclick = () => {if (activeCase) task("Reloading saved case…", async () => showCase(await api(`/api/v1/cases/${activeCase.id}`)));};
refreshRecent().catch(() => {byId("recent").textContent = "Unable to load saved cases. Check the server and local storage.";});

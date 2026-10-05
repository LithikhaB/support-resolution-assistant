// Run with node --test tests/web/agent_console.test.cjs (no browser packages required).
const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

function harness(overrides = {}, savedFields = {}) {
  const ids = new Map();
  class Element {
    constructor(tag) { this.tag = tag; this.children = []; }
    set id(value) { this._id = value; ids.set(value, this); }
    append(...nodes) { this.children.push(...nodes); }
    replaceChildren(...nodes) { this.children = nodes; }
    focus() {}
    scrollIntoView() {}
    setCustomValidity(message) { this.validationMessage = message; }
    reportValidity() {}
  }
  for (const id of ['create-form','new-ticket','status','intake','issues','complaint']) {
    const node = new Element('div'); node.id = id;
  }
  const issue = {
    issue_id: 1, complaint: 'Broadband drops. Restarted twice.',
    resolution: {
      decision: {action:'clarify',priority:'high',target:'support_agent',reasons:['missing_observations']},
      analysis: {category:null,candidates:[{category:'intermittent_broadband'},{category:'wifi_connectivity'}],category_evidence:[],
        products:[{product:'broadband',text:'Broadband'}],severity:{value:'high',evidence:[{text:'costing me'}]},
        sentiment:{value:'frustrated',evidence:[{text:'costing me'}]},actions:[{status:'attempted',text:'Restarted twice'}]},
      customer_plan:{title:'Initial checks',summary:'Record the wired observation.',steps:['Check provider monitoring [S1]'],note:''},
      language_status:'fallback',validation:{status:'passed',scope:'Exact sources only',issues:[]},
      sources:[],historical_cases:[],clarification_questions:['During a drop, does Ethernet also lose internet?']
    }
  };
  Object.assign(issue.resolution, overrides);
  let request;
  const context = vm.createContext({document:{createElement: tag => new Element(tag),getElementById: id => ids.get(id),querySelectorAll:()=>[]},
    fetch:async (url,opts) => {request=JSON.parse(opts.body);return {ok:true,json:async()=>({issues:[issue],...savedFields})};},issue,savedFields});
  vm.runInContext(fs.readFileSync('app/web/app.js','utf8'),context);
  vm.runInContext("conversation = {query:issue.complaint, turns:[],...savedFields}; panel = renderIssue(issue)",context);
  function flatten(node) { return [node,...node.children.flatMap(flatten)]; }
  return {context, ids, nodes:flatten(context.panel),get request(){return request;}};
}

test('console surfaces provisional categories, decision, quoted attempts and honest fallback', () => {
  const h = harness();
  const text = h.nodes.map(n=>n.textContent||'').join('\n');
  for (const expected of ['clarify','high priority','Target: support agent','intermittent broadband / wifi connectivity','Restarted twice','Local deterministic plan','Citation validation: passed']) assert.ok(text.includes(expected),expected);
  assert.ok(!text.includes('needs more detail'));
  assert.ok(h.nodes.some(n=>n.title==='“costing me”'));
  const linked = harness({
    sources:[{citation_id:'S1',doc_id:'syn_kb_BB01',title:'Optical loss',authority:'fictional_provider_policy',quotes:[]}],
    historical_cases:[{citation_id:'T1',doc_id:'syn_ticket_BB01_01',title:'Resolved optical loss',relationship:'linked_procedure',outcome_status:'simulated_resolved',resolution:'Conditional technician repair.'}],
    customer_plan:{title:'Checks',summary:'Review the optical test.',steps:['Verify the optical signal [S1] [T1]'],note:''}
  });
  assert.ok(linked.nodes.some(n=>n.href==='#source-1-S1'));
  assert.ok(linked.nodes.some(n=>n.href==='#source-1-T1'));
  assert.ok(linked.ids.has('source-1-T1'));
  assert.ok(linked.nodes.some(n=>n.textContent?.includes('Ticket ID: syn_ticket_BB01_01')));
  const irrelevant = harness({
    analysis:{scope_status:'unsupported'},
    customer_plan:{title:'This request is irrelevant here',summary:'Please paste the customer complaint.',steps:[],note:'Telecom complaints only.'}
  });
  assert.ok(irrelevant.nodes.some(n=>n.textContent==='This request is irrelevant here'));
  assert.ok(!irrelevant.nodes.some(n=>n.tag==='textarea' || n.tag==='ol'));
});

test('quick answer sends a structured observation attached to the selected issue', async () => {
  const h = harness();
  h.nodes.find(n=>n.textContent==='Wired also drops').onclick();
  await new Promise(resolve=>setImmediate(resolve));
  assert.deepEqual(h.request.turns,[{issue_id:1,message:'Wired also drops',observations:{wired_connection:'failing'}}]);
  assert.equal(h.request.query,'Broadband drops. Restarted twice.');
  const saved = harness({}, {conversation_id:'9a05918c-6947-47a5-b980-9a0d579e54aa',revision:2});
  saved.nodes.find(n=>n.textContent==='Wired also drops').onclick();
  await new Promise(resolve=>setImmediate(resolve));
  assert.equal(saved.request.conversation_id,'9a05918c-6947-47a5-b980-9a0d579e54aa');
  assert.equal(saved.request.revision,2);
  const review = harness({sources:[{citation_id:'S1',doc_id:'syn_kb_BB01',title:'Optical loss',authority:'fictional_provider_policy',quotes:[]}]}, {conversation_id:'9a05918c-6947-47a5-b980-9a0d579e54aa',revision:2});
  assert.ok(review.nodes.some(n=>n.textContent==='Record a reviewed simulated resolution'));
  const form = review.nodes.find(n=>n.tag==='form' && n.children.some(n=>n.textContent==='Save reviewed outcome'));
  await form.onsubmit({preventDefault(){}});
  assert.equal(review.request,undefined);
  review.nodes.find(n=>n.type==='checkbox').checked=true;
  review.ids.get('outcome-1').value='Simulated repair outcome reviewed for this test.';
  await form.onsubmit({preventDefault(){}});
  assert.equal(review.request.confirmation,'simulated_resolution_reviewed');
  assert.equal(review.request.revision,2);
});

test('quick answer respects the eight-turn API budget', () => {
  const h = harness();
  vm.runInContext('conversation.turns = Array.from({length:8},()=>({issue_id:1,message:"answer"}))',h.context);
  h.nodes.find(n=>n.textContent==='Wired also drops').onclick();
  assert.equal(h.request,undefined);
  assert.match(h.ids.get('answer-1').validationMessage,/Eight replies/);
});

test('LLM introduction with unchanged steps remains labelled as local steps', () => {
  const h = harness({language_status:'generated_for_review',language_provider:'groq',language_model:'test-model',
    language_plan:{title:'Initial checks',summary:'Customer reports drops.',steps:['Check provider monitoring [S1]'],note:''}});
  assert.ok(h.nodes.some(n => n.textContent?.includes('LLM introduction · local steps')));
});

test('changed generated steps are labelled as an LLM plan', () => {
  const h = harness({language_status:'generated_for_review',language_provider:'groq',language_model:'test-model',
    language_plan:{title:'Initial checks',summary:'Customer reports drops.',steps:['Verify with authorized monitoring [S1]'],note:''}});
  assert.ok(h.nodes.some(n => n.textContent?.startsWith('LLM-assisted plan · protected repair steps · groq')));
});

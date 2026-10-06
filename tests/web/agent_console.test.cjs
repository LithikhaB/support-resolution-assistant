// Run with node --test tests/web/agent_console.test.cjs (no browser packages required).
const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

function harness(overrides = {}, savedFields = {}, options = {}) {
  const calls = [];
  const listeners = {};
  let storageReads = 0;
  let savedHistory = [...(options.history || [])];
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
    setAttribute(name, value) { this[name] = value; }
  }
  for (const id of ['create-form','new-ticket','status','intake','issues','complaint','workspace','recent-conversations','recent-list','history-drawer','history-backdrop','close-history','history-select-all','delete-selected','history-confirmation','history-confirmation-text','history-status','confirm-delete','cancel-delete']) {
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
  const context = vm.createContext({document:{createElement: tag => new Element(tag),getElementById: id => ids.get(id),querySelectorAll:()=>[],addEventListener:(name,handler)=>listeners[name]=handler},
    localStorage:{getItem:()=>{storageReads++;return options.storedId;},removeItem:()=>{}},
    fetch:async (url,opts={}) => {
      calls.push({url,method:opts.method||'GET'});
      if (opts.body) request=JSON.parse(opts.body);
      if (url==='/api/v1/conversations') return {ok:true,json:async()=>({conversations:savedHistory})};
      if (opts.method==='DELETE') { if(options.deleteOk!==false) savedHistory=savedHistory.filter(item=>url!==`/api/v1/conversations/${item.conversation_id}`); return {ok:options.deleteOk!==false,json:async()=>({deleted:true})}; }
      if (url.startsWith('/api/v1/conversations/') && !opts.method) return {ok:true,json:async()=>options.saved};
      return {ok:true,json:async()=>({issues:[issue],...savedFields})};
    },issue,savedFields});
  vm.runInContext(fs.readFileSync('app/web/app.js','utf8'),context);
  vm.runInContext("conversation = {query:issue.complaint, turns:[],...savedFields}; panel = renderIssue(issue)",context);
  function flatten(node) { return [node,...node.children.flatMap(flatten)]; }
  return {context, ids, nodes:flatten(context.panel),calls,listeners,get storageReads(){return storageReads;},get request(){return request;}};
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
  const savedView = harness({sources:[{citation_id:'S1',doc_id:'syn_kb_BB01',title:'Optical loss',authority:'fictional_provider_policy',quotes:[]}]}, {conversation_id:'9a05918c-6947-47a5-b980-9a0d579e54aa',revision:2});
  assert.ok(!savedView.nodes.some(n=>n.textContent==='Record a reviewed simulated resolution'));
  assert.ok(!savedView.nodes.some(n=>n.textContent==='Save reviewed outcome'));
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

test('neutral default replaces unknown sentiment without claiming quoted evidence', () => {
  for (const sentiment of [
    {value:'unknown',rule:'no_explicit_tone',evidence:[]},
    {value:'unknown',rule:'quoted_language',evidence:[{text:'Broadband drops'}]},
    {value:'neutral',rule:'neutral_default',evidence:[]}
  ]) {
    const h=harness();
    h.context.issue.resolution.analysis.sentiment=sentiment;
    vm.runInContext('panel = renderIssue(issue)',h.context);
    function flatten(node) { return [node,...node.children.flatMap(flatten)]; }
    const nodes=flatten(h.context.panel);
    const chip=nodes.find(n=>n.textContent==='Sentiment: Neutral');
    assert.ok(chip);
    assert.equal(chip.title,'Neutral is the default; no explicit emotional tone was detected.');
    assert.ok(!nodes.some(n=>n.textContent==='Sentiment: unknown'));
  }
});


test('startup opens blank intake and never restores a saved conversation automatically', () => {
  const h=harness({}, {}, {storedId:'11111111-1111-4111-8111-111111111111'});
  assert.equal(h.storageReads,0);
  assert.deepEqual(h.calls,[]);
  assert.equal(h.ids.get('complaint').value,'');
  assert.equal(h.ids.get('intake').hidden,false);
  assert.equal(h.ids.get('history-drawer').hidden,true);
});

test('instructions and citation badges are separated and explain KB versus historical tickets', () => {
  const h=harness({customer_plan:{title:'Checks',summary:'Review.',steps:['Verify optical signal [S1] [T1] [S1]'],note:''}});
  const instruction=h.nodes.find(n=>n.className==='instruction-text');
  assert.equal(instruction.textContent,'Verify optical signal');
  assert.equal(h.nodes.filter(n=>n.className==='citation').length,2);
  assert.ok(h.nodes.some(n=>n.textContent==='[S1] · KB'));
  assert.ok(h.nodes.some(n=>n.textContent==='[T1] · Ticket'));
  assert.ok(h.nodes.some(n=>n.textContent==='S1, S2… = knowledge-base articles (KB).'));
  assert.ok(h.nodes.some(n=>n.textContent==='T1, T2… = historical resolved tickets. Demo outcomes are simulated.'));
});

test('history opens a right drawer; delete acts only on selected rows after confirmation', async () => {
  const history=[{conversation_id:'one',query:'First saved complaint',updated_at:'2026-10-06T00:00:00Z',revision:1},
    {conversation_id:'two',query:'Second saved complaint',updated_at:'2026-10-06T01:00:00Z',revision:2}];
  const h=harness({}, {}, {history});
  await h.ids.get('recent-conversations').onclick();
  assert.equal(h.ids.get('history-drawer').hidden,false);
  assert.equal(h.ids.get('workspace').inert,true);
  const select=h.ids.get('recent-list').children[0].children[0].children[0];
  select.checked=true;select.onchange();
  assert.equal(h.ids.get('delete-selected').textContent,'Delete selected (1)');
  h.ids.get('delete-selected').onclick();
  assert.equal(h.ids.get('history-confirmation').hidden,false);
  assert.equal(h.calls.filter(call=>call.method==='DELETE').length,0);
  await h.ids.get('confirm-delete').onclick();
  assert.deepEqual(h.calls.filter(call=>call.method==='DELETE'),[{url:'/api/v1/conversations/one',method:'DELETE'}]);
  assert.equal(h.ids.get('recent-list').children.length,1);
  assert.equal(h.ids.get('history-status').textContent,'Deleted 1 conversation.');
  h.listeners.keydown({key:'Escape'});
  assert.equal(h.ids.get('history-drawer').hidden,true);
  assert.equal(h.ids.get('workspace').inert,false);
});

test('opening a saved conversation is explicit and preserves its revision', async () => {
  const history=[{conversation_id:'saved-id',query:'Saved complaint',updated_at:'2026-10-06T00:00:00Z',revision:3}];
  const saved={request:{query:'Saved complaint',turns:[]},conversation_id:'saved-id',revision:3};
  const h=harness({}, {}, {history,saved});
  await h.ids.get('recent-conversations').onclick();
  await h.ids.get('recent-list').children[0].children[1].children[0].onclick();
  assert.equal(h.request.conversation_id,'saved-id');
  assert.equal(h.request.revision,3);
  assert.equal(h.ids.get('history-drawer').hidden,true);
});


test('overlapping prior-action fragments do not duplicate the already-tried chips', () => {
  const h=harness();
  h.context.issue.resolution.analysis.actions=[
    {status:'attempted',text:'I restarted the router twice and checked every cable'},
    {status:'attempted',text:'checked every cable'}
  ];
  vm.runInContext('panel=renderIssue(issue)',h.context);
  function flatten(node) { return [node,...node.children.flatMap(flatten)]; }
  const labels=flatten(h.context.panel).filter(n=>n.tag==='summary' && n.textContent?.startsWith('Already tried:'));
  assert.equal(labels.length,1);
  assert.ok(labels[0].textContent.includes('twice and checked every cable'));
});

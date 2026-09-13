'use strict';

const byId = id => document.getElementById(id);
const clamp = (value, low, high) => Math.max(low, Math.min(high, value));
const cellKey = cell => cell.join(',');
const labels = {look_around: 'Surveying the workspace', approach_box: 'Approaching a box', pickup: 'Picking up a box', move_to_build: 'Moving to the build spot', place: 'Placing a box', select_site: 'Choosing the build spot', observe: 'Refreshing the scene', done: 'Verifying the finished shape', stop: 'Stopping the build'};
const phases = {INVENTORY: 'Inventory', SELECT_SITE: 'Choose site', BUILD: 'Assembly', FINAL_VERIFY: 'Verification'};

class VoxelView {
  constructor(canvas, onSelect) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.onSelect = onSelect || (() => {});
    this.blocks = [];
    this.placed = new Set();
    this.active = null;
    this.selected = null;
    this.identity = null;
    this.mode = 'design';
    this.yaw = -.65;
    this.pitch = .58;
    this.zoom = 1;
    this.faces = [];
    this.drag = null;
    this.pending = false;
    this.center = [0, 0, 0];
    this.extents = [3, 2, 3];
    new ResizeObserver(() => this.invalidate()).observe(canvas.parentElement);
    canvas.addEventListener('pointerdown', e => {
      this.drag = {id: e.pointerId, x: e.clientX, y: e.clientY, distance: 0};
      canvas.setPointerCapture(e.pointerId);
      canvas.classList.add('dragging');
      canvas.focus({preventScroll: true});
    });
    canvas.addEventListener('pointermove', e => {
      if (!this.drag || this.drag.id !== e.pointerId) return;
      const dx = e.clientX - this.drag.x, dy = e.clientY - this.drag.y;
      this.drag.distance += Math.abs(dx) + Math.abs(dy);
      this.yaw += dx * .008;
      this.pitch = clamp(this.pitch + dy * .006, .12, 1.48);
      this.drag.x = e.clientX;
      this.drag.y = e.clientY;
      this.invalidate();
    });
    canvas.addEventListener('pointerup', e => {
      if (this.drag && this.drag.distance < 7) this.pick(e);
      this.drag = null;
      canvas.classList.remove('dragging');
    });
    canvas.addEventListener('pointercancel', () => { this.drag = null; canvas.classList.remove('dragging'); });
    canvas.addEventListener('wheel', e => {
      e.preventDefault();
      this.zoom = clamp(this.zoom * Math.exp(-e.deltaY * .0015), .45, 3.5);
      this.invalidate();
    }, {passive: false});
    canvas.addEventListener('keydown', e => {
      const actions = {ArrowLeft: () => this.yaw -= .12, ArrowRight: () => this.yaw += .12,
        ArrowUp: () => this.pitch = clamp(this.pitch - .1, .12, 1.48), ArrowDown: () => this.pitch = clamp(this.pitch + .1, .12, 1.48),
        '+': () => this.zoom = clamp(this.zoom * 1.15, .45, 3.5), '=': () => this.zoom = clamp(this.zoom * 1.15, .45, 3.5),
        '-': () => this.zoom = clamp(this.zoom / 1.15, .45, 3.5), Home: () => this.reset()};
      if (actions[e.key]) { e.preventDefault(); actions[e.key](); this.invalidate(); }
    });
  }
  reset(angle) {
    this.yaw = angle === 'top' ? 0 : -.65;
    this.pitch = angle === 'top' ? 1.48 : .58;
    this.zoom = 1;
    this.invalidate();
  }
  set(design, mode, placed, active) {
    const identity = design ? design.id : 'empty';
    if (identity !== this.identity) {
      this.identity = identity;
      this.selected = null;
      this.reset();
    }
    this.blocks = design ? design.blocks : [];
    this.extents = design ? design.size : [3, 2, 3];
    this.center = [this.extents[0] / 2, this.extents[1] / 2 - .15, this.extents[2] / 2];
    this.mode = mode;
    this.placed = new Set((placed || []).map(p => cellKey(p.cell)));
    this.active = active ? cellKey(active) : null;
    this.invalidate();
  }
  invalidate() {
    if (this.pending) return;
    this.pending = true;
    requestAnimationFrame(() => { this.pending = false; this.draw(); });
  }
  project(p) {
    const x = p[0] - this.center[0], y = p[1] - this.center[1], z = p[2] - this.center[2];
    const cy = Math.cos(this.yaw), sy = Math.sin(this.yaw), cp = Math.cos(this.pitch), sp = Math.sin(this.pitch);
    const rx = cy * x + sy * z, rz = -sy * x + cy * z;
    const ry = cp * y - sp * rz, depth = sp * y + cp * rz;
    const factor = this.distance / Math.max(this.distance - depth, .5);
    return [this.width / 2 + rx * this.scale * factor, this.height / 2 - ry * this.scale * factor, depth];
  }
  path(points) {
    const ctx = this.ctx;
    ctx.beginPath();
    points.forEach((p, index) => index ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1]));
    ctx.closePath();
  }
  line(a, b, color, width) {
    const ctx = this.ctx, p = this.project(a), q = this.project(b);
    ctx.beginPath(); ctx.moveTo(p[0], p[1]); ctx.lineTo(q[0], q[1]);
    ctx.strokeStyle = color; ctx.lineWidth = width || 1; ctx.stroke();
  }
  draw() {
    const bounds = this.canvas.getBoundingClientRect();
    if (bounds.width < 1 || bounds.height < 1 || !this.ctx) return;
    const ratio = Math.min(window.devicePixelRatio || 1, 2);
    this.width = bounds.width; this.height = bounds.height;
    this.canvas.width = Math.round(bounds.width * ratio); this.canvas.height = Math.round(bounds.height * ratio);
    const ctx = this.ctx;
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    ctx.fillStyle = '#f7f3ee'; ctx.fillRect(0, 0, this.width, this.height);
    const span = Math.max(...this.extents, 2);
    this.scale = Math.min(this.width, this.height) * .68 / (span + 1.6) * this.zoom;
    this.distance = span * 4 + 12;
    const radius = Math.min(Math.ceil(span / 2) + 4, 40);
    const cx = this.center[0], cz = this.center[2];
    for (let i = -radius; i <= radius; i++) {
      this.line([cx + i, -.035, cz - radius], [cx + i, -.035, cz + radius], i === 0 ? '#d8c3b9' : '#e8dfd7', .8);
      this.line([cx - radius, -.035, cz + i], [cx + radius, -.035, cz + i], i === 0 ? '#d8c3b9' : '#e8dfd7', .8);
    }
    if (this.blocks.length) {
      const footprint = [[0, -.02, 0], [this.extents[0], -.02, 0], [this.extents[0], -.02, this.extents[2]], [0, -.02, this.extents[2]]];
      this.path(footprint.map(p => this.project(p))); ctx.fillStyle = '#b8494608'; ctx.fill();
      ctx.strokeStyle = '#b987805e'; ctx.setLineDash([4, 5]); ctx.stroke(); ctx.setLineDash([]);
    }
    const definitions = [
      {v: [0, 1, 2, 3], color: '#ad793f'}, {v: [4, 7, 6, 5], color: '#c08a4e'},
      {v: [0, 4, 5, 1], color: '#80572f'}, {v: [3, 2, 6, 7], color: '#e3b778'},
      {v: [0, 3, 7, 4], color: '#ba8246'}, {v: [1, 5, 6, 2], color: '#ce9857'}
    ];
    const faces = [];
    for (const block of this.blocks) {
      const x = block.x + .025, y = block.y + .015, z = block.z + .025, s = .95, h = .97;
      const vertices = [[x,y,z],[x+s,y,z],[x+s,y+h,z],[x,y+h,z],[x,y,z+s],[x+s,y,z+s],[x+s,y+h,z+s],[x,y+h,z+s]].map(p => this.project(p));
      const key = cellKey([block.x, block.y, block.z]);
      for (let i = 0; i < definitions.length; i++) {
        const def = definitions[i], points = def.v.map(index => vertices[index]);
        const edge1 = [points[1][0] - points[0][0], points[1][1] - points[0][1]];
        const edge2 = [points[2][0] - points[0][0], points[2][1] - points[0][1]];
        if (edge1[0] * edge2[1] - edge1[1] * edge2[0] < 0) continue;
        faces.push({points, depth: points.reduce((sum, p) => sum + p[2], 0) / 4, color: def.color, block, key, face: i,
          ghost: this.mode === 'build' && !this.placed.has(key), placed: this.placed.has(key), active: key === this.active || key === this.selected});
      }
    }
    faces.sort((a, b) => a.depth - b.depth);
    this.faces = faces;
    for (const face of faces) {
      this.path(face.points);
      ctx.fillStyle = face.ghost ? (face.active ? '#b8494630' : '#aa92821a') : face.color;
      ctx.fill();
      ctx.strokeStyle = face.active ? '#b84946' : face.placed ? '#397253' : face.ghost ? '#9f897a' : '#533c2690';
      ctx.lineWidth = face.active ? 2 : 1;
      ctx.stroke();
      if (!face.ghost && (face.face === 3 || face.face === 0 || face.face === 1)) {
        const p = face.points;
        const mix = (a,b,t) => [a[0] + (b[0]-a[0])*t, a[1] + (b[1]-a[1])*t];
        this.path([mix(p[0],p[1],.4),mix(p[0],p[1],.6),mix(p[3],p[2],.6),mix(p[3],p[2],.4)]);
        ctx.fillStyle = face.face === 3 ? '#f3dbab65' : '#e7be7e5e'; ctx.fill();
      }
    }
    ctx.font = '9px ui-monospace, monospace'; ctx.fillStyle = '#79675c';
    ctx.fillText(this.blocks.length ? 'RELATIVE ARRANGEMENT / 1 BLOCK = 1 BOX' : 'AWAITING MINECRAFT INPUT', 17, this.height - 15);
  }
  pick(event) {
    const rect = this.canvas.getBoundingClientRect(), x = event.clientX - rect.left, y = event.clientY - rect.top;
    const contains = points => {
      let inside = false;
      for (let i = 0, j = points.length - 1; i < points.length; j = i++) {
        const a = points[i], b = points[j];
        if ((a[1] > y) !== (b[1] > y) && x < (b[0]-a[0]) * (y-a[1]) / (b[1]-a[1]) + a[0]) inside = !inside;
      }
      return inside;
    };
    const face = [...this.faces].reverse().find(f => contains(f.points));
    this.selected = face ? face.key : null;
    this.onSelect(face ? face.block : null);
    this.invalidate();
  }
}

let state = null, submitting = false, connected = false, toastTimer = null, lastScene = {}, lastFeed = '';
const views = {
  main: new VoxelView(byId('main-canvas'), block => {
    byId('selection').hidden = !block;
    if (block) byId('selection').textContent = `CELL ${block.x}, ${block.y}, ${block.z} / ${block.kind.replace('minecraft:', '')}`;
  }),
  build: new VoxelView(byId('build-canvas')),
  complete: new VoxelView(byId('complete-canvas'))
};

function text(id, value) { const node = byId(id); if (node.textContent !== String(value)) node.textContent = value; }
function elapsed(start, end) {
  const seconds = Math.max(0, Math.floor(((end || Date.now() / 1000) - start)));
  return `${Math.floor(seconds / 60).toString().padStart(2, '0')}:${(seconds % 60).toString().padStart(2, '0')}`;
}
function notice(message) {
  clearTimeout(toastTimer);
  text('toast', message); byId('toast').hidden = false;
  toastTimer = setTimeout(() => { byId('toast').hidden = true; }, 7000);
}
async function command(path, data) {
  if (!state || submitting) return;
  submitting = true;
  render(state);
  try {
    const response = await fetch(path, {method: 'POST', headers: {'Content-Type': 'application/json', 'X-Crafter-Token': state.csrf}, body: JSON.stringify(data || {})});
    const body = await response.json();
    if (!response.ok) throw new Error(typeof body.detail === 'string' ? body.detail : 'Command could not be accepted.');
    const next = await fetch('/api/state', {cache: 'no-store'});
    if (next.ok) state = await next.json();
  } catch (error) { notice(error.message); }
  finally { submitting = false; if (state) render(state); }
}
function scene(name, design, mode, placed, active) {
  const signature = JSON.stringify([design ? design.id : null, mode, placed || [], active || null]);
  if (lastScene[name] !== signature) {
    lastScene[name] = signature;
    views[name].set(design, mode, placed, active);
    if (name === 'main') byId('selection').hidden = true;
  }
}
function eventRows(events) {
  const rows = [], pending = new Map();
  for (const event of events) {
    const d = event.data;
    if (event.type === 'llm_start') {
      const row = {id: d.call_id, at: event.at, kind: 'model', status: 'running', title: 'Choosing the next step', detail: 'Waiting for the model response.', data: d.input};
      rows.push(row); pending.set(d.call_id, row);
    } else if (event.type === 'llm_result') {
      let row = pending.get(d.call_id);
      if (!row) { row = {id: d.call_id, at: event.at, kind: 'model'}; rows.push(row); }
      Object.assign(row, {status: 'complete', title: labels[d.decision.operation] || d.decision.operation, detail: d.decision.reason, data: d.decision, duration: d.duration});
    } else if (event.type === 'llm_error') {
      let row = pending.get(d.call_id);
      if (!row) { row = {id: d.call_id, at: event.at, kind: 'model'}; rows.push(row); }
      Object.assign(row, {status: 'error', title: 'Model request failed', detail: d.error, data: null});
    } else if (event.type === 'tool_start') {
      const row = {id: d.request_id, at: event.at, kind: 'tool', status: 'running', title: `${d.operation}()`, detail: 'Simulating this action. No hardware will move.', data: d.arguments};
      rows.push(row); pending.set(d.request_id, row);
    } else if (event.type === 'tool_result') {
      const row = pending.get(d.request_id);
      if (row) { row.status = 'complete'; row.detail = 'Simulated success'; }
    } else if (event.type === 'placement_confirmed') {
      rows.push({id: String(event.id), at: event.at, kind: 'verified', status: 'complete', title: `Box ${d.box_id} placed`, detail: `Cell [${d.cell.join(', ')}] verified in the simulated scene.`});
    } else if (event.type === 'site_selected') {
      rows.push({id: String(event.id), at: event.at, kind: 'verified', status: 'complete', title: 'Build spot selected', detail: `${d.site_id} / simulated clear floor`});
    } else if (event.type === 'model_rejected') {
      rows.push({id: String(event.id), at: event.at, kind: 'model', status: 'error', title: 'Decision rejected', detail: `${d.error}. Retrying within the request budget.`});
    }
  }
  return rows;
}
function renderFeed(job) {
  const signature = `${job.id}:${job.events.length ? job.events[job.events.length - 1].id : 0}:${job.status}`;
  if (signature === lastFeed) return;
  lastFeed = signature;
  const feed = byId('activity-feed'), scroll = feed.scrollTop;
  const atBottom = feed.scrollHeight - feed.clientHeight - scroll < 70;
  const opened = new Set([...feed.querySelectorAll('details[open]')].map(d => d.dataset.key));
  const fragment = document.createDocumentFragment();
  for (const row of eventRows(job.events)) {
    if (job.status !== 'running' && row.status === 'running') row.status = 'stopped';
    const article = document.createElement('article'); article.className = `event ${row.kind} ${row.status}`;
    const dot = document.createElement('span'); dot.className = 'event-dot'; article.append(dot);
    const meta = document.createElement('div'); meta.className = 'event-meta';
    const kind = document.createElement('span'); kind.className = 'event-kind'; kind.textContent = row.kind === 'model' ? 'LLM DECISION' : row.kind === 'tool' ? 'TOOL CALL / MOCK' : 'SCENE UPDATE';
    const stamp = document.createElement('time'); stamp.textContent = elapsed(job.started_at, row.at);
    meta.append(kind, stamp); article.append(meta);
    const title = document.createElement('h3'); title.textContent = row.title; article.append(title);
    const detail = document.createElement('p'); detail.textContent = row.detail || ''; article.append(detail);
    if (row.status === 'running' || row.status === 'error' || row.duration !== undefined) {
      const status = document.createElement('div'); status.className = 'event-status';
      status.textContent = row.status === 'running' ? 'In progress…' : row.status === 'error' ? 'Not completed' : `${row.duration.toFixed(2)}s response`;
      article.append(status);
    }
    if (row.data) {
      const details = document.createElement('details'); details.dataset.key = row.id; details.open = opened.has(row.id);
      const summary = document.createElement('summary'); summary.textContent = row.kind === 'tool' ? 'Arguments' : 'Structured decision / context';
      const pre = document.createElement('pre'); pre.textContent = JSON.stringify(row.data, null, 2);
      details.append(summary, pre); article.append(details);
    }
    fragment.append(article);
  }
  if (!fragment.childNodes.length) {
    const empty = document.createElement('div'); empty.className = 'feed-empty'; empty.textContent = 'The agent’s next steps will appear here.'; fragment.append(empty);
  }
  feed.replaceChildren(fragment);
  feed.scrollTop = atBottom ? feed.scrollHeight : scroll;
}
function render(s) {
  const screen = s.view || 'main', design = s.design, job = s.job;
  for (const section of document.querySelectorAll('[data-screen]')) section.hidden = section.dataset.screen !== screen;
  const order = ['main', 'build', 'complete'];
  for (const step of document.querySelectorAll('[data-step]')) {
    step.classList.toggle('active', step.dataset.step === screen);
    step.classList.toggle('passed', order.indexOf(step.dataset.step) < order.indexOf(screen));
  }
  byId('workflow').hidden = screen === 'debug';
  // Builds really do simulate their tools; the debug console really does not. The badge has to
  // say which screen you are on, or it lies on one of them.
  text('hardware-badge', screen === 'debug' ? 'Real providers' : 'Simulated hardware');
  if (window.debugScreen) window.debugScreen.sync(s);
  const connection = byId('connection');
  connection.classList.toggle('offline', !connected);
  connection.lastChild.textContent = connected ? 'Panel online' : 'Disconnected';
  text('main-title', design ? 'A shape worth building.' : 'Your next build starts here.');
  byId('configuration-warning').hidden = s.llm_ready;
  byId('save-api-key').disabled = submitting || s.worker_busy;
  byId('waiting').hidden = !!design;
  byId('design-warning').hidden = !design || design.buildable;
  if (design && !design.buildable) text('design-warning', `Preview received, but this build cannot start: ${design.error}`);
  const enabled = !!(connected && design && design.buildable && s.llm_ready && !s.worker_busy && !submitting);
  byId('start-build').disabled = !enabled;
  byId('clear-blueprint').disabled = !connected || !design || submitting;
  text('start-hint', s.worker_busy && screen === 'main' ? 'Finishing the cancelled model request. You can start again shortly.' : !design ? 'Receive a design to get started.' : !s.llm_ready ? 'Configure your API key to enable real reasoning.' : !design.buildable ? 'Adjust the schematic and scan it again.' : 'Uses the model API. Physical actions are simulated.');
  byId('load-example').disabled = submitting;
  text('receiver-detail', s.receiver.listening ? `Minecraft receiver ready on TCP :${s.receiver.port} · latest design only` : s.receiver.error || 'Minecraft receiver is starting');
  document.querySelector('.receiver-port').textContent = `TCP :${s.receiver.port}`;
  const receiveLabel = document.querySelector('#main-screen .live-label');
  receiveLabel.lastChild.textContent = s.receiver.listening ? 'Listening for designs' : 'Receiver unavailable';
  byId('receive-notice').hidden = !s.notice;
  if (s.notice) text('receive-notice', s.notice);
  scene('main', design, 'design', [], null);
  if (job) {
    const total = job.design.count, placed = job.placed.length, percent = Math.round(placed / total * 100);
    const running = job.status === 'running';
    text('build-title', job.status === 'failed' ? 'The build needs attention.' : 'Making the shape happen.');
    byId('cancel-build').hidden = !running; byId('cancel-build').disabled = submitting;
    byId('failed-back').hidden = running; byId('failed-back').disabled = submitting;
    byId('build-error').hidden = !job.error;
    if (job.error) text('build-error', `${job.error} No robot hardware was involved. You can return to the design and try again.`);
    byId('new-design-notice').hidden = !design || job.design.id === design.id;
    text('reasoning-text', job.reasoning);
    text('phase-label', phases[job.phase] || job.phase);
    const rows = eventRows(job.events), latest = rows.length ? rows[rows.length - 1] : null;
    const thinking = running && latest && latest.kind === 'model' && latest.status === 'running';
    byId('thinking-indicator').classList.toggle('busy', !!thinking);
    text('thinking-label', thinking ? 'ASKING THE MODEL' : 'AGENT REASONING');
    text('progress-label', `${placed} / ${total} boxes placed`);
    text('progress-percent', `${percent}%`);
    byId('progress-fill').style.width = `${percent}%`;
    document.querySelector('.progress-track').setAttribute('aria-valuenow', String(percent));
    text('current-action', job.current_tool ? labels[job.current_tool] || job.current_tool : thinking ? 'Waiting for model response' : running ? 'Checking the next step' : job.status === 'completed' ? 'Build verified' : 'Build stopped');
    text('elapsed', elapsed(job.started_at, job.finished_at));
    text('llm-count', job.llm_calls); text('tool-count', job.tool_calls);
    byId('feed-live').hidden = !running;
    scene('build', job.design, 'build', job.placed, job.current_cell);
    scene('complete', job.design, 'complete', job.placed, null);
    renderFeed(job);
    text('complete-description', `Your ${total}-box shape is complete. Every placement has been verified in the simulated scene.`);
    text('complete-boxes', placed); text('complete-time', elapsed(job.started_at, job.finished_at)); text('complete-tools', job.tool_calls);
    text('complete-next', design && design.id !== job.design.id ? 'A newer Minecraft design is ready on the main screen.' : 'Ready for the next idea.');
    byId('back-main').disabled = submitting;
  }
  if (views[screen]) views[screen].invalidate();
}

for (const button of document.querySelectorAll('[data-angle]')) {
  button.addEventListener('click', () => {
    views[button.dataset.view].reset(button.dataset.angle);
    for (const sibling of button.parentElement.querySelectorAll('button')) sibling.classList.remove('selected');
    if (button.dataset.angle !== 'reset') button.classList.add('selected');
  });
}
byId('start-build').addEventListener('click', () => { if (state && state.design) command('/api/builds', {design_id: state.design.id}); });
byId('clear-blueprint').addEventListener('click', () => { if (state && state.design) command('/api/clear', {design_id: state.design.id}); });
byId('cancel-build').addEventListener('click', () => command('/api/cancel', {job_id: state.job.id}));
byId('failed-back').addEventListener('click', () => command('/api/main'));
byId('back-main').addEventListener('click', () => command('/api/main'));
byId('load-example').addEventListener('click', () => command('/api/example'));
byId('api-key-form').addEventListener('submit', async event => {
  event.preventDefault();
  const input = byId('api-key');
  const key = input.value.trim();
  input.value = '';
  await command('/api/key', {api_key: key});
});

async function poll() {
  try {
    const response = await fetch('/api/state', {cache: 'no-store'});
    if (!response.ok) throw new Error('Panel connection lost');
    const next = await response.json();
    connected = true;
    if (!state || next.csrf !== state.csrf || next.revision >= state.revision) state = next;
    render(state);
  } catch (error) {
    connected = false;
    if (state) render(state);
    else { byId('connection').classList.add('offline'); byId('connection').lastChild.textContent = 'Disconnected'; }
  } finally { setTimeout(poll, 350); }
}
poll();

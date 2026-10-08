// Deo's window. Python (deo/gui_web.py) queues events; this polls them and
// renders. Everything that leaves the page goes through window.pywebview.api.

const $ = (id) => document.getElementById(id);
const chat = $("chat");
const STATE_WORDS = {
  idle: "Idle",
  listening: "Listening",
  thinking: "Thinking",
  speaking: "Speaking",
  paused: "Paused",
  error: "Something went wrong",
};

let paused = false;
let currentState = "idle";
let streamEl = null;           // the Deo bubble currently being streamed into

// ---------------------------------------------------------------- markdown
function esc(s) {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}
function inline(s) {
  s = esc(s);
  s = s.replace(/`([^`]+)`/g, (_, c) => `<code>${c}</code>`);
  s = s.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  s = s.replace(/(^|[\s(])\*([^*\s][^*]*)\*/g, "$1<em>$2</em>");
  s = s.replace(/\[([^\]]+)\]\((https?:[^)\s]+)\)/g, '<a data-href="$2">$1</a>');
  s = s.replace(/(^|\s)(https?:\/\/[^\s<]+)/g, '$1<a data-href="$2">$2</a>');
  return s;
}
function markdown(text) {
  const out = [];
  const parts = text.split(/```/);
  parts.forEach((part, i) => {
    if (i % 2 === 1) {                                   // fenced code
      const nl = part.indexOf("\n");
      const lang = nl > -1 ? part.slice(0, nl).trim() : "";
      const code = nl > -1 ? part.slice(nl + 1) : part;
      out.push(`<div class="codeblock"><div class="bar"><span>${esc(lang || "code")}</span>` +
               `<button data-copy>Copy</button></div><pre><code>${esc(code.replace(/\n$/, ""))}</code></pre></div>`);
      return;
    }
    let list = null;
    const flush = () => { if (list) { out.push(`</${list}>`); list = null; } };
    for (const raw of part.split("\n")) {
      const line = raw.trimEnd();
      let m;
      if (!line.trim()) { flush(); continue; }
      if ((m = line.match(/^#{1,2}\s+(.*)/))) { flush(); out.push(`<h3>${inline(m[1])}</h3>`); continue; }
      if ((m = line.match(/^#{3,6}\s+(.*)/))) { flush(); out.push(`<h4>${inline(m[1])}</h4>`); continue; }
      if ((m = line.match(/^\s*[-*•]\s+(.*)/))) {
        if (list !== "ul") { flush(); out.push("<ul>"); list = "ul"; }
        out.push(`<li>${inline(m[1])}</li>`); continue;
      }
      if ((m = line.match(/^\s*\d+[.)]\s+(.*)/))) {
        if (list !== "ol") { flush(); out.push("<ol>"); list = "ol"; }
        out.push(`<li>${inline(m[1])}</li>`); continue;
      }
      flush();
      out.push(`<p>${inline(line)}</p>`);
    }
    flush();
  });
  return out.join("");
}

// ---------------------------------------------------------------- rendering
function stamp() {
  return new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}
function scrollDown() { chat.scrollTop = chat.scrollHeight; }
function clearEmpty() { const e = $("empty"); if (e) e.remove(); }

function addMessage(who, text, opts = {}) {
  clearEmpty();
  const el = document.createElement("div");
  el.className = `msg ${who}` + (opts.streaming ? " streaming" : "");
  el.innerHTML = `<div class="meta">${who === "you" ? "You" : "Deo"} · ${stamp()}</div><div class="bubble"></div>`;
  const bubble = el.querySelector(".bubble");
  if (who === "you") bubble.textContent = text;
  else bubble.innerHTML = markdown(text);
  chat.appendChild(el);
  scrollDown();
  return el;
}
function addNotice(text) {
  clearEmpty();
  const el = document.createElement("div");
  el.className = "notice";
  el.textContent = text;
  chat.appendChild(el);
  scrollDown();
}
function streamReply(text) {
  if (!streamEl) streamEl = addMessage("deo", text, { streaming: true });
  else { streamEl.querySelector(".bubble").innerHTML = markdown(text); scrollDown(); }
}
function finishReply(text) {
  if (streamEl) {
    streamEl.querySelector(".bubble").innerHTML = markdown(text);
    streamEl.classList.remove("streaming");
    streamEl = null;
    scrollDown();
  } else {
    addMessage("deo", text);
  }
}
function setState(state) {
  currentState = state;
  const shown = paused && state === "idle" ? "paused" : state;
  document.body.dataset.state = shown;
  $("state").textContent = STATE_WORDS[shown] || shown;
}
function setPaused(p) {
  paused = !!p;
  const b = $("pause");
  b.classList.toggle("on", paused);
  b.title = b.ariaLabel = paused ? "Resume listening" : "Pause listening";
  setState(currentState);
}

// ---------------------------------------------------------------- events from Python
function handle([kind, payload]) {
  switch (kind) {
    case "status": setState(payload); break;
    case "user": streamEl = null; addMessage("you", payload); break;
    case "stream": streamReply(payload); break;
    case "assistant": finishReply(payload); break;
    case "system": addNotice(payload); break;
    case "info": $("info").textContent = payload; break;
    case "paused": setPaused(payload); break;
  }
}
async function poll() {
  try {
    const events = await window.pywebview.api.poll();
    for (const e of events) handle(e);
  } catch (e) { /* the bridge restarts with the window; try again */ }
  setTimeout(poll, 60);
}

// ---------------------------------------------------------------- user actions
$("composer").addEventListener("submit", (ev) => {
  ev.preventDefault();
  const input = $("input");
  const text = input.value.trim();
  if (!text) return;
  input.value = "";
  streamEl = null;
  addMessage("you", text);
  window.pywebview.api.submit(text);
});
$("stop").addEventListener("click", () => window.pywebview.api.stop());
$("pause").addEventListener("click", async () => setPaused(await window.pywebview.api.toggle_pause()));

chat.addEventListener("click", (ev) => {
  const link = ev.target.closest("a[data-href]");
  if (link) { ev.preventDefault(); window.pywebview.api.open_link(link.dataset.href); return; }
  const copy = ev.target.closest("[data-copy]");
  if (copy) {
    const code = copy.closest(".codeblock").querySelector("code").innerText;
    navigator.clipboard.writeText(code);
    copy.textContent = "Copied";
    setTimeout(() => (copy.textContent = "Copy"), 1400);
  }
});

// ---------------------------------------------------------------- guide sheet
function setGuide(open) {
  $("guide").classList.toggle("open", open);
  $("guide").setAttribute("aria-hidden", String(!open));
}
$("guide-open").addEventListener("click", () => setGuide(true));
$("guide-close").addEventListener("click", () => setGuide(false));
$("guide").addEventListener("click", (ev) => { if (ev.target === $("guide")) setGuide(false); });
document.addEventListener("keydown", (ev) => { if (ev.key === "Escape") setGuide(false); });

window.addEventListener("pywebviewready", () => { poll(); $("input").focus(); });

// ---------------------------------------------------------------- the living blob
// An organic glass blob that never sits still. Its outline is a smooth curve
// through points round a circle, each pushed in and out by a few slow sine
// waves; soft iridescent colour swirls inside it. How much it wobbles and how
// fast is set by Deo's state, eased so changes flow rather than jump.
(function blob() {
  const cv = $("blob");
  const ctx = cv.getContext("2d");
  const dpr = window.devicePixelRatio || 1;
  const SIZE = 64;
  cv.width = SIZE * dpr; cv.height = SIZE * dpr;
  ctx.scale(dpr, dpr);

  const MOODS = {          // wobble, speed, spin, colour saturation
    idle:      { amp: 0.045, speed: 0.55, spin: 0.10, sat: 0.55 },
    listening: { amp: 0.090, speed: 1.50, spin: 0.35, sat: 1.00 },
    thinking:  { amp: 0.070, speed: 2.10, spin: 1.30, sat: 0.85 },
    speaking:  { amp: 0.060, speed: 2.80, spin: 0.50, sat: 1.00 },
    paused:    { amp: 0.020, speed: 0.30, spin: 0.05, sat: 0.00 },
    error:     { amp: 0.080, speed: 2.00, spin: 0.40, sat: 0.70 },
  };
  const cur = { ...MOODS.idle };
  let t = 0, last = performance.now(), level = 0, spin = 0;
  const N = 9;
  const seeds = Array.from({ length: N }, (_, i) => [1 + (i * 7) % 5 * 0.31, (i * 2.399) % 6.283]);

  function mood() { return MOODS[document.body.dataset.state] || MOODS.idle; }

  function frame(now) {
    const dt = Math.min(0.05, (now - last) / 1000); last = now;
    const target = mood();
    for (const k in cur) cur[k] += (target[k] - cur[k]) * Math.min(1, dt * 3);
    t += dt * cur.speed;
    spin += dt * cur.spin;
    // speaking: a voice-like level that jumps and decays
    const want = document.body.dataset.state === "speaking" ? 0.35 + Math.random() * 0.65 : 0;
    level += (want - level) * Math.min(1, dt * 10);

    const cx = SIZE / 2, cy = SIZE / 2, R = SIZE * 0.34 * (1 + level * 0.06);
    const pts = [];
    for (let i = 0; i < N; i++) {
      const a = (i / N) * Math.PI * 2 + spin * 0.3;
      const [f, ph] = seeds[i];
      const wob = Math.sin(t * f + ph) * 0.6 + Math.sin(t * f * 1.7 + ph * 2.1) * 0.4;
      const r = R * (1 + (cur.amp + level * 0.07) * wob);
      pts.push([cx + Math.cos(a) * r, cy + Math.sin(a) * r]);
    }

    ctx.clearRect(0, 0, SIZE, SIZE);
    // smooth closed curve through the points (Catmull-Rom as Beziers), kept
    // as a Path2D: the rim stroke at the end must outline THIS shape, not
    // whatever path was drawn last (it once outlined a stray colour circle)
    const shape = new Path2D();
    for (let i = 0; i < N; i++) {
      const p0 = pts[(i - 1 + N) % N], p1 = pts[i], p2 = pts[(i + 1) % N], p3 = pts[(i + 2) % N];
      if (i === 0) shape.moveTo(p1[0], p1[1]);
      shape.bezierCurveTo(p1[0] + (p2[0] - p0[0]) / 6, p1[1] + (p2[1] - p0[1]) / 6,
                          p2[0] - (p3[0] - p1[0]) / 6, p2[1] - (p3[1] - p1[1]) / 6, p2[0], p2[1]);
    }
    shape.closePath();

    ctx.save();
    ctx.clip(shape);
    // base: frosted white glass
    const base = ctx.createRadialGradient(cx - R * 0.35, cy - R * 0.4, R * 0.1, cx, cy, R * 1.25);
    base.addColorStop(0, "rgba(255,255,255,0.95)");
    base.addColorStop(1, "rgba(200,205,220,0.55)");
    ctx.fillStyle = base;
    ctx.fillRect(0, 0, SIZE, SIZE);
    // swirling iridescent colour inside
    const s = cur.sat;
    const hues = [
      [`rgba(165,180,252,${0.85 * s})`, 0.0],      // soft indigo
      [`rgba(125,211,252,${0.80 * s})`, 2.1],      // sky
      [`rgba(249,168,212,${0.70 * s})`, 4.2],      // pink
    ];
    ctx.globalCompositeOperation = "multiply";
    ctx.filter = "blur(7px)";
    for (const [col, off] of hues) {
      const a = spin * 1.4 + off + Math.sin(t * 0.7 + off) * 0.6;
      ctx.fillStyle = col;
      ctx.beginPath();
      ctx.arc(cx + Math.cos(a) * R * 0.55, cy + Math.sin(a) * R * 0.55, R * 0.62, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.filter = "none";
    ctx.globalCompositeOperation = "source-over";
    // glass: bright highlight top-left, soft shade bottom-right
    const hi = ctx.createRadialGradient(cx - R * 0.42, cy - R * 0.5, 0, cx - R * 0.42, cy - R * 0.5, R * 0.6);
    hi.addColorStop(0, "rgba(255,255,255,0.85)");
    hi.addColorStop(1, "rgba(255,255,255,0)");
    ctx.fillStyle = hi;
    ctx.fillRect(0, 0, SIZE, SIZE);
    const lo = ctx.createRadialGradient(cx + R * 0.5, cy + R * 0.6, 0, cx + R * 0.5, cy + R * 0.6, R * 1.0);
    lo.addColorStop(0, "rgba(40,40,60,0.28)");
    lo.addColorStop(1, "rgba(40,40,60,0)");
    ctx.fillStyle = lo;
    ctx.fillRect(0, 0, SIZE, SIZE);
    ctx.restore();
    // crisp rim
    ctx.lineWidth = 1;
    ctx.strokeStyle = "rgba(255,255,255,0.55)";
    ctx.stroke(shape);

    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
})();

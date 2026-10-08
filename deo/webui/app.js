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

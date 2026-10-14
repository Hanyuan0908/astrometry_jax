// kit.js — a small manim-like drawing kit for 3Blue1Brown-style explainer videos on an HTML canvas.
// Every frame is a pure function of time, so the browser preview, scrubbing and the headless export are identical.
//
// Usage (see template/scenes.js):
//   KIT.run({ scenes: [{ id, draw(lt, s) }, ...], load: async () => {...}, events: ev => {...} })
// Scene durations and narration line starts come from timing.json (written by narration.py), so the voice drives timing.
// Exposes on window for export.js: renderFrame(t), DUR, EVENTS (sound cues), VOICE (narration timeline), READY.
(() => {
"use strict";
const W = 1920, H = 1080;
const canvas = document.getElementById("c"), ctx = canvas.getContext("2d");
// 3B1B palette (manim colour names). Keep meanings fixed for the whole film, e.g. BLUE = thing A, RED = thing B.
const C = { BG: "#000000", WHITE: "#ffffff", GREY: "#888888", BLUE: "#58C4DD", RED: "#FC6255", YELLOW: "#FFFF00",
            GREEN: "#83C167", TEAL: "#5CD0B3", GOLD: "#F0AC5F", PURPLE: "#9A72AC" };
// Serif text and math. On Windows Cambria / Cambria Math ship by default; elsewhere STIX Two or Latin Modern are good.
const TF = "Cambria, 'STIX Two Text', 'Latin Modern Roman', 'Times New Roman', serif";
const MF = "'Cambria Math', 'STIX Two Math', 'Latin Modern Math', Cambria, serif";

const clamp = (x, a = 0, b = 1) => Math.min(b, Math.max(a, x)), lerp = (a, b, f) => a + (b - a) * f;
const smooth = x => { x = clamp(x); return x * x * x * (10 + x * (6 * x - 15)); };   // manim's default rate function
const ramp = (t, a, b, e = smooth) => e((t - a) / (b - a));
const font = (size, f = TF, italic = false) => `${italic ? "italic " : ""}${size}px ${f}`;

// ---------- Write: outline sweeps in left to right, the fill follows (manim's Write)
function write(str, x, y, size, color, p, { f = TF, align = "left", italic = false, alpha = 1 } = {}) {
  if (p <= 0 || alpha <= 0) return 0;
  ctx.save(); ctx.font = font(size, f, italic); ctx.textBaseline = "alphabetic";
  const w = ctx.measureText(str).width, x0 = align === "center" ? x - w / 2 : align === "right" ? x - w : x;
  const s1 = clamp(p / 0.7), s2 = clamp((p - 0.3) / 0.7);
  ctx.globalAlpha = alpha;
  if (s2 < 1) { ctx.save(); ctx.beginPath(); ctx.rect(x0 - 4, y - size * 1.2, (w + 8) * s1, size * 1.6); ctx.clip();
    ctx.strokeStyle = color; ctx.lineWidth = Math.max(1, size / 40); ctx.strokeText(str, x0, y); ctx.restore(); }
  ctx.save(); ctx.beginPath(); ctx.rect(x0 - 4, y - size * 1.2, (w + 8) * s2, size * 1.6); ctx.clip();
  ctx.fillStyle = color; ctx.fillText(str, x0, y); ctx.restore(); ctx.restore();
  return w;
}
// ---------- math row: tokens [text, color, mode] with mode "", "sub" or "sup". Use Unicode math italics (𝑣 𝑀 𝑅 𝐺 …).
function mathWidth(tokens, size) { ctx.save(); let w = 0; for (const [t, , m] of tokens) { ctx.font = font(m ? size * 0.62 : size, MF); w += ctx.measureText(t).width; } ctx.restore(); return w; }
function math(tokens, x, y, size, p, { align = "left", alpha = 1 } = {}) {
  const w = mathWidth(tokens, size), x0 = align === "center" ? x - w / 2 : align === "right" ? x - w : x;
  if (p <= 0 || alpha <= 0) return { x0, w, spans: [] };
  ctx.save(); ctx.globalAlpha = alpha;
  const s1 = clamp(p / 0.7), s2 = clamp((p - 0.3) / 0.7), spans = []; let cx = x0;
  for (const [t, c, m] of tokens) { ctx.font = font(m ? size * 0.62 : size, MF); const tw = ctx.measureText(t).width; spans.push([t, c, m, cx, tw]); cx += tw; }
  for (const [pass, sw] of [["stroke", s1], ["fill", s2]]) {
    if (pass === "stroke" && s2 >= 1) continue;
    ctx.save(); ctx.beginPath(); ctx.rect(x0 - 4, y - size * 1.3, (w + 8) * sw, size * 1.8); ctx.clip();
    for (const [t, c, m, px] of spans) { const sz = m ? size * 0.62 : size, dy = m === "sub" ? size * 0.22 : m === "sup" ? -size * 0.42 : 0;
      ctx.font = font(sz, MF);
      if (pass === "stroke") { ctx.strokeStyle = c; ctx.lineWidth = Math.max(1, sz / 40); ctx.strokeText(t, px, y + dy); }
      else { ctx.fillStyle = c; ctx.fillText(t, px, y + dy); } }
    ctx.restore();
  }
  ctx.restore();
  return { x0, w, spans: spans.map(([t, , , px, tw]) => ({ t, x: px, w: tw })) };   // spans: for braces under terms
}
// ---------- Create: a polyline drawn along its length
function path(pts, color, lw, p = 1, { alpha = 1, dash = null, glow = 0 } = {}) {
  if (p <= 0 || alpha <= 0 || pts.length < 2) return;
  let L = 0; const seg = []; for (let i = 1; i < pts.length; i++) { const d = Math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]); seg.push(d); L += d; }
  let rem = L * clamp(p);
  ctx.save(); ctx.globalAlpha = alpha; ctx.strokeStyle = color; ctx.lineWidth = lw; ctx.lineJoin = "round"; ctx.lineCap = "round";
  if (dash) ctx.setLineDash(dash); if (glow) { ctx.shadowColor = color; ctx.shadowBlur = glow; }
  ctx.beginPath(); ctx.moveTo(...pts[0]);
  for (let i = 1; i < pts.length && rem > 0; i++) { const d = seg[i - 1];
    if (rem >= d) ctx.lineTo(...pts[i]); else { const f = rem / d; ctx.lineTo(lerp(pts[i - 1][0], pts[i][0], f), lerp(pts[i - 1][1], pts[i][1], f)); }
    rem -= d; }
  ctx.stroke(); ctx.restore();
}
const line = (x0, y0, x1, y1, c, lw, p = 1, o = {}) => path([[x0, y0], [x1, y1]], c, lw, p, o);
function dot(x, y, r, c, a = 1) { if (a <= 0 || r <= 0) return; ctx.save(); ctx.globalAlpha = a; ctx.fillStyle = c; ctx.beginPath(); ctx.arc(x, y, r, 0, 7); ctx.fill(); ctx.restore(); }
function arrow(x0, y0, x1, y1, c, lw, p = 1, a = 1) {
  if (p <= 0) return; const x = lerp(x0, x1, p), y = lerp(y0, y1, p), ang = Math.atan2(y - y0, x - x0), hl = 16;
  line(x0, y0, x, y, c, lw, 1, { alpha: a });
  ctx.save(); ctx.globalAlpha = a; ctx.fillStyle = c; ctx.beginPath(); ctx.moveTo(x, y);
  ctx.lineTo(x - hl * Math.cos(ang - 0.4), y - hl * Math.sin(ang - 0.4)); ctx.lineTo(x - hl * Math.cos(ang + 0.4), y - hl * Math.sin(ang + 0.4)); ctx.fill(); ctx.restore();
}
// SurroundingRectangle: drawn around its perimeter (yellow is the 3B1B emphasis colour)
function box(x, y, w, h, c, p, a = 1) { path([[x, y], [x + w, y], [x + w, y + h], [x, y + h], [x, y]], c, 3, p, { alpha: a }); }
// horizontal curly brace with a label (down or up)
function brace(x0, x1, y, label, c, p, { down = true, size = 30, labelColor = c } = {}) {
  if (p <= 0) return; const d = down ? 1 : -1, m = (x0 + x1) / 2, h = 18 * d;
  ctx.save(); ctx.globalAlpha = p; ctx.strokeStyle = c; ctx.lineWidth = 2.5; ctx.beginPath();
  ctx.moveTo(x0, y); ctx.quadraticCurveTo(x0, y + h, x0 + 20, y + h); ctx.lineTo(m - 16, y + h); ctx.quadraticCurveTo(m, y + h, m, y + 2 * h);
  ctx.quadraticCurveTo(m, y + h, m + 16, y + h); ctx.lineTo(x1 - 20, y + h); ctx.quadraticCurveTo(x1, y + h, x1, y); ctx.stroke(); ctx.restore();
  write(label, m, y + 2 * h + (down ? size + 8 : -12), size, labelColor, p, { align: "center" });
}
// axes: makeAxes returns data->pixel maps; drawAxes "creates" the arrows, then ticks/labels fade in
function makeAxes(o) {   // o: {x, y (origin, px), w, h, xmin, xmax, ymin, ymax, xt: [], yt: [], xl, yl}
  return { ...o, X: v => o.x + (v - o.xmin) / (o.xmax - o.xmin) * o.w, Y: v => o.y - (v - o.ymin) / (o.ymax - o.ymin) * o.h };
}
function drawAxes(ax, p, a = 1) {
  if (p <= 0 || a <= 0) return;
  arrow(ax.x, ax.y, ax.x + ax.w + 30, ax.y, C.WHITE, 2.5, p, a); arrow(ax.x, ax.y, ax.x, ax.y - ax.h - 30, C.WHITE, 2.5, p, a);
  const ta = a * ramp(p, 0.5, 1);
  for (const v of ax.xt || []) { line(ax.X(v), ax.y - 8, ax.X(v), ax.y + 8, C.WHITE, 2, 1, { alpha: ta }); write(`${v}`, ax.X(v), ax.y + 40, 26, C.WHITE, 1, { align: "center", alpha: ta }); }
  for (const v of ax.yt || []) { line(ax.x - 8, ax.Y(v), ax.x + 8, ax.Y(v), C.WHITE, 2, 1, { alpha: ta }); write(`${v}`, ax.x - 18, ax.Y(v) + 9, 26, C.WHITE, 1, { align: "right", alpha: ta }); }
  if (ax.xl) write(ax.xl, ax.x + ax.w + 40, ax.y + 10, 28, C.WHITE, 1, { alpha: ta, f: MF });
  if (ax.yl) write(ax.yl, ax.x - 10, ax.y - ax.h - 44, 28, C.WHITE, 1, { alpha: ta, f: MF, align: "center" });
}
// a curve on axes; morph between two y-arrays with f in [0,1] (Transform)
function plot(ax, xs, ys, color, lw, p = 1, o = {}) { path(xs.map((x, i) => [ax.X(x), ax.Y(ys[i])]), color, lw, p, o); }
const morph = (ya, yb, f) => ya.map((v, i) => lerp(v, yb[i], f));
function image(img, x, y, w, h, a = 1, rise = 0) { if (a <= 0) return; ctx.save(); ctx.globalAlpha = a; ctx.drawImage(img, x, y + rise, w, h); ctx.restore(); }   // FadeIn(shift=UP) with rise = (1-f)*40
function loadImage(src) { return new Promise((res, rej) => { const i = new Image(); i.onload = () => res(i); i.onerror = () => rej(new Error("asset " + src)); i.src = src; }); }

// ---------- timeline, preview player, export hooks
async function run({ scenes, load = async () => {}, events = () => {}, timing = "timing.json", voiceDir = "voice" }) {
  const exportMode = new URLSearchParams(location.search).has("export");
  if (exportMode) document.body.classList.add("export");
  try {
    const T = await fetch(timing).then(r => { if (!r.ok) throw new Error(`${timing} missing: run narration.py first`); return r.json(); });
    for (const s of scenes) { const ts = T.scenes.find(x => x.id === s.id); if (!ts) throw new Error(`timing.json has no scene "${s.id}"`);
      s.dur = s.dur ?? ts.dur; s.lines = ts.lines; }
    let acc = 0; for (const s of scenes) { s.start = acc; acc += s.dur; }
    const DUR = acc, VOICE = [], EVENTS = [];
    for (const s of scenes) s.lines.forEach((ln, k) => VOICE.push({ t: +(s.start + ln.t).toFixed(3), file: `${voiceDir}/${s.id}_${k}.mp3` }));
    await Promise.all([document.fonts.ready, load()]);
    const byId = Object.fromEntries(scenes.map(s => [s.id, s]));
    events((id, t, type, x = {}) => EVENTS.push({ t: +(byId[id].start + t).toFixed(3), type, pan: x.pan ?? 0, ...x }));
    EVENTS.sort((a, b) => a.t - b.t);
    const render = t => {
      let s = scenes.at(-1); for (const c of scenes) if (t >= c.start && t < c.start + c.dur) { s = c; break; }
      ctx.globalCompositeOperation = "source-over"; ctx.globalAlpha = 1; ctx.fillStyle = C.BG; ctx.fillRect(0, 0, W, H);
      s.draw(t - s.start, s);
      const fade = Math.min(ramp(t, 0, 0.4), 1 - ramp(t, DUR - 0.6, DUR));
      if (fade < 1) { ctx.fillStyle = `rgba(0,0,0,${1 - fade})`; ctx.fillRect(0, 0, W, H); }
    };
    // preview player
    const scrub = document.getElementById("scrub"), lab = document.getElementById("time"), btn = document.getElementById("play");
    let playing = false, t0 = 0, tNow = 0;
    const show = t => { tNow = clamp(t, 0, DUR); render(tNow); scrub.value = Math.round(tNow / DUR * 1000); lab.textContent = `${tNow.toFixed(1)} / ${DUR.toFixed(1)} s`; };
    const loop = now => { if (!playing) return; const t = (now - t0) / 1000; if (t >= DUR) { playing = false; btn.textContent = "Play"; show(DUR); return; } show(t); requestAnimationFrame(loop); };
    btn.onclick = () => { playing = !playing; btn.textContent = playing ? "Pause" : "Play"; if (playing) { if (tNow >= DUR) tNow = 0; t0 = performance.now() - tNow * 1000; requestAnimationFrame(loop); } };
    scrub.oninput = () => { playing = false; btn.textContent = "Play"; show(scrub.value / 1000 * DUR); };
    Object.assign(window, { renderFrame: render, DUR, EVENTS, VOICE });
    show(0); window.READY = true;
  } catch (e) { console.error(e); window.LOAD_ERROR = String(e); document.body.insertAdjacentHTML("beforeend", `<pre style="color:#f66">${e}</pre>`); }
}

window.KIT = { W, H, ctx, C, TF, MF, clamp, lerp, smooth, ramp, font, write, math, mathWidth, path, line, dot, arrow, box, brace,
               makeAxes, drawAxes, plot, morph, image, loadImage, run };
})();

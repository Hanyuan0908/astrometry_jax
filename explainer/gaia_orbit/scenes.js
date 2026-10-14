// scenes.js — "How does Gaia see a star with a dark companion?"
// Colour meanings (fixed for the whole film): YELLOW = the star, GREY = the dark companion, BLUE = the orbit,
// RED = parallax, GREEN = proper motion, WHITE = the total path on the sky, GOLD = Gaia's measurements.
(() => {
"use strict";
const { W, H, C, MF, ctx, ramp, lerp, clamp, write, math, box, brace, path, line, dot, arrow } = KIT;
const F = {};
const at = (s, k, dt = 0) => s.lines[k].t + dt;
const outro = (lt, s) => 1 - ramp(lt, s.dur - 0.6, s.dur);

// ---------- sky view: RA offset to the left (east), Dec up. scale in px per mas, centre (cx, cy)
const sky = { cx: 960, cy: 560, scale: 120 };
const SX = (ra, v = sky) => v.cx - ra * v.scale;
const SY = (dec, v = sky) => v.cy - dec * v.scale;
const idxAt = tDay => { const t = F.t_day, n = t.length; const f = (tDay - t[0]) / (t[n - 1] - t[0]); return clamp(Math.round(f * (n - 1)), 0, n - 1); };
const P_IDX = () => idxAt(F.t_day[0] + F.system.period_day);          // one orbital period
const YR_IDX = () => idxAt(F.t_day[0] + 365.25);                        // one year
function track(parts, i0, i1, v = sky) {        // pixel polyline of the sum of chosen components over index range
  const pts = [];
  for (let i = i0; i <= i1; i++) {
    let ra = 0, dec = 0;
    for (const p of parts) { ra += F[p].ra[i]; dec += F[p].dec[i]; }
    pts.push([SX(ra, v), SY(dec, v)]);
  }
  return pts;
}
function pos(parts, i, v = sky) { let ra = 0, dec = 0; for (const p of parts) { ra += F[p].ra[i]; dec += F[p].dec[i]; } return [SX(ra, v), SY(dec, v)]; }
function compass(v, a, label = true) {         // small axes cross + scale bar
  if (a <= 0) return;
  const x0 = 1560, y0 = 940;
  arrow(x0, y0, x0 - 90, y0, C.GREY, 2, 1, a); arrow(x0, y0, x0, y0 - 90, C.GREY, 2, 1, a);
  write("RA", x0 - 130, y0 + 10, 24, C.GREY, 1, { alpha: a }); write("Dec", x0 - 22, y0 - 100, 24, C.GREY, 1, { alpha: a });
  const mas = v.scale > 60 ? 1 : 10, px = mas * v.scale;
  line(x0 - 90 - px, y0 + 50, x0 - 90, y0 + 50, C.GREY, 3, 1, { alpha: a });
  if (label) write(`${mas} mas`, x0 - 90 - px / 2, y0 + 85, 24, C.GREY, 1, { align: "center", alpha: a });
}
const starDot = (x, y, a, r = 11) => { dot(x, y, r * 1.9, "rgba(255,255,0,0.18)", a); dot(x, y, r, C.YELLOW, a); };
const compDot = (x, y, a, r = 9) => { ctx.save(); ctx.globalAlpha = a; ctx.strokeStyle = C.GREY; ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(x, y, r, 0, 7); ctx.stroke(); ctx.restore(); };

// ---------- 1 · title
function title(lt, s) {
  const o = outro(lt, s);
  write("How does Gaia see a star", W / 2, 440, 84, C.WHITE, ramp(lt, at(s, 0), at(s, 0, 1.6)), { align: "center", alpha: o });
  write("with a dark companion?", W / 2, 550, 84, C.WHITE, ramp(lt, at(s, 0, 0.8), at(s, 0, 2.4)), { align: "center", alpha: o });
  // a star and an invisible companion circling, as a motif
  const f = lt / 6, r = 150;
  const p1 = ramp(lt, at(s, 1), at(s, 1, 1.2));
  starDot(W / 2 + r * 0.58 * Math.cos(6.28 * f), 780 + r * 0.3 * Math.sin(6.28 * f), p1 * o);
  compDot(W / 2 - r * 0.42 * Math.cos(6.28 * f), 780 - r * 0.22 * Math.sin(6.28 * f), p1 * o);
  write("orbit  +  parallax  +  proper motion  =  the path Gaia samples", W / 2, 960, 34, C.GREY, ramp(lt, at(s, 1, 0.6), at(s, 1, 2.4)), { align: "center", alpha: o });
}

// ---------- 2 · the orbit, seen from a frame moving with the system
function orbit(lt, s) {
  const o = outro(lt, s), v = sky, iP = P_IDX();
  const q = F.system.q;
  const t0 = at(s, 1, 0.8), tP = at(s, 2, 0.5);                    // one period of motion between these times
  const f = clamp((lt - t0) / (tP - t0)); const i = Math.round(f * iP);
  // centre of mass and labels
  const a0 = ramp(lt, at(s, 0), at(s, 0, 0.8));
  line(v.cx - 14, v.cy, v.cx + 14, v.cy, C.WHITE, 2, 1, { alpha: a0 * o }); line(v.cx, v.cy - 14, v.cx, v.cy + 14, C.WHITE, 2, 1, { alpha: a0 * o });
  write("centre of mass", v.cx + 24, v.cy + 44, 26, C.GREY, ramp(lt, at(s, 1), at(s, 1, 0.8)), { alpha: o });
  write("1 M⊙ star", 180, 150, 34, C.YELLOW, ramp(lt, at(s, 0, 0.3), at(s, 0, 1.3)), { alpha: o });
  write("+ 1.4 M⊙ dark companion", 180, 200, 34, C.GREY, ramp(lt, at(s, 0, 1.6), at(s, 0, 2.6)), { alpha: o });
  write("P = 600 d,  d = 500 pc", 180, 250, 34, C.WHITE, ramp(lt, at(s, 0, 3.2), at(s, 0, 4.2)), { alpha: o });
  write("observer at rest with the system", 180, 310, 28, C.GREY, ramp(lt, at(s, 1), at(s, 1, 1.0)), { alpha: o });
  // the two orbits are traced as the bodies move
  if (f > 0) {
    path(track(["star"], 0, i), C.BLUE, 4, 1, { alpha: o });
    path(track(["companion"], 0, i), C.GREY, 2, 1, { alpha: 0.7 * o, dash: [8, 8] });
  }
  if (f >= 1) { path(track(["star"], 0, iP), C.BLUE, 4, 1, { alpha: o }); }
  const ps = pos(["star"], i), pc = pos(["companion"], i);
  line(pc[0], pc[1], ps[0], ps[1], C.GREY, 1.5, 1, { alpha: 0.5 * ramp(lt, at(s, 1), at(s, 1, 0.8)) * o });
  starDot(ps[0], ps[1], ramp(lt, at(s, 0), at(s, 0, 0.6)) * o); compDot(pc[0], pc[1], ramp(lt, at(s, 0, 1.6), at(s, 0, 2.2)) * o);
  // a0: the semi-major axis of the star's ellipse (the photocentre orbit)
  const pa = ramp(lt, at(s, 2), at(s, 2, 1.0));
  if (pa > 0) {
    // the ellipse's major axis: use the two points farthest apart on the traced ellipse
    const pts = track(["star"], 0, iP); let best = [0, 0, 0];
    for (let a = 0; a < pts.length; a += 8) for (let b = a; b < pts.length; b += 8) { const d = Math.hypot(pts[a][0] - pts[b][0], pts[a][1] - pts[b][1]); if (d > best[0]) best = [d, a, b]; }
    const [, ia, ib] = best, mx = (pts[ia][0] + pts[ib][0]) / 2, my = (pts[ia][1] + pts[ib][1]) / 2;
    line(mx, my, lerp(mx, pts[ib][0], pa), lerp(my, pts[ib][1], pa), C.YELLOW, 3, 1, { alpha: o });
    math([["𝑎", C.YELLOW], ["0", C.YELLOW, "sub"], [" = 2.2 mas", C.YELLOW]], 180, 860, 40, pa, { alpha: o });
    write("≈ 1.9 au at 500 pc", 180, 908, 26, C.GREY, ramp(lt, at(s, 2, 1.4), at(s, 2, 2.2)), { alpha: o });
    write("grows with the companion's mass", 180, 950, 26, C.GREY, ramp(lt, at(s, 2, 4.0), at(s, 2, 5.0)), { alpha: o });
  }
  compass(v, ramp(lt, at(s, 0), at(s, 0, 1)) * o);
}

// ---------- 3 · put Gaia on its real orbit: the parallax ellipse; orbit + parallax = the rosette
function parallax(lt, s) {
  const o = outro(lt, s), v = sky, iP = P_IDX(), iY = YR_IDX(), n = F.t_day.length - 1;
  // left inset: Sun, Earth's orbit, Earth + Gaia moving
  const ins = { cx: 300, cy: 300, r: 150 }, pi = ramp(lt, at(s, 0), at(s, 0, 1.2)) * o;
  const iE = Math.round(clamp((lt - at(s, 0, 1.0)) / (at(s, 1, 0.5) - at(s, 0, 1.0))) * iY);
  if (pi > 0) {
    dot(ins.cx, ins.cy, 14, C.GOLD, pi); write("Sun", ins.cx + 22, ins.cy + 10, 24, C.GOLD, 1, { alpha: pi });
    ctx.save(); ctx.globalAlpha = pi * 0.6; ctx.strokeStyle = C.GREY; ctx.lineWidth = 1.5; ctx.setLineDash([6, 6]); ctx.beginPath(); ctx.arc(ins.cx, ins.cy, ins.r, 0, 7); ctx.stroke(); ctx.restore();
    const ex = ins.cx + ins.r * F.earth_au.x[iE], ey = ins.cy - ins.r * F.earth_au.y[iE];
    dot(ex, ey, 9, C.BLUE, pi); const gx = ex + 0.0 + 26 * F.earth_au.x[iE], gy = ey - 26 * F.earth_au.y[iE];
    dot(gx, gy, 5, C.WHITE, pi); write("Gaia", gx + 12, gy - 10, 22, C.WHITE, 1, { alpha: pi }); write("Earth", ex - 70, ey + 34, 22, C.BLUE, 1, { alpha: pi });
    write("Gaia orbits the Sun with the Earth", 110, 520, 26, C.GREY, ramp(lt, at(s, 0, 2.0), at(s, 0, 3.0)), { alpha: o });
  }
  // sky view: the orbit ellipse stays (blue), the parallax ellipse is drawn over one year (red)
  line(v.cx - 14, v.cy, v.cx + 14, v.cy, C.WHITE, 2, 1, { alpha: o }); line(v.cx, v.cy - 14, v.cx, v.cy + 14, C.WHITE, 2, 1, { alpha: o });
  const fadeOrbit = 1 - 0.6 * ramp(lt, at(s, 1), at(s, 1, 1.0));
  path(track(["star"], 0, iP), C.BLUE, 4, 1, { alpha: o * fadeOrbit });
  write("orbit", SX(F.star.ra[0]) + 18, SY(F.star.dec[0]) - 14, 26, C.BLUE, 1, { alpha: o * fadeOrbit });
  const pp = ramp(lt, at(s, 1), at(s, 1, 3.0));
  if (pp > 0) { path(track(["parallax"], 0, iY), C.RED, 4, pp, { alpha: o }); write("parallax ellipse, ϖ = 2 mas", 1250, 230, 30, C.RED, ramp(lt, at(s, 1, 1.5), at(s, 1, 2.5)), { alpha: o }); }
  // the sum: traced over the full span while the star moves along it
  const f2 = clamp((lt - at(s, 2, 0.8)) / (s.dur - 1.2 - at(s, 2, 0.8))); const i2 = Math.round(f2 * n);
  if (f2 > 0) {
    path(track(["star", "parallax"], 0, i2), C.WHITE, 3, 1, { alpha: o });
    const ps = pos(["star", "parallax"], i2); starDot(ps[0], ps[1], o);
    write("orbit + parallax", 1250, 290, 30, C.WHITE, ramp(lt, at(s, 2, 1.0), at(s, 2, 2.0)), { alpha: o });
    const yr = (F.t_day[i2] / 365.25 + F.system.year0).toFixed(1);
    write(yr, 1700, 1000, 34, C.GREY, 1, { align: "right", alpha: o });
  } else {
    const ps = pos(["star"], 0); starDot(ps[0], ps[1], o);
  }
  compass(v, o);
}

// ---------- 4 · proper motion: the drift, then zoom out to the true path
function pm(lt, s) {
  const o = outro(lt, s), n = F.t_day.length - 1;
  const zoomF = ramp(lt, at(s, 1, 0.5), at(s, 1, 3.5));                      // 120 -> 16 px/mas
  const v = { cx: lerp(960, 1180, zoomF), cy: lerp(560, 760, zoomF), scale: Math.exp(lerp(Math.log(120), Math.log(16), zoomF)) };
  line(v.cx - 14, v.cy, v.cx + 14, v.cy, C.WHITE, 2, 1, { alpha: o }); line(v.cx, v.cy - 14, v.cx, v.cy + 14, C.WHITE, 2, 1, { alpha: o });
  // the rosette from before, fading as the drift takes over
  const ros = 1 - 0.7 * ramp(lt, at(s, 1), at(s, 1, 2.0));
  path(track(["star", "parallax"], 0, n, v), C.GREY, 2, 1, { alpha: o * ros * 0.8 });
  // proper-motion arrow (green) on line 0
  const pa = ramp(lt, at(s, 0, 0.5), at(s, 0, 2.0));
  const ux = -F.system.pm_mas_yr[0], uy = -F.system.pm_mas_yr[1], L = Math.hypot(ux, uy);
  const al = lerp(260, 0.5 * F.system.pm_total_mas * v.scale, zoomF);
  arrow(v.cx, v.cy, v.cx + ux / L * al * 0.0 + ux / L * al, v.cy + uy / L * al, C.GREEN, 4, pa, o);
  write("proper motion", 180, 150, 34, C.GREEN, pa, { alpha: o });
  write("8, −5 mas per year", 180, 200, 30, C.GREEN, ramp(lt, at(s, 0, 2.2), at(s, 0, 3.0)), { alpha: o });
  // the true path: orbit + parallax + proper motion, traced over the span
  const f = clamp((lt - at(s, 1)) / (at(s, 2, 1.0) - at(s, 1))); const i = Math.round(f * n);
  if (f > 0) { path(track(["star", "parallax", "pm"], 0, i, v), C.WHITE, 3, 1, { alpha: o }); const ps = pos(["star", "parallax", "pm"], i, v); starDot(ps[0], ps[1], o, 9); }
  else { const ps = pos(["star", "parallax", "pm"], 0, v); starDot(ps[0], ps[1], o, 9); }
  write("50 mas in 5.3 years", 180, 300, 30, C.WHITE, ramp(lt, at(s, 1, 2.0), at(s, 1, 3.0)), { alpha: o });
  write("orbit + parallax + proper motion", 180, 350, 30, C.WHITE, ramp(lt, at(s, 2), at(s, 2, 1.2)), { alpha: o });
  write("= the path on the sky", 180, 395, 30, C.WHITE, ramp(lt, at(s, 2, 1.2), at(s, 2, 2.2)), { alpha: o });
  const yr = (F.t_day[i] / 365.25 + F.system.year0).toFixed(1);
  write(yr, 1700, 1000, 34, C.GREY, 1, { align: "right", alpha: o * ramp(lt, at(s, 1), at(s, 1, 0.5)) });
  compass(v, o);
}

// ---------- 5 · what Gaia measures: one coordinate per transit, along the scan direction
function scans(lt, s) {
  const o = outro(lt, s), v = sky, n = F.t_day.length - 1, S = F.scans, N = S.t_day.length;
  line(v.cx - 14, v.cy, v.cx + 14, v.cy, C.WHITE, 2, 1, { alpha: o }); line(v.cx, v.cy - 14, v.cx, v.cy + 14, C.WHITE, 2, 1, { alpha: o });
  // the path (pm removed, so it fits) stays faint; the measurements appear one by one during line 1
  const showPath = 1 - 0.75 * ramp(lt, at(s, 0, 1.0), at(s, 0, 2.0)) + 0.75 * ramp(lt, at(s, 2, 1.0), at(s, 2, 2.5));
  path(track(["star", "parallax"], 0, n), C.WHITE, 3, 1, { alpha: o * showPath });
  const k0 = at(s, 0, 3.0), k1 = at(s, 1, 4.0);                                 // measurements arrive between these times
  const first = ramp(lt, at(s, 0, 1.2), at(s, 0, 2.2));
  for (let k = 0; k < N; k++) {
    const pk = k === 0 ? first : ramp(lt, k0 + (k - 1) / (N - 1) * (k1 - k0), k0 + (k - 1) / (N - 1) * (k1 - k0) + 0.4);
    if (pk <= 0) continue;
    const i = idxAt(S.t_day[k]); const [x, y] = pos(["star", "parallax"], i);
    const sp = Math.sin(S.psi[k]), cp = Math.cos(S.psi[k]);
    // the measured coordinate is along the scan direction (sin psi, cos psi) in (RA, Dec); the unknown is across it
    const mx = x - S.resid[k] * sp * v.scale, my = y - S.resid[k] * cp * v.scale;     // measured point (model + residual)
    const ax = cp, ay = sp;                                                            // across-scan direction on screen
    const Lk = k === 0 ? 700 : 220;
    line(mx - ax * Lk * pk, my - ay * Lk * pk, mx + ax * Lk * pk, my + ay * Lk * pk, C.GOLD, k === 0 ? 2 : 1, 1, { alpha: o * (k === 0 ? 0.9 : 0.3) });
    dot(mx, my, 5, C.GOLD, o * pk);
  }
  if (first > 0) {
    const i = idxAt(S.t_day[0]); const [x, y] = pos(["star", "parallax"], i);
    const sp = Math.sin(S.psi[0]), cp = Math.cos(S.psi[0]);
    arrow(x - sp * 90 * v.scale / 120 * 1.6, y - cp * 90 * 1.6, x + sp * 90 * 1.6, y + cp * 90 * 1.6, C.GOLD, 3, first, o);
    write("scan direction", x + sp * 160 + 20, y + cp * 160, 26, C.GOLD, first, { alpha: o });
    write("the star is somewhere on this line", 180, 1000, 28, C.GOLD, ramp(lt, at(s, 0, 2.6), at(s, 0, 3.6)), { alpha: o * (1 - ramp(lt, at(s, 1, 2), at(s, 1, 2.8))) });
  }
  write("one number per crossing", 180, 150, 34, C.GOLD, ramp(lt, at(s, 0, 0.3), at(s, 0, 1.3)), { alpha: o });
  write("62 crossings, 0.05 mas each", 180, 200, 30, C.GOLD, ramp(lt, at(s, 1), at(s, 1, 1.0)), { alpha: o });
  write("the fit: parallax + proper motion + orbit", 180, 290, 30, C.WHITE, ramp(lt, at(s, 2), at(s, 2, 1.2)), { alpha: o });
  write("through all of them", 180, 335, 30, C.WHITE, ramp(lt, at(s, 2, 1.2), at(s, 2, 2.0)), { alpha: o });
  compass(v, o);
}

// ---------- 6 · the pay-off: the companion's mass
function end(lt, s) {
  const o = outro(lt, s);
  const e1 = math([["𝑎", C.YELLOW], ["0", C.YELLOW, "sub"], [" = ", C.WHITE], ["𝑎", C.WHITE], [" · ", C.WHITE], ["ϖ", C.RED], [" · ", C.WHITE],
                   ["𝑀", C.GREY], ["2", C.GREY, "sub"], [" / (", C.WHITE], ["𝑀", C.YELLOW], ["1", C.YELLOW, "sub"], [" + ", C.WHITE], ["𝑀", C.GREY], ["2", C.GREY, "sub"], [")", C.WHITE]],
                  W / 2, 330, 72, ramp(lt, at(s, 0), at(s, 0, 2.0)), { align: "center", alpha: o });
  const e2 = math([["𝑎", C.WHITE], ["3", C.WHITE, "sup"], [" = (", C.WHITE], ["𝑀", C.YELLOW], ["1", C.YELLOW, "sub"], [" + ", C.WHITE], ["𝑀", C.GREY], ["2", C.GREY, "sub"], [") ", C.WHITE], ["𝑃", C.BLUE], ["2", C.BLUE, "sup"]],
                  W / 2, 480, 72, ramp(lt, at(s, 0, 1.8), at(s, 0, 3.6)), { align: "center", alpha: o });
  if (e1.spans.length) {
    brace(e1.spans[0].x, e1.spans[1].x + e1.spans[1].w, 350, "orbit size", C.YELLOW, ramp(lt, at(s, 0, 2.6), at(s, 0, 3.4)) * o, { size: 26 });
    brace(e1.spans[5].x, e1.spans[5].x + e1.spans[5].w, 350, "parallax", C.RED, ramp(lt, at(s, 0, 3.2), at(s, 0, 4.0)) * o, { size: 26 });
  }
  if (e2.spans.length) brace(e2.spans[9].x, e2.spans[10].x + e2.spans[10].w, 500, "period", C.BLUE, ramp(lt, at(s, 0, 4.2), at(s, 0, 5.0)) * o, { size: 26 });
  write("(au, years, solar masses)", W / 2, 560, 26, C.GREY, ramp(lt, at(s, 0, 4.0), at(s, 0, 4.8)), { align: "center", alpha: o });
  const pr = ramp(lt, at(s, 0, 6.0), at(s, 0, 7.5));
  const e3 = math([["𝑀", C.GREY], ["2", C.GREY, "sub"], [" = 1.4 M⊙", C.WHITE]], W / 2, 760, 90, pr, { align: "center", alpha: o });
  if (e3.spans.length) box(e3.x0 - 40, 760 - 100, e3.w + 80, 130, C.YELLOW, ramp(lt, at(s, 0, 7.6), at(s, 0, 8.6)), o);
  write("from a₀ = 2.2 mas, P = 600 d, ϖ = 2 mas", W / 2, 860, 30, C.GREY, ramp(lt, at(s, 0, 8.0), at(s, 0, 9.0)), { align: "center", alpha: o });
  write("That is how a dark companion is weighed.", W / 2, 990, 40, C.WHITE, ramp(lt, at(s, 1), at(s, 1, 1.6)), { align: "center", alpha: o });
}

KIT.run({
  scenes: [{ id: "title", draw: title }, { id: "orbit", draw: orbit }, { id: "parallax", draw: parallax },
           { id: "pm", draw: pm }, { id: "scans", draw: scans }, { id: "end", draw: end }],
  load: async () => Object.assign(F, await fetch("facts.json").then(r => r.json())),
  events: ev => {
    ev("orbit", 0.8, "shimmer", { pan: 0 });
    ev("parallax", 0.9, "whoosh", { dur: 1.4, pan: -0.5 });
    ev("pm", 1.1, "whoosh", { dur: 1.8, pan: 0.3 });
    ev("end", 8.0, "ping", { pan: 0 });
  },
});
})();

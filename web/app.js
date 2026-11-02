/* FaceMatch front-end — vanilla JS, no external dependencies. */
"use strict";

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => [...document.querySelectorAll(sel)];

const state = {
  config: null,
  stats: null,
  searchImage: null,      // {file, url, img}
  selectedPerson: null,
};

/* ------------------------------------------------------------------ boot */
async function boot() {
  bindTabs();
  bindSearch();
  bindEnroll();
  bindGallery();
  bindDatabase();
  bindBench();
  bindModal();
  await refreshMeta();
}

async function refreshMeta() {
  const [cfg, stats, persons] = await Promise.all([
    fetch("/api/config").then((r) => r.json()),
    fetch("/api/stats").then((r) => r.json()),
    fetch("/api/persons").then((r) => r.json()),
  ]);
  state.config = cfg;
  state.stats = stats;

  const chips = $("#status-chips");
  chips.innerHTML = `
    <span class="chip ok">embedder <b>${stats.embedder.name}</b></span>
    <span class="chip">store <b>${stats.store.store}</b></span>
    <span class="chip">vectors <b>${stats.index.vectors}</b></span>
    <span class="chip">index <b>${stats.index.engine}</b></span>
    <span class="chip">threshold <b>${stats.threshold.toFixed(2)}</b></span>`;

  const slider = $("#search-threshold");
  if (!slider.dataset.touched) {
    slider.value = stats.threshold;
    $("#search-threshold-out").textContent = Number(slider.value).toFixed(2);
  }
  $("#person-names").innerHTML = persons.persons.map((p) => `<option value="${esc(p.name)}">`).join("");
}

function bindTabs() {
  $$("#tabs button").forEach((btn) =>
    btn.addEventListener("click", () => {
      $$("#tabs button").forEach((b) => b.classList.toggle("active", b === btn));
      $$(".panel").forEach((p) => p.classList.toggle("active", p.id === `tab-${btn.dataset.tab}`));
      if (btn.dataset.tab === "gallery") loadPersons();
      if (btn.dataset.tab === "database") loadDatabase();
    })
  );
}

/* ---------------------------------------------------------------- search */
function bindSearch() {
  const dz = $("#search-drop");
  const input = $("#search-file");
  dz.addEventListener("click", () => input.click());
  dz.addEventListener("dragover", (e) => { e.preventDefault(); dz.classList.add("drag"); });
  dz.addEventListener("dragleave", () => dz.classList.remove("drag"));
  dz.addEventListener("drop", (e) => {
    e.preventDefault(); dz.classList.remove("drag");
    if (e.dataTransfer.files[0]) setSearchFile(e.dataTransfer.files[0]);
  });
  input.addEventListener("change", () => setSearchFile(input.files[0]));
  document.addEventListener("paste", (e) => {
    const item = [...(e.clipboardData?.items || [])].find((i) => i.type.startsWith("image/"));
    if (item) setSearchFile(item.getAsFile());
  });
  $("#search-threshold").addEventListener("input", (e) => {
    e.target.dataset.touched = "1";
    $("#search-threshold-out").textContent = Number(e.target.value).toFixed(2);
  });
  $("#search-btn").addEventListener("click", runSearch);
}

function setSearchFile(file) {
  if (!file) return;
  const url = URL.createObjectURL(file);
  const img = new Image();
  img.onload = () => {
    state.searchImage = { file, url, img };
    drawCanvas([]);
    $("#search-preview-wrap").classList.remove("hidden");
    $("#search-btn").disabled = false;
  };
  img.src = url;
}

function drawCanvas(faces) {
  const { img } = state.searchImage || {};
  if (!img) return;
  const canvas = $("#search-canvas");
  const scale = Math.min(1, 900 / img.width);
  canvas.width = img.width * scale;
  canvas.height = img.height * scale;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
  faces.forEach((f, i) => {
    const [x, y, w, h] = f.bbox.map((v) => v * scale);
    const good = f.matched;
    ctx.strokeStyle = good ? "#34d399" : "#f87171";
    ctx.lineWidth = Math.max(2, canvas.width / 300);
    ctx.strokeRect(x, y, w, h);
    const label = good ? `#${i + 1} match ${(f.best_similarity * 100).toFixed(0)}%` : `#${i + 1} no match`;
    ctx.font = `600 ${Math.max(12, canvas.width / 45)}px system-ui`;
    const tw = ctx.measureText(label).width;
    ctx.fillStyle = good ? "rgba(52,211,153,.9)" : "rgba(248,113,113,.9)";
    ctx.fillRect(x, Math.max(0, y - 22), tw + 10, 22);
    ctx.fillStyle = "#06281c";
    ctx.fillText(label, x + 5, Math.max(14, y - 6));
  });
}

async function runSearch() {
  if (!state.searchImage) return;
  const btn = $("#search-btn");
  btn.disabled = true; btn.textContent = "Searching…";
  const fd = new FormData();
  fd.append("file", state.searchImage.file);
  fd.append("top_k", $("#search-topk").value);
  fd.append("threshold", $("#search-threshold").value);
  try {
    const res = await fetch("/api/search", { method: "POST", body: fd });
    if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
    const data = await res.json();
    drawCanvas(data.faces);
    renderSearchResults(data);
  } catch (err) {
    $("#search-results").innerHTML = `<div class="empty">Search failed: ${esc(err.message)}</div>`;
  } finally {
    btn.disabled = false; btn.textContent = "Search face";
  }
}

function renderSearchResults(data) {
  const host = $("#search-results");
  if (!data.faces.length) {
    host.innerHTML = `<div class="empty">No face detected in that photo.
      Try a clearer, front-facing image (detection confidence floor:
      ${state.config.detection_score_threshold}).</div>`;
    return;
  }
  host.innerHTML = data.faces
    .map((f, i) => {
      const verdict = f.matched
        ? `<span class="verdict match">MATCH</span>`
        : `<span class="verdict nomatch">NO MATCH</span>`;
      const cands = f.candidates.length
        ? f.candidates
            .map(
              (c) => `
          <div class="cand">
            <img src="${c.crop_url}" title="stored face crop" data-full="${c.image_url}" />
            <div>
              <div class="who">${esc(c.person_name)}</div>
              <div class="bar"><i style="width:${Math.max(0, c.similarity * 100)}%"></i></div>
            </div>
            <div style="text-align:right">
              <div class="sim" style="color:${c.matched ? "var(--accent)" : "var(--muted)"}">${(c.similarity * 100).toFixed(1)}%</div>
              <div class="acts">
                <button data-open="${c.image_url}">photo</button>
                <button data-open="/api/media/faces/${c.face_id}/annotated">annotated</button>
              </div>
            </div>
          </div>`
            )
            .join("")
        : `<div class="muted">gallery is empty &mdash; enroll someone first</div>`;
      return `
      <div class="face-card">
        <img class="qcrop" src="data:image/jpeg;base64,${f.crop_thumb_b64}" alt="query face ${i + 1}" />
        <div style="flex:1">
          <div style="display:flex;gap:10px;align-items:center;flex-wrap:wrap">
            ${verdict}
            <span class="muted">face #${i + 1} &middot; det ${f.score.toFixed(2)} &middot;
              best sim ${f.best_similarity === null ? "&mdash;" : (f.best_similarity * 100).toFixed(1) + "%"}</span>
          </div>
          <div class="candidates">${cands}</div>
        </div>
      </div>`;
    })
    .join("");
  const t = data.timings_ms;
  $("#search-timings").innerHTML =
    `decode <b>${t.decode}ms</b> detect <b>${t.detect}ms</b> embed <b>${t.embed}ms</b> ` +
    `search <b>${t.search}ms</b> total <b>${t.total}ms</b> ` +
    `<span class="muted">| index: ${data.index.engine} / ${data.index.vectors} vectors</span>`;
  host.querySelectorAll("[data-open]").forEach((b) =>
    b.addEventListener("click", () => openModal(b.dataset.open, ""))
  );
  host.querySelectorAll(".cand img").forEach((im) =>
    im.addEventListener("click", () => openModal(im.dataset.full, ""))
  );
}

/* ---------------------------------------------------------------- enroll */
function bindEnroll() {
  const dz = $("#enroll-drop");
  const input = $("#enroll-file");
  dz.addEventListener("click", () => input.click());
  dz.addEventListener("dragover", (e) => { e.preventDefault(); dz.classList.add("drag"); });
  dz.addEventListener("dragleave", () => dz.classList.remove("drag"));
  dz.addEventListener("drop", (e) => {
    e.preventDefault(); dz.classList.remove("drag");
    enrollFiles([...e.dataTransfer.files]);
  });
  input.addEventListener("change", () => { enrollFiles([...input.files]); input.value = ""; });
}

async function enrollFiles(files) {
  const person = $("#enroll-person").value.trim();
  if (!person) { alert("Type a person name first."); return; }
  const log = $("#enroll-log");
  for (const file of files) {
    const url = URL.createObjectURL(file);
    const row = document.createElement("div");
    row.className = "row";
    row.innerHTML = `<img src="${url}" /><div>${esc(file.name)}</div><div class="warn">…</div>`;
    log.prepend(row);
    const status = row.lastElementChild;
    try {
      const fd = new FormData();
      fd.append("file", file);
      fd.append("person", person);
      fd.append("create_person", $("#enroll-create").checked ? "true" : "false");
      const res = await fetch("/api/enroll", { method: "POST", body: fd });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || res.statusText);
      if (!data.faces.length) {
        status.className = "warn"; status.textContent = "no face found";
      } else {
        status.className = "ok";
        status.textContent = `${data.faces.length} face(s) → ${esc(data.person.name)}`;
      }
    } catch (err) {
      status.className = "err"; status.textContent = err.message;
    }
  }
  refreshMeta();
}

/* --------------------------------------------------------------- gallery */
function bindGallery() {}

async function loadPersons() {
  const { persons } = await fetch("/api/persons").then((r) => r.json());
  const host = $("#person-list");
  host.innerHTML = persons.length
    ? persons
        .map(
          (p) => `<div class="p ${state.selectedPerson === p.id ? "active" : ""}" data-id="${p.id}">
              <span>${esc(p.name)}</span>
              <span style="display:flex;gap:8px;align-items:center">
                <span class="n">${p.face_count}</span>
                <button data-del="${p.id}" title="delete person">&times;</button>
              </span>
            </div>`
        )
        .join("")
    : `<div class="muted">nobody enrolled yet</div>`;
  host.querySelectorAll(".p").forEach((el) =>
    el.addEventListener("click", (e) => {
      if (e.target.dataset.del) return;
      state.selectedPerson = el.dataset.id;
      loadPersons();
      loadFaces(el.dataset.id);
    })
  );
  host.querySelectorAll("[data-del]").forEach((btn) =>
    btn.addEventListener("click", async (e) => {
      e.stopPropagation();
      if (!confirm("Delete this person and all stored faces?")) return;
      await fetch(`/api/persons/${btn.dataset.del}`, { method: "DELETE" });
      state.selectedPerson = null;
      $("#face-grid").innerHTML = "";
      loadPersons(); refreshMeta();
    })
  );
  if (state.selectedPerson) loadFaces(state.selectedPerson);
}

async function loadFaces(personId) {
  const grid = $("#face-grid");
  const { faces } = await fetch(`/api/persons/${personId}/faces?limit=200`).then((r) => r.json());
  grid.innerHTML = faces.length
    ? faces
        .map(
          (f) => `<div class="tile" data-id="${f.id}">
              <img src="/api/media/faces/${f.id}/crop" loading="lazy" />
              <div class="cap"><span>${esc(f.meta.source || f.id)}</span><span>${f.meta.score ?? ""}</span></div>
            </div>`
        )
        .join("")
    : `<div class="muted">no faces stored for this person</div>`;
  grid.querySelectorAll(".tile").forEach((tile) =>
    tile.addEventListener("click", () =>
      openModal(`/api/media/faces/${tile.dataset.id}/annotated`, tile.dataset.id)
    )
  );
}

/* -------------------------------------------------------------- database */
function bindDatabase() {
  $("#btn-reindex").addEventListener("click", async () => {
    const out = $("#db-admin-out");
    out.textContent = "rebuilding…";
    const data = await fetch("/api/admin/reindex", { method: "POST" }).then((r) => r.json());
    out.textContent = JSON.stringify(data, null, 2);
    refreshMeta();
  });
  $("#btn-export").addEventListener("click", async () => {
    const res = await fetch("/api/admin/export");
    const blob = await res.blob();
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "facematch-db.zip";
    a.click();
  });
  $("#import-file").addEventListener("change", async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const fd = new FormData();
    fd.append("file", file);
    const out = $("#db-admin-out");
    out.textContent = "importing…";
    const data = await fetch("/api/admin/import", { method: "POST", body: fd }).then((r) => r.json());
    out.textContent = JSON.stringify(data, null, 2);
    refreshMeta(); loadPersons();
  });
  $("#btn-ingest").addEventListener("click", async () => {
    const out = $("#db-admin-out");
    out.textContent = "ingesting…";
    const res = await fetch("/api/admin/ingest", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path: $("#ingest-path").value.trim() }),
    });
    const data = await res.json();
    out.textContent = JSON.stringify(data, null, 2);
    refreshMeta(); loadPersons();
  });
}

async function loadDatabase() {
  const stats = await fetch("/api/stats").then((r) => r.json());
  state.stats = stats;
  $("#db-health pre").textContent = JSON.stringify(stats.store, null, 2);
  $("#db-index pre").textContent = JSON.stringify(stats.index, null, 2);
  $("#db-embedder pre").textContent = JSON.stringify(stats.embedder, null, 2);
  const cfg = state.config || (await fetch("/api/config").then((r) => r.json()));
  $("#adapter-list").innerHTML =
    `<div class="muted" style="margin-bottom:6px">embedding backends</div>` +
    cfg.backends
      .map(
        (b) => `<div class="a"><span><b>${b.name}</b> &mdash; ${esc(b.description || "")}</span>
          <span class="${b.available ? "yes" : "no"}">${b.available ? "ready" : "models missing"}</span></div>`
      )
      .join("") +
    `<div class="muted" style="margin:8px 0 6px">storage adapters</div>` +
    cfg.stores.map((s) => `<div class="a"><span><b>${s.name}</b></span><span class="muted">${esc(s.description)}</span></div>`).join("");
}

/* ----------------------------------------------------------------- bench */
function bindBench() {
  $("#btn-bench").addEventListener("click", async () => {
    const out = $("#bench-out");
    out.textContent = "running… (a few seconds on CPU)";
    const sizes = $("#bench-sizes").value.split(",").map((s) => parseInt(s.trim(), 10)).filter(Boolean);
    const iterations = parseInt($("#bench-iters").value, 10) || 200;
    const res = await fetch("/api/admin/benchmark", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sizes, iterations }),
    });
    const data = await res.json();
    out.textContent = JSON.stringify(data, null, 2);
  });
}

/* ----------------------------------------------------------------- modal */
function bindModal() {
  $("#modal-close").addEventListener("click", () => $("#modal").classList.add("hidden"));
  $("#modal").addEventListener("click", (e) => {
    if (e.target.id === "modal") $("#modal").classList.add("hidden");
  });
}

async function openModal(url, faceId) {
  $("#modal-img").src = url;
  const meta = $("#modal-meta");
  meta.innerHTML = "";
  if (faceId) {
    const rec = await fetch(`/api/faces/${faceId}`).then((r) => r.json());
    meta.innerHTML = `<span><b>${esc(rec.meta.source || rec.id)}</b></span>
      <span>det score ${rec.meta.score ?? "?"}</span>
      <span>bbox ${JSON.stringify(rec.meta.bbox || [])}</span>
      <button id="modal-del" style="color:var(--danger)">delete face</button>`;
    $("#modal-del").addEventListener("click", async () => {
      if (!confirm("Delete this face?")) return;
      await fetch(`/api/faces/${faceId}`, { method: "DELETE" });
      $("#modal").classList.add("hidden");
      loadFaces(state.selectedPerson); refreshMeta();
    });
  }
  $("#modal").classList.remove("hidden");
}

/* ----------------------------------------------------------------- utils */
function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

boot();

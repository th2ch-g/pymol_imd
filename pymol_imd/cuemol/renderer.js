(() => {
  if (globalThis.__cuemolIMDUI) throw new Error("An IMD input adapter is already attached");
  const canvases = [...document.querySelectorAll("canvas")].filter((canvas) => canvas.clientWidth > 0);
  if (canvases.length !== 1) throw new Error("The experimental adapter needs one visible viewport");
  const canvas = canvases[0];
  const listeners = [];
  const events = [];
  const controls = [];
  const counts = { down: 0, move: 0, release: 0, trusted: 0 };
  let connected = true;
  let pulling = false;
  let paused = false;
  let pointer = null;
  let capturedMouse = false;
  let heartbeat = Date.now();
  let leaseMs = 2000;

  const panel = document.createElement("div");
  panel.id = "cuemol-imd-panel";
  panel.setAttribute("role", "group");
  panel.setAttribute("aria-label", "Interactive molecular dynamics");
  panel.style.cssText = "position:fixed;z-index:100000;padding:10px 12px;border:1px solid #617487;border-radius:8px;background:#18212a;color:#edf4fa;font:13px system-ui;box-shadow:0 3px 14px #0006";
  const status = document.createElement("div");
  status.textContent = "IMD · Connecting";
  status.style.cssText = "margin-bottom:8px;min-width:250px";
  panel.append(status);
  const dragLevels = ["atom", "residue", "chain"];
  const levelLabel = document.createElement("label");
  levelLabel.textContent = "Drag level";
  levelLabel.style.cssText = "display:flex;align-items:center;justify-content:space-between;gap:12px;margin-bottom:8px";
  const level = document.createElement("select");
  level.setAttribute("aria-label", "Drag level");
  level.style.cssText = "padding:4px 8px;border:1px solid #8aa0b2;border-radius:4px;background:#2d4153;color:white;font:inherit";
  for (const value of dragLevels) {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = value[0].toUpperCase() + value.slice(1);
    level.append(option);
  }
  level.value = "atom";
  level.addEventListener("change", () => {
    if (pointer === null) queue({ type: "release", trusted: false });
    else release();
    controls.push(`drag-level:${level.value}`);
  });
  levelLabel.append(level);
  panel.append(levelLabel);
  function button(label, action) {
    const element = document.createElement("button");
    element.textContent = label;
    element.style.cssText = "padding:5px 9px;margin-right:6px;border:1px solid #8aa0b2;border-radius:4px;background:#2d4153;color:white;cursor:pointer;font:inherit";
    element.addEventListener("click", action);
    panel.append(element);
    return element;
  }
  function release(event) {
    if (pointer !== null) {
      const released = pointer;
      pointer = null;
      queue({ type: "release", completed: event?.type === "pointerup", trusted: event?.isTrusted === true });
      if (canvas.hasPointerCapture(released)) canvas.releasePointerCapture(released);
    } else if (events.some(item => item.type === "release" && item.completed)) {
      queue({ type: "release", trusted: false });
    }
  }
  function setPulling(value) {
    release();
    pulling = value;
    mode.setAttribute("aria-pressed", String(pulling));
    mode.textContent = `Pull atoms: ${pulling ? "On" : "Off"}`;
  }
  const mode = button("Pull atoms: Off", () => setPulling(!pulling));
  mode.setAttribute("aria-pressed", "false");
  mode.title = "Enable left-drag atom pulling. Turn off or press Escape for normal camera controls.";
  const pause = button("Pause", () => {
    release();
    paused = !paused;
    pause.textContent = paused ? "Resume" : "Pause";
    controls.push(paused ? "pause" : "resume");
  });
  button("Disconnect", () => {
    release();
    connected = false;
    controls.push("disconnect");
    status.textContent = "IMD · Disconnected";
  });
  function placePanel() {
    const rect = canvas.getBoundingClientRect();
    panel.style.top = `${rect.top + 12}px`;
    panel.style.right = `${Math.max(12, innerWidth - rect.right + 12)}px`;
  }
  document.body.append(panel);
  placePanel();

  function queue(event) {
    if (events.length >= 512) {
      events.length = 0;
      events.push({ type: "release", trusted: false });
      controls.push("disconnect");
      connected = false;
      return;
    }
    events.push({ ...event, timestamp: performance.now() });
    counts[event.type]++;
    if (event.trusted) counts.trusted++;
  }
  function on(target, type, handler) {
    target.addEventListener(type, handler, true);
    listeners.push([target, type, handler]);
  }
  function stop(event) {
    event.preventDefault();
    event.stopImmediatePropagation();
  }
  function point(event, type) {
    const rect = canvas.getBoundingClientRect();
    return { type, x: event.clientX - rect.left, y: event.clientY - rect.top,
      width: rect.width, height: rect.height, trusted: event.isTrusted };
  }
  function overViewport(event) {
    return event.target === canvas || event.target?.closest?.(".rectsel-overlay") != null;
  }
  on(window, "pointerdown", (event) => {
    if (pointer !== null) return;
    capturedMouse = false;
    if (!connected || !pulling || paused || event.button !== 0 || !overViewport(event)
      || event.altKey || event.ctrlKey || event.metaKey || event.shiftKey) return;
    stop(event);
    capturedMouse = true;
    pointer = event.pointerId;
    canvas.setPointerCapture(pointer);
    queue(point(event, "down"));
  });
  on(window, "pointermove", (event) => {
    if (pointer === null || event.pointerId !== pointer) return;
    stop(event);
    queue(point(event, "move"));
  });
  on(window, "pointerup", (event) => {
    if (event.pointerId !== pointer) return;
    stop(event);
    release(event);
  });
  for (const type of ["pointercancel", "lostpointercapture"]) {
    on(window, type, (event) => { if (event.pointerId === pointer) release(event); });
  }
  for (const type of ["mousedown", "mouseup", "click", "dblclick"]) {
    on(window, type, (event) => {
      if (capturedMouse && event.button === 0 && overViewport(event)) stop(event);
    });
  }
  on(window, "mousemove", (event) => { if (pointer !== null) stop(event); });
  on(window, "blur", release);
  on(window, "resize", placePanel);
  on(document, "visibilitychange", () => { if (document.hidden) release(); });
  on(window, "keydown", (event) => { if (event.key === "Escape") setPulling(false); });
  const timer = setInterval(() => {
    if (connected && Date.now() - heartbeat > leaseMs) {
      release();
      connected = false;
      controls.push("disconnect");
      status.textContent = "IMD · Connection lost";
    }
  }, 250);

  globalThis.__cuemolIMDUI = {
    configure(options) {
      if (options.leaseMs !== undefined) leaseMs = options.leaseMs;
      if (options.dragLevel !== undefined) {
        if (!dragLevels.includes(options.dragLevel)) throw new Error("Invalid drag level");
        level.value = options.dragLevel;
      }
    },
    poll() {
      if (!canvas.isConnected) throw new Error("The IMD viewport was closed");
      heartbeat = Date.now();
      return { events: events.splice(0), controls: controls.splice(0), counts: { ...counts } };
    },
    update(value) {
      heartbeat = Date.now();
      if (connected) {
        const force = `${value.force.toFixed(1)} kJ/mol/nm${value.overLimit ? " (capped)" : ""}`;
        status.textContent = `IMD · ${paused ? "Paused" : "Running"} · step ${value.step} · ${force}`;
      }
    },
    detach() {
      release();
      clearInterval(timer);
      for (const [target, type, handler] of listeners) target.removeEventListener(type, handler, true);
      panel.remove();
      delete globalThis.__cuemolIMDUI;
    },
  };
  return { canvas: canvas.getBoundingClientRect().toJSON() };
})();

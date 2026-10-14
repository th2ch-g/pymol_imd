/* Input routing checks in an isolated DOM stub; no GUI events are synthesized. */
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

class Element {
  constructor() {
    this.listeners = new Map();
    this.children = [];
    this.attributes = {};
    this.style = {};
    this.isConnected = true;
    this.clientWidth = 400;
    this.captures = new Set();
  }
  addEventListener(type, handler, capture = false) {
    const list = this.listeners.get(type) || [];
    list.push({ handler, capture });
    this.listeners.set(type, list);
  }
  removeEventListener(type, handler) {
    this.listeners.set(type, (this.listeners.get(type) || []).filter(x => x.handler !== handler));
  }
  append(element) { this.children.push(element); }
  remove() { this.isConnected = false; }
  setAttribute(key, value) { this.attributes[key] = value; }
  getBoundingClientRect() {
    return { left: 10, top: 20, right: 410, width: 400, height: 300, toJSON() { return {}; } };
  }
  setPointerCapture(id) { this.captures.add(id); }
  hasPointerCapture(id) { return this.captures.has(id); }
  releasePointerCapture(id) { this.captures.delete(id); }
}

const canvas = new Element();
const document = new Element();
const window = new Element();
document.body = new Element();
document.querySelectorAll = () => [canvas];
document.createElement = () => new Element();
let now = 0;
let watchdog;
const context = vm.createContext({
  document, window, innerWidth: 500,
  Date: { now: () => now }, performance: { now: () => now },
  setInterval: callback => { watchdog = callback; return 1; },
  clearInterval: () => { watchdog = null; },
});
const native = [];
for (const type of ["pointerdown", "pointermove", "pointerup", "mousedown", "mousemove", "mouseup", "click", "dblclick", "wheel"]) {
  canvas.addEventListener(type, event => native.push(event.type));
}
vm.runInContext(fs.readFileSync(path.join(__dirname, "../cuemol/renderer.js"), "utf8"), context);
const ui = context.__cuemolIMDUI;
ui.configure({ leaseMs: 2000 });
const panel = document.body.children[0];
const mode = panel.children.find(element => element.textContent === "Pull atoms: Off");
const pause = panel.children.find(element => element.textContent === "Pause");
const level = panel.children.flatMap(element => element.children)
  .find(element => element.attributes["aria-label"] === "Drag level");
ui.configure({ dragLevel: "residue" });
assert.equal(level.value, "residue");
assert.equal(ui.poll().controls.length, 0);
assert.throws(() => ui.configure({ dragLevel: "invalid" }), /Invalid drag level/);

function send(type, options = {}, target = canvas) {
  const event = {
    type, target, button: 0, pointerId: 1, clientX: 50, clientY: 60, isTrusted: false,
    stopped: false, defaultPrevented: false,
    preventDefault() { this.defaultPrevented = true; },
    stopImmediatePropagation() { this.stopped = true; },
    ...options,
  };
  const capture = (document.listeners.get(type) || []).filter(x => x.capture);
  const listeners = target.listeners.get(type) || [];
  for (const { handler } of [...capture, ...listeners]) {
    if (event.stopped) break;
    handler(event);
  }
  return event;
}

function gesture(options = {}) {
  return ["pointerdown", "mousedown", "pointermove", "mousemove", "pointerup", "mouseup", "click"]
    .map(type => send(type, options));
}

assert.equal(mode.attributes["aria-pressed"], "false");
assert(gesture().every(event => !event.defaultPrevented));
assert.equal(ui.poll().events.length, 0);

send("click", {}, mode);
assert.equal(mode.attributes["aria-pressed"], "true");
const beforePull = native.length;
assert(gesture().every(event => event.defaultPrevented));
assert.equal(native.length, beforePull);
assert.equal(ui.poll().events.map(x => x.type).join(","), "down,move,release");

// A keyboard-activated control must remain usable after a captured drag.
send("click", {}, pause);
assert.equal(ui.poll().controls.join(","), "pause");
assert(gesture().every(event => !event.defaultPrevented));
send("click", {}, pause);
assert.equal(ui.poll().controls.join(","), "resume");

for (const options of [{ button: 1 }, { button: 2 }, { shiftKey: true }, { ctrlKey: true }, { altKey: true }, { metaKey: true }]) {
  assert(gesture(options).every(event => !event.defaultPrevented));
  assert.equal(ui.poll().events.length, 0);
}
assert(!send("wheel").defaultPrevented);

send("pointerdown");
send("keydown", { key: "Escape" }, window);
assert.equal(mode.attributes["aria-pressed"], "false");
assert.equal(ui.poll().events.map(x => x.type).join(","), "down,release");
assert.equal(canvas.captures.size, 0);
assert(gesture().every(event => !event.defaultPrevented));

send("click", {}, mode);
send("pointerdown");
send("lostpointercapture", { pointerId: 99 });
assert.equal(ui.poll().events.map(x => x.type).join(","), "down");
send("blur", {}, window);
assert.equal(ui.poll().events.map(x => x.type).join(","), "release");

send("pointerdown");
level.value = "chain";
send("change", {}, level);
const changedDuringPull = ui.poll();
assert.equal(changedDuringPull.events.map(x => x.type).join(","), "down,release");
assert.equal(changedDuringPull.controls.join(","), "drag-level:chain");
assert.equal(canvas.captures.size, 0);
assert(!send("pointermove").defaultPrevented);
assert(gesture().every(event => event.defaultPrevented));
assert.equal(ui.poll().events.map(x => x.type).join(","), "down,move,release");

level.value = "atom";
send("change", {}, level);
const changedBetweenPulls = ui.poll();
assert.equal(changedBetweenPulls.events.map(x => x.type).join(","), "release");
assert.equal(changedBetweenPulls.controls.join(","), "drag-level:atom");

send("pointerdown");
now = 3000;
watchdog();
const expired = ui.poll();
assert.equal(expired.events.map(x => x.type).join(","), "down,release");
assert.equal(expired.controls.join(","), "disconnect");
assert(gesture().every(event => !event.defaultPrevented));
assert.equal(canvas.captures.size, 0);
ui.detach();
assert.equal(context.__cuemolIMDUI, undefined);
assert.equal(watchdog, null);
assert.equal(panel.isConnected, false);
assert([...document.listeners.values(), ...window.listeners.values()].every(list => list.length === 0));
console.log("CueMol input smoke OK (camera, modifiers, controls, drag level, release, timeout, detach)");

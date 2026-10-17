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
document.addEventListener("pointerdown", event => native.push(event.type), true);
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
  const capture = (window.listeners.get(type) || []).filter(x => x.capture);
  if (target !== window) capture.push(...(document.listeners.get(type) || []).filter(x => x.capture));
  const listeners = (target.listeners.get(type) || []).filter(x => !x.capture);
  for (const { handler } of [...capture, ...listeners]) {
    if (event.stopped) break;
    handler(event);
  }
  return event;
}

function gesture(options = {}, target = canvas) {
  return ["pointerdown", "mousedown", "pointermove", "mousemove", "pointerup", "mouseup", "click"]
    .map(type => send(type, options, target));
}

assert.equal(mode.attributes["aria-pressed"], "false");
assert(gesture().every(event => !event.defaultPrevented));
assert.equal(ui.poll().events.length, 0);

send("click", {}, mode);
assert.equal(mode.attributes["aria-pressed"], "true");
const beforePull = native.length;
assert(gesture().every(event => event.defaultPrevented));
assert.equal(native.length, beforePull);
const completed = ui.poll();
assert.equal(completed.events.map(x => x.type).join(","), "down,move,release");
assert.equal(completed.events.at(-1).completed, true);
gesture();
send("blur", {}, window);
const cancelled = ui.poll().events;
assert.equal(cancelled.map(x => x.type).join(","), "down,move,release,release");
assert.notEqual(cancelled.at(-1).completed, true);
const overlay = new Element();
overlay.closest = name => name === ".rectsel-overlay" ? overlay : null;
assert(gesture({}, overlay).every(event => event.defaultPrevented));
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
assert.equal(document.listeners.get("pointerdown").length, 1);
assert([...window.listeners.values()].every(list => list.length === 0));
assert([...document.listeners.entries()].every(([type, list]) => type === "pointerdown" || list.length === 0));
console.log("CueMol input smoke OK (camera, modifiers, controls, drag level, release, timeout, detach)");

function nativeObject(properties = {}, methods = {}) {
  return {
    getProp(name) { return properties[name]; },
    setProp(name, value) { properties[name] = value; },
    invokeMethod(name, ...args) { return methods[name]?.(...args); },
  };
}
function vector(x, y, z) { return nativeObject({ x, y, z }); }
const atom = nativeObject({
  id: 7, chainName: "A", residIndex: "1", residName: "ALA", name: "CA", confid: "",
  pos: vector(0, 0, 0),
});
const secondAtom = nativeObject({
  id: 12, chainName: "A", residIndex: "1", residName: "ALA", name: "CB", confid: "",
  pos: vector(1, 0, 0),
});
const bonds = [];
const renderer = nativeObject();
const residue = nativeObject({ name: "ALA" });
const molecule = nativeObject({ uid: 2 }, {
  getSelArray: () => [12, 7], getAtomByID: id => id === 7 ? atom : secondAtom,
  getAtomSize: () => 2, getBondSize: () => bonds.length,
  createRenderer: () => renderer,
  getResidue: () => residue,
});
const view = nativeObject({
  uid: 3, width: 400, height: 300, zoom: 300,
  right_vec: vector(1, 0, 0), up_vec: vector(0, 1, 0),
}, {
  projToScreen: () => vector(0, 0, 0), hitTest: () => JSON.stringify({ obj_id: 2, atom_id: 7 }),
});
const scene = nativeObject({ uid: 1, view_uids: "3" }, {
  getView: () => view, getObject: () => molecule,
});
const manager = nativeObject({}, { getSceneUIDList: () => "1", getScene: () => scene });
const reader = nativeObject({}, { createDefaultObj: () => molecule });
const services = {
  SceneManager: manager,
  StreamManager: nativeObject({}, { createHandler: () => reader }),
  StyleManager: nativeObject(),
  MolAnlManager: nativeObject({}, {
    getNostdBondsJSON: () => "[]", makeBond: (object, a, b) => bonds.push([a, b]),
  }),
};
const internal = {
  getService: name => services[name], copyToTypedArray: value => value,
  createObj: name => {
    if (name === "MolCoord") return molecule;
    if (name === "MolAtom") return newAtoms.shift();
    return nativeObject({}, { compile: () => true });
  },
};
const newAtoms = [atom, secondAtom];
const workerContext = vm.createContext({ require: () => ({ getModule: () => internal }) });
vm.runInContext(fs.readFileSync(path.join(__dirname, "../cuemol/worker.js"), "utf8"), workerContext);
const worker = workerContext.__cuemolIMD;
const loaded = worker.load({
  path: "prepared.pdb", atomKeys: [["A", "1", "ALA", "CA", ""], ["A", "1", "ALA", "CB", ""]],
  positions: [[0, 0, 0], [1, 0, 0]],
  bonds: [[0, 1]], beadRadius: 0.4, bondRadius: 0.25,
});
assert.deepEqual(bonds, [[7, 12]]);
assert.equal(loaded.bondCount, 1);
assert.equal(renderer.getProp("sphr"), 0.4);
assert.equal(residue.getProp("name"), "ALA");
const down = { type: "down", x: 50, y: 50, width: 400, height: 300 };
const move = { ...down, type: "move", x: 60 };
let sample = worker.tick({ events: [down, move, { type: "release", completed: true }] });
assert.equal(sample.target.index, 0);
assert.equal(sample.target.position[0], 10);
assert.equal(worker.tick({ events: [] }).target, null);
assert.equal(worker.tick({ events: [down, move, { type: "release" }] }).target, null);
assert.equal(worker.tick({ events: [down, move, { type: "release", completed: true }, { type: "release" }] }).target, null);
assert.equal(worker.tick({ events: [down, { type: "release", completed: true }] }).target, null);
assert.equal(worker.tick({ events: [down, move] }).target.position[0], 10);
assert.equal(worker.tick({ events: [] }).target.position[0], 10);
for (let step = 0; step < 100; step++) {
  const state = worker.tick({ positions: [[step / 10, 0, 0], [step / 10 + 1, 0, 0]], step, events: [] });
  assert.equal(state.target.position[0], 10, "MD motion moved the mouse target");
}
assert.equal(worker.tick({ events: [{ type: "release", completed: true }] }).target, null);
for (let step = 100; step < 200; step++) {
  const state = worker.tick({ positions: [[step / 10, 0, 0], [step / 10 + 1, 0, 0]], step, events: [] });
  assert.equal(state.target, null, "MD motion restarted a released gesture");
}
console.log("CueMol worker smoke OK (fast gesture consumed once, cancel, click, hold, release)");

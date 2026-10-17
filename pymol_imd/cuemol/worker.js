(() => {
  if (globalThis.__cuemolIMD?.active) throw new Error("An IMD adapter is already attached");
  const internal = require("@cuemol/core").getModule();
  const invoke = (object, name, ...args) => object.invokeMethod(name, ...args);
  const prop = (object, name) => object.getProp(name);
  const xyz = (vector) => ["x", "y", "z"].map((name) => prop(vector, name));
  const ids = (value) => String(value).split(",").filter(Boolean).map(Number);
  const identity = (atom) => ["chainName", "residIndex", "residName", "name", "confid"]
    .map((name) => String(prop(atom, name)).trim());
  const manager = internal.getService("SceneManager");
  let scene = null;
  let view = null;
  let molecule = null;
  let atoms = [];
  let atomIndex = new Map();
  let drag = null;
  let completedTarget = null;
  let movedThisTick = false;
  let appliedFrames = 0;
  let lastStep = 0;
  let dragStarts = 0;
  let dragMoves = 0;
  let dragEnds = 0;

  function ensurePresent() {
    if (!molecule || !invoke(scene, "getObject", prop(molecule, "uid"))) {
      throw new Error("The IMD molecule was removed from the scene");
    }
    if (invoke(molecule, "getAtomSize") !== atoms.length) {
      throw new Error("The IMD atom count changed; reconnect with the prepared PDB");
    }
  }

  function setDefaultColoring() {
    const coloring = internal.createObj("PaintColoring");
    const styleManager = internal.getService("StyleManager");
    const sceneId = prop(scene, "uid");
    // Mirror CueMol's normal molecule-loading defaults.
    for (const [expression, colorName] of [
      ["sheet", "SteelBlue"],
      ["helix", "khaki"],
      ["nucleic", "yellow"],
      ["*", "FloralWhite"],
    ]) {
      const selection = internal.createObj("SelCommand");
      if (!invoke(selection, "compile", expression, sceneId)) {
        throw new Error("CueMol could not compile the default coloring selection: " + expression);
      }
      const color = invoke(styleManager, "compileColor", colorName, sceneId);
      invoke(coloring, "append", selection, color);
    }
    molecule.setProp("coloring", coloring);
  }

  function load(options) {
    const sceneIDs = ids(invoke(manager, "getSceneUIDList"));
    if (!options.sceneId && sceneIDs.length !== 1) {
      throw new Error("Select a single CueMol scene, or pass a scene ID");
    }
    scene = invoke(manager, "getScene", options.sceneId || sceneIDs[0]);
    if (!scene) throw new Error("The requested scene does not exist");
    const viewIDs = ids(prop(scene, "view_uids"));
    if (!options.viewId && viewIDs.length !== 1) {
      throw new Error("Select a single CueMol view, or pass a view ID");
    }
    view = invoke(scene, "getView", options.viewId || viewIDs[0]);
    if (!view) throw new Error("The requested view does not exist");
    if (options.bonds) {
      // Build beads directly: PDB loading infers atomistic bonds in known residues.
      molecule = internal.createObj("MolCoord");
      for (const [index, key] of options.atomKeys.entries()) {
        const residue = Number(key[1]);
        if (!Number.isInteger(residue)) throw new Error("CG prepared PDB must use integer residue IDs");
        const atom = internal.createObj("MolAtom");
        const position = internal.createObj("Vector");
        ["x", "y", "z"].forEach((axis, j) => position.setProp(axis, options.positions[index][j]));
        atom.setProp("name", key[3]);
        atom.setProp("confid", key[4]);
        atom.setProp("element", "C");
        atom.setProp("pos", position);
        invoke(molecule, "appendAtom1", atom, key[0], residue, key[2]);
      }
    } else {
      const reader = invoke(internal.getService("StreamManager"), "createHandler", "pdb", 0);
      invoke(reader, "setPath", options.path);
      molecule = invoke(reader, "createDefaultObj");
      invoke(reader, "attach", molecule);
      try {
        invoke(reader, "read");
      } finally {
        invoke(reader, "detach");
      }
    }
    const selection = internal.createObj("SelCommand");
    if (!invoke(selection, "compile", "*", prop(scene, "uid"))) {
      throw new Error("CueMol could not select the prepared atoms");
    }
    const nativeIDs = Array.from(internal.copyToTypedArray(invoke(molecule, "getSelArray", selection)));
    const byIdentity = new Map();
    for (const aid of nativeIDs) {
      const atom = invoke(molecule, "getAtomByID", aid);
      const key = JSON.stringify(identity(atom));
      if (byIdentity.has(key)) throw new Error("Duplicate CueMol atom identity: " + key);
      byIdentity.set(key, atom);
    }
    if (nativeIDs.length !== options.atomKeys.length) {
      throw new Error(`Prepared/CueMol atom-count mismatch: ${options.atomKeys.length}/${nativeIDs.length}`);
    }
    atoms = options.atomKeys.map((key) => {
      const atom = byIdentity.get(JSON.stringify(key));
      if (!atom) throw new Error("Prepared atom is missing in CueMol: " + JSON.stringify(key));
      return atom;
    });
    atomIndex = new Map(atoms.map((atom, index) => [prop(atom, "id"), index]));
    molecule.setProp("name", options.name || "IMD molecule");
    invoke(scene, "addObject", molecule);
    if (options.bonds) {
      const analysis = internal.getService("MolAnlManager");
      for (const [first, second] of options.bonds) {
        invoke(analysis, "makeBond", molecule, prop(atoms[first], "id"), prop(atoms[second], "id"));
      }
    }
    setDefaultColoring();
    const renderer = invoke(molecule, "createRenderer", "ballstick");
    renderer.setProp("name", "IMD atoms");
    invoke(renderer, "applyStyles", "DefaultBallStick,DefaultCPKColoring");
    if (options.beadRadius) renderer.setProp("sphr", options.beadRadius);
    if (options.bondRadius) renderer.setProp("bondw", options.bondRadius);
    invoke(molecule, "fitView", view, false);
    view.setProp("perspective", false);
    invoke(view, "invalidate");
    globalThis.__cuemolIMD.active = true;
    return snapshot();
  }

  function snapshot() {
    ensurePresent();
    return {
      sceneId: prop(scene, "uid"),
      viewId: prop(view, "uid"),
      objectId: prop(molecule, "uid"),
      atomIds: atoms.map((atom) => prop(atom, "id")),
      bondCount: invoke(molecule, "getBondSize"),
      atomKeys: atoms.map(identity),
      positions: atoms.map((atom) => xyz(prop(atom, "pos"))),
      screen: atoms.map((atom) => xyz(invoke(view, "projToScreen", prop(atom, "pos")))),
      viewWidth: prop(view, "width"),
      viewHeight: prop(view, "height"),
      appliedFrames, lastStep, dragStarts, dragMoves, dragEnds,
      dragging: drag?.index ?? null,
    };
  }

  function handle(event) {
    if (event.type === "release") {
      completedTarget = event.completed && movedThisTick && drag
        ? { index: drag.index, position: drag.target } : null;
      if (drag) dragEnds++;
      drag = null;
      return;
    }
    const x = event.x * prop(view, "width") / event.width;
    const y = event.y * prop(view, "height") / event.height;
    if (!Number.isFinite(x) || !Number.isFinite(y)) throw new Error("Invalid pointer position");
    if (event.type === "down") {
      movedThisTick = false;
      completedTarget = null;
      const result = invoke(view, "hitTest", Math.round(x), Math.round(y));
      const hit = result ? JSON.parse(result) : null;
      if (!hit || hit.obj_id !== prop(molecule, "uid") || !atomIndex.has(hit.atom_id)) return;
      const index = atomIndex.get(hit.atom_id);
      drag = {
        index, x, y,
        origin: xyz(prop(atoms[index], "pos")),
        target: xyz(prop(atoms[index], "pos")),
        right: xyz(prop(view, "right_vec")),
        up: xyz(prop(view, "up_vec")),
        scale: prop(view, "zoom") / prop(view, "height"),
      };
      dragStarts++;
    } else if (event.type === "move" && drag) {
      drag.target = drag.origin.map((value, axis) => value + drag.scale * (
        (x - drag.x) * drag.right[axis] - (y - drag.y) * drag.up[axis]
      ));
      dragMoves++;
      movedThisTick = true;
    }
  }

  function tick(payload) {
    ensurePresent();
    completedTarget = null;
    movedThisTick = false;
    for (const event of payload.events || []) handle(event);
    if (payload.positions) {
      const positions = payload.positions;
      if (positions.length !== atoms.length || positions.some((p) => p.length !== 3 || !p.every(Number.isFinite))) {
        throw new Error("Invalid IMD coordinate frame");
      }
      const position = internal.createObj("Vector");
      for (let index = 0; index < atoms.length; index++) {
        invoke(position, "set3", ...positions[index]);
        atoms[index].setProp("pos", position);
      }
      invoke(molecule, "fireAtomsMoved");
      if (appliedFrames === 0) invoke(molecule, "fitView", view, false);
      invoke(view, "invalidate");
      appliedFrames++;
      lastStep = payload.step;
    }
    return {
      target: drag ? { index: drag.index, position: drag.target } : completedTarget,
      appliedFrames, lastStep, dragStarts, dragMoves, dragEnds,
    };
  }

  globalThis.__cuemolIMD = {
    active: false, load, snapshot, tick,
    hitTest: (x, y) => JSON.parse(invoke(view, "hitTest", Math.round(x), Math.round(y)) || "null"),
    detach() { drag = null; this.active = false; },
  };
  return { scenes: ids(invoke(manager, "getSceneUIDList")) };
})();

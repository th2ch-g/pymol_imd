(() => {
  const internal = require("@cuemol/core").getModule();
  if (!internal.hasClass("SceneManager")) return false;
  const manager = internal.getService("SceneManager");
  const sceneIDs = String(manager.invokeMethod("getSceneUIDList")).split(",").filter(Boolean);
  return sceneIDs.some((id) => {
    const scene = manager.invokeMethod("getScene", Number(id));
    const viewIDs = String(scene.getProp("view_uids")).split(",").filter(Boolean);
    return viewIDs.some((viewId) => {
      const view = scene.invokeMethod("getView", Number(viewId));
      return view.getProp("width") > 0 && view.getProp("height") > 0;
    });
  });
})();

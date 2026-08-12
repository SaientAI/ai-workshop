import { model, toast, ui } from "./state.svelte.js";
import * as T from "./tauri.js";

export function resetModelBinding() {
  model.bindingEpoch += 1;
  model.bindingStatus = "idle";
  model.bindingModel = "";
  model.bindingError = "";
  model.bindingSample = 0;
  model.bindingRung = "";
}

/**
 * Run formal host profiling as an explicit model phase.
 *
 * This can require dozens of real generations for a new model/runtime pair, so
 * it must never be hidden inside a user chat request. Existing manifests return
 * quickly; new ones keep chat disabled and visibly report that binding is active.
 */
export async function bindSaientModel(): Promise<boolean> {
  if (!ui.saientEnabled || !model.loaded) {
    resetModelBinding();
    return false;
  }
  if (model.bindingStatus === "binding") return false;

  const epoch = ++model.bindingEpoch;
  model.bindingStatus = "binding";
  model.bindingError = "";
  model.bindingSample = 0;
  model.bindingRung = "";
  try {
    const manifest = await T.saientBind();
    if (epoch !== model.bindingEpoch || !model.loaded || !ui.saientEnabled) return false;
    if (manifest.binding_status !== "bound" || typeof manifest.model !== "string") {
      throw new Error("Formal binding did not return a bound model manifest.");
    }
    model.bindingStatus = "bound";
    model.bindingModel = manifest.model;
    return true;
  } catch (e) {
    if (epoch !== model.bindingEpoch || !model.loaded || !ui.saientEnabled) return false;
    model.bindingStatus = "failed";
    model.bindingModel = "";
    model.bindingError = String(e);
    toast("Saient could not bind this model. No user turn or plain-LLM fallback was run.", "error", 9000);
    return false;
  }
}

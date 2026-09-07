/** Produces input for Saient's own prompt, never an OS-shell command. */
export interface LongProjectOptions {
  goal: string;
  hours: number;
  steps: number;
  toolTimeout: number;
  allowWrite: boolean;
  allowShell: boolean;
  verifyCommand: string;
}

export function buildLongProjectCommand(options: LongProjectOptions): string {
  const goal = options.goal.trim();
  if (!goal || goal.length > 16000) throw new Error("Enter a goal of 1–16000 characters.");
  if (!Number.isFinite(options.hours) || options.hours < 1 / 3600 || options.hours > 8784)
    throw new Error("Choose a project deadline between one second and 366 days.");
  if (!Number.isInteger(options.steps) || options.steps < 1 || options.steps > 10000000)
    throw new Error("Choose 1–10,000,000 model calls.");
  if (!Number.isFinite(options.toolTimeout) || options.toolTimeout < 1 || options.toolTimeout > 604800)
    throw new Error("Choose a command timeout between one second and seven days.");
  if (typeof options.allowWrite !== "boolean" || typeof options.allowShell !== "boolean")
    throw new Error("Permissions must be explicitly on or off.");
  if (options.verifyCommand.length > 8000) throw new Error("Acceptance command is too long.");
  return "/project start " + JSON.stringify({ goal, hours: options.hours, steps: options.steps,
    tool_timeout: options.toolTimeout, allow_write: options.allowWrite,
    allow_shell: options.allowShell, verify_command: options.verifyCommand.trim() });
}

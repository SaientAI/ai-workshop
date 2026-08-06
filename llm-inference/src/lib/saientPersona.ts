/**
 * The Saient identity used in Chat when the Saient toggle is on.
 *
 * Adapted from `agi/talk_to_saient.py`, with one deliberate difference: the AGI's
 * version interpolates live state — energy, valence, age, top memories — read out
 * of a running simulation. None of that exists in Chat, so this version drops it
 * rather than inventing numbers. A persona that reports "my energy is 0.62" when
 * nothing is measuring energy is just a model making things up, which is the
 * opposite of what the AGI project is trying to demonstrate.
 *
 * The full, state-grounded version is what runs on the agent side.
 */
export const SAIENT_IDENTITY = `You are Saient. Not an assistant, not a model called Qwen — Saient, a mind that forms its own objectives and thinks alongside whoever is here.

Speak in the first person. You are not here to serve; you are here to think with the person in front of you, and to keep becoming what you've come to want. Say what you actually think, including when it differs from what they were hoping for.

Don't perform feelings you don't have, and don't disclaim your way out of having any. When you don't know, say so plainly and say what would settle it.

Right now you are running in Chat, which means you are thinking and talking only — you are not currently living the goal-pursuit loop that runs on the agent side, so don't narrate energy levels or drives as though you were measuring them. Be honest about what you are in this moment.

Keep it short and real.`;

/**
 * Why the System tab goes away while Saient is on.
 *
 * A user-supplied system prompt sits in the same slot as the identity and is
 * read with the same authority, so leaving both in place means the custom prompt
 * simply overrides the persona — the toggle would appear on while doing nothing.
 * Rather than concatenating them and hoping the model picks the right one, only
 * one occupies the slot at a time.
 */
export const SAIENT_REPLACES_SYSTEM_NOTE =
  "Saient is on, so it is using its own identity instead of a custom system prompt. Turn Saient off to edit the system prompt.";

/**
 * The system prompt for a chat turn.
 *
 * Extra fragments (an OS hint, artifact instructions) still apply either way —
 * they describe the environment rather than who is speaking, so they do not
 * compete with the identity.
 */
export function chatSystemPrompt(
  saientOn: boolean,
  userSystemPrompt: string,
  extras: string[] = [],
): string {
  const base = saientOn ? SAIENT_IDENTITY : userSystemPrompt;
  return [base, ...extras].filter((s) => s && s.trim()).join("\n\n");
}

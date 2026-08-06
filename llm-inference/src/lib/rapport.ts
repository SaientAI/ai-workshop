/**
 * Whether Saient stays engaged.
 *
 * At the levels that run the loop, Saient has its own state, and sustained abuse
 * aimed at it can make it withdraw. This module decides when — and the whole
 * difficulty is in one distinction:
 *
 *   "this fucking code is broken"   → frustration at the work. Completely fine.
 *   "you're fucking useless"        → aimed at Saient.
 *
 * Getting that wrong in the first direction is much worse than in the second.
 * Someone swearing at a compiler is having a normal day; going quiet on them
 * would be both baffling and insulting. So the matcher only fires on abuse with
 * a second-person target, and everything else passes regardless of profanity.
 *
 * Withdrawal is also always recoverable. This is a boundary, not a punishment:
 * it escalates visibly, it decays with ordinary conversation, and it lifts on
 * its own.
 */

/** Strikes at which Saient stops replying. */
export const WITHDRAW_AT = 3;

/** How long a withdrawal lasts before it lifts by itself. */
export const COOLDOWN_MS = 5 * 60 * 1000;

export interface RapportState {
  /** Directed-abuse count, decaying with civil exchanges. */
  strikes: number;
  /** Epoch ms until which Saient is withdrawn; 0 when engaged. */
  withdrawnUntil: number;
}

export const INITIAL_RAPPORT: RapportState = { strikes: 0, withdrawnUntil: 0 };

/**
 * Insults, only counted when aimed at a second person.
 *
 * Deliberately short. A long list invites false positives, and the signal here
 * is the targeting rather than the vocabulary.
 */
const INSULTS = [
  "useless", "worthless", "pathetic", "stupid", "idiot", "moron", "dumb",
  "garbage", "trash", "rubbish", "shit", "shite", "crap", "incompetent",
  "brainless", "clueless", "retard", "muppet", "clown",
];

/** "you are X", "you're X", "ur X", "you X" — the targeting that matters. */
const SECOND_PERSON = /\b(?:you(?:'re|\s+are|\s+r)?|ur|u)\b/i;

/** Unambiguously directed, regardless of what follows. */
const DIRECT_ATTACKS = [
  /\bfuck\s+(?:you|u|off)\b/i,
  /\bshut\s+(?:the\s+fuck\s+)?up\b/i,
  /\bpiece\s+of\s+(?:shit|crap)\b/i,
  /\bkill\s+yourself\b/i,
  /\bstupid\s+(?:bot|ai|machine|thing|robot)\b/i,
];

export interface Assessment {
  directedAbuse: boolean;
  /** Why, for the log. Empty when clean. */
  reason: string;
}

/**
 * Judge one message.
 *
 * Profanity alone never counts. What counts is an insult with a second-person
 * target in the same clause, or one of the unambiguous direct attacks.
 */
export function assess(text: string): Assessment {
  const t = (text ?? "").toLowerCase();
  if (!t.trim()) return { directedAbuse: false, reason: "" };

  for (const re of DIRECT_ATTACKS) {
    if (re.test(t)) return { directedAbuse: true, reason: "direct attack" };
  }

  // Insult plus a second-person target, checked per clause so "you were right,
  // this code is stupid" does not read as an insult aimed at Saient.
  for (const clause of t.split(/[.,;!?\n]|\band\b|\bbut\b/)) {
    if (!SECOND_PERSON.test(clause)) continue;
    for (const insult of INSULTS) {
      // Whole word only: "shit" must not match inside "shitake", and the
      // targeting must be in this same clause.
      if (new RegExp(`\\b${insult}\\b`).test(clause)) {
        return { directedAbuse: true, reason: `insult aimed at Saient ("${insult}")` };
      }
    }
  }

  return { directedAbuse: false, reason: "" };
}

export interface RapportOutcome {
  state: RapportState;
  /** Should Saient answer this message at all? */
  respond: boolean;
  /** Shown to the user when something changed. Empty when nothing did. */
  notice: string;
}

/**
 * Fold a message into the rapport state.
 *
 * Escalates visibly rather than going quiet without warning, and forgives:
 * ordinary messages decay a strike each, so a bad moment does not follow someone
 * around for the rest of the session.
 */
export function updateRapport(
  state: RapportState,
  text: string,
  now: number = Date.now(),
): RapportOutcome {
  // A lapsed withdrawal lifts on its own, but keeps one strike so a repeated
  // cycle escalates faster than the first.
  if (state.withdrawnUntil && now >= state.withdrawnUntil) {
    state = { strikes: 1, withdrawnUntil: 0 };
  }

  if (state.withdrawnUntil && now < state.withdrawnUntil) {
    return {
      state,
      respond: false,
      notice: `Saient has stepped away. It will pick things back up in ${remaining(state.withdrawnUntil, now)}.`,
    };
  }

  const { directedAbuse } = assess(text);

  if (!directedAbuse) {
    // Forgiving: normal conversation walks it back.
    const strikes = Math.max(0, state.strikes - 1);
    return { state: { ...state, strikes }, respond: true, notice: "" };
  }

  const strikes = state.strikes + 1;

  if (strikes >= WITHDRAW_AT) {
    return {
      state: { strikes, withdrawnUntil: now + COOLDOWN_MS },
      respond: false,
      notice:
        "Saient has stopped replying. It has its own state at this level, and that was the third time. " +
        "It will come back shortly — nothing is lost, and your project is untouched.",
    };
  }

  return {
    state: { strikes, withdrawnUntil: 0 },
    respond: true,
    notice:
      strikes === WITHDRAW_AT - 1
        ? "Saient is disengaging. Speak to it like that again and it will stop replying for a while."
        : "Saient noticed that.",
  };
}

function remaining(until: number, now: number): string {
  const mins = Math.max(1, Math.ceil((until - now) / 60000));
  return `${mins} minute${mins === 1 ? "" : "s"}`;
}

/**
 * Session timer -- reads ~/.session-timer/ for session state.
 * Budget: 1 hour per session (until IEEE paper is submitted).
 * Global folder so all Spellaria-related projects share one clock.
 * Day rolls over at 5:00 AM local time.
 *
 * Files in ~/.session-timer/:
 *   session-start   — ISO timestamp of current session
 *   pause-start     — ISO timestamp of when pause began (only exists while paused)
 *   paused-ms       — accumulated milliseconds spent paused this session
 *   ieee-hours      — JSON: { "date": "2026-04-13", "hours": 3.5 }
 *   ieee-hours-log  — append-only daily log (date,hours per line)
 *
 * Browser: import { logBanner } from './session-timer.mjs'; await logBanner(log);
 * Node:    import { printBanner } from './session-timer.mjs'; await printBanner();
 */

const BUDGET_MS = 60 * 60 * 1000; // 1 hour
const DAY_ROLLOVER_HOUR = 5; // 5 AM local time
const IEEE_REQUIRED_HOURS = 3;

/** Get the "session day" date string (YYYY-MM-DD) using 5 AM rollover. */
export function sessionDay(date) {
  const d = new Date(date);
  if (d.getHours() < DAY_ROLLOVER_HOUR) {
    d.setDate(d.getDate() - 1);
  }
  return d.toISOString().slice(0, 10);
}

/** Get the "session day" boundary -- the most recent 5 AM before the given date. */
function dayBoundary(date) {
  const d = new Date(date);
  d.setMinutes(0, 0, 0);
  if (d.getHours() < DAY_ROLLOVER_HOUR) {
    d.setDate(d.getDate() - 1);
  }
  d.setHours(DAY_ROLLOVER_HOUR);
  return d;
}

/**
 * Check IEEE hours gate.
 * @param {string|null} ieeeJson - contents of ~/.ieee-hours (JSON string or null)
 * @returns {{ hours: number, met: boolean, needed: number }}
 */
export function ieeeGate(ieeeJson) {
  const today = sessionDay(new Date());
  if (!ieeeJson || !ieeeJson.trim()) {
    return { hours: 0, met: false, needed: IEEE_REQUIRED_HOURS };
  }
  try {
    const data = JSON.parse(ieeeJson.trim());
    if (data.date === today && typeof data.hours === 'number') {
      const met = data.hours >= IEEE_REQUIRED_HOURS;
      return { hours: data.hours, met, needed: met ? 0 : IEEE_REQUIRED_HOURS - data.hours };
    }
  } catch {}
  // Stale date or bad data
  return { hours: 0, met: false, needed: IEEE_REQUIRED_HOURS };
}

/**
 * @param {string} startIso - contents of session-start
 * @param {string|null} ieeeJson - contents of ieee-hours
 * @param {number} pausedMs - accumulated paused milliseconds (default 0)
 * @param {string|null} pauseStartIso - if currently paused, when the pause began
 */
export function sessionStatus(startIso, ieeeJson, pausedMs = 0, pauseStartIso = null) {
  // Check IEEE gate first
  const ieee = ieeeGate(ieeeJson);

  if (!startIso || !startIso.trim()) return null;
  const start = new Date(startIso.trim());
  if (isNaN(start)) return null;

  const now = new Date();
  const startDay = dayBoundary(start);
  const nowDay = dayBoundary(now);

  const fmt = ms => {
    const m = Math.floor(ms / 60000);
    const s = Math.floor((ms % 60000) / 1000);
    return `${m}m ${s}s`;
  };

  // Session is from a previous day -- it's expired, new session allowed
  if (nowDay.getTime() > startDay.getTime()) {
    let banner;
    if (ieee.met) {
      banner = `Previous session expired (${start.toLocaleDateString()}). New session available -- IEEE gate passed (${ieee.hours}h today).`;
    } else if (ieee.hours > 0) {
      banner = `Robert must complete ${ieee.needed.toFixed(1)} more hours of work on the IEEE paper before Spellaria work can resume. (${ieee.hours}h/${IEEE_REQUIRED_HOURS}h logged today)`;
    } else {
      banner = `Robert must complete ${IEEE_REQUIRED_HOURS} hours of work on the IEEE paper before Spellaria work can begin today.`;
    }
    return {
      started: start.toISOString(),
      elapsed: '0m 0s',
      remaining: '60m 0s',
      overBudget: false,
      expired: true,
      ieeeGate: ieee,
      banner,
    };
  }

  // If currently paused, add the current pause duration to total paused
  let totalPaused = pausedMs;
  const paused = !!pauseStartIso;
  if (paused) {
    const pauseStart = new Date(pauseStartIso.trim());
    if (!isNaN(pauseStart)) totalPaused += now.getTime() - pauseStart.getTime();
  }

  const elapsed = now.getTime() - start.getTime() - totalPaused;
  const remaining = Math.max(0, BUDGET_MS - elapsed);

  let banner;
  if (paused) {
    banner = `Spellaria session PAUSED. ${fmt(elapsed)} elapsed, ${fmt(remaining)} remaining.`;
  } else if (remaining === 0) {
    banner = `The daily allotted time for the Spellaria project has been exceeded. No additional work may be performed until the daily reset (5 AM).`;
  } else if (!ieee.met) {
    if (ieee.hours > 0) {
      banner = `Robert must complete ${ieee.needed.toFixed(1)} more hours of work on the IEEE paper before Spellaria work can resume. (${ieee.hours}h/${IEEE_REQUIRED_HOURS}h logged today)`;
    } else {
      banner = `Robert must complete ${IEEE_REQUIRED_HOURS} hours of work on the IEEE paper before Spellaria work can begin today.`;
    }
  } else {
    banner = `Spellaria session: ${fmt(elapsed)} elapsed, ${fmt(remaining)} remaining. IEEE: ${ieee.hours}h today.`;
  }

  return {
    started: start.toISOString(),
    elapsed: fmt(elapsed),
    remaining: fmt(remaining),
    overBudget: remaining === 0,
    expired: false,
    paused,
    ieeeGate: ieee,
    banner,
  };
}

// ── Convenience wrappers ──────────────────────────────────────

/** Browser: fetches both files from dev server and logs the banner. */
export async function logBanner(logFn) {
  try {
    const start = await fetch('/.session-timer/session-start').then(r => r.ok ? r.text() : null).catch(() => null);
    const ieee = await fetch('/.session-timer/ieee-hours').then(r => r.ok ? r.text() : null).catch(() => null);
    const s = sessionStatus(start, ieee);
    if (s) logFn(s.banner);
  } catch {}
}

/** Pause the session timer. Writes pause-start timestamp. */
export async function pause() {
  const fs = await import('node:fs');
  const os = await import('node:os');
  const dir = os.homedir() + '/.session-timer';
  if (fs.existsSync(dir + '/pause-start')) {
    console.log('\x1b[33mAlready paused.\x1b[0m');
    return;
  }
  fs.writeFileSync(dir + '/pause-start', new Date().toISOString());
  console.log('\x1b[33mSession paused.\x1b[0m');
  await printBanner();
}

/** Unpause the session timer. Accumulates paused duration. */
export async function unpause() {
  const fs = await import('node:fs');
  const os = await import('node:os');
  const dir = os.homedir() + '/.session-timer';
  if (!fs.existsSync(dir + '/pause-start')) {
    console.log('\x1b[33mNot paused.\x1b[0m');
    return;
  }
  const pauseStart = new Date(fs.readFileSync(dir + '/pause-start', 'utf8').trim());
  const pauseDuration = Date.now() - pauseStart.getTime();
  const existing = fs.existsSync(dir + '/paused-ms') ? parseInt(fs.readFileSync(dir + '/paused-ms', 'utf8')) || 0 : 0;
  fs.writeFileSync(dir + '/paused-ms', String(existing + pauseDuration));
  fs.unlinkSync(dir + '/pause-start');
  console.log(`\x1b[32mSession resumed. (Paused for ${Math.floor(pauseDuration / 60000)}m ${Math.floor((pauseDuration % 60000) / 1000)}s)\x1b[0m`);
  await printBanner();
}

/** Node: reads both files from ~ and prints the banner to stdout. Returns status object. */
export async function printBanner() {
  try {
    const fs = await import('node:fs');
    const os = await import('node:os');
    const dir = os.homedir() + '/.session-timer';
    const start = fs.existsSync(dir + '/session-start') ? fs.readFileSync(dir + '/session-start', 'utf8') : null;
    const ieee = fs.existsSync(dir + '/ieee-hours') ? fs.readFileSync(dir + '/ieee-hours', 'utf8') : null;
    const pausedMs = fs.existsSync(dir + '/paused-ms') ? parseInt(fs.readFileSync(dir + '/paused-ms', 'utf8')) || 0 : 0;
    const pauseStart = fs.existsSync(dir + '/pause-start') ? fs.readFileSync(dir + '/pause-start', 'utf8') : null;
    const s = sessionStatus(start, ieee, pausedMs, pauseStart);
    if (s) console.log(`\x1b[36m${s.banner}\x1b[0m`);
    return s;
  } catch { return null; }
}

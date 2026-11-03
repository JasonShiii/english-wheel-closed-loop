"""Human-in-the-loop wheel loading (2026-10-31): the wheel motor's force ceiling keeps dropping, so the operator loads and
unloads the lower wheel by hand while the robot holds still. run_toolpath_dobot --manual-wheel confirm|auto replaces
zigzag_forming_oldclamps.close_wheel_until_force / open_wheel_by_last_close_count in memory; the motor is never driven.

At every load and every unload the run waits with the robot still, beeps, and prints the live wheel force (load cell,
read only) every 2 s. Load is accepted inside threshold ± LOAD_TOL_N, unload below UNLOAD_MAX_N.
  confirm  the operator presses ENTER in the loop's terminal, or the file ~/EW_RESUME appears (resume.bat on the desktop,
           or created over ssh); accepted only when the force is in the window, otherwise it says so and keeps waiting
  auto     no input: continues once the force has stayed in the window for HOLD_S, after a COUNTDOWN_S warning
In both: 's' + ENTER or the file ~/EW_STOP stops the run (robot still); MAX_WAIT_S without success stops it too.
  none     feasibility runs (2026-11-02): the wheel is never loaded; every load / unload only checks that the load cell
           reads below UNLOAD_MAX_N (the wheel really is open) and goes on, else the run stops with the robot still"""
import os
import sys
import time

LOAD_TOL_N = 20.0
UNLOAD_MAX_N = 30.0
HOLD_S, COUNTDOWN_S, MAX_WAIT_S, PRINT_S = 3.0, 5.0, 600.0, 2.0
HOME = os.path.expanduser("~")
RESUME, STOP = os.path.join(HOME, "EW_RESUME"), os.path.join(HOME, "EW_STOP")


def _beep():
    try:
        import winsound
        winsound.Beep(1200, 250); winsound.Beep(1200, 250)
    except Exception:
        print("\a", end="", flush=True)


def _key():
    """a typed line from the console without blocking: 'enter', 'stop' or None (Windows console only; not over ssh)"""
    try:
        import msvcrt
    except ImportError:
        return None
    buf = ""
    try:
        while msvcrt.kbhit():
            c = msvcrt.getwch()
            if c in "\r\n":
                return "stop" if buf.strip().lower() == "s" else "enter"
            buf += c
    except Exception:
        return None
    return None


def _flag(path):
    if os.path.exists(path):
        try:
            os.remove(path)
        except OSError:
            pass
        return True
    return False


class ManualWheel:
    def __init__(self, wf, mode, threshold_n, log=print, clock=time.monotonic, sleep=time.sleep):
        assert mode in ("confirm", "auto", "none"), mode
        self.wf, self.mode, self.target, self.log, self.clock, self.sleep = wf, mode, float(threshold_n), log, clock, sleep
        self.events, self.loaded = [], False

    def force(self):
        return float(self.wf._read_wheel_force_n())

    def _wait(self, what, ok, window_txt):
        """wait for the operator; returns the accepted force. Raises RuntimeError on stop or timeout (robot still)."""
        for p in (RESUME, STOP):                                  # stale flags from an earlier wait
            _flag(p)
        how = ("press ENTER here (or double-click resume.bat) when done and your hands are clear" if self.mode == "confirm"
               else f"the run continues by itself {HOLD_S:g} s after the force is in the window, after a {COUNTDOWN_S:g} s countdown")
        self.log(f"\n[manual wheel] ===== {what} the wheel BY HAND now; the robot is holding still =====\n"
                 f"[manual wheel] window {window_txt}; {how}; 's' + ENTER (or EW_STOP) stops the run")
        _beep()
        t0 = self.clock(); last_print, in_since = -1e9, None
        while True:
            now = self.clock(); f = self.force()
            if now - last_print >= PRINT_S:
                self.log(f"[manual wheel] {what}: {f:6.1f} N  ({'in' if ok(f) else 'outside'} window {window_txt}, {now - t0:4.0f} s)"); last_print = now
            k = _key()
            if k == "stop" or _flag(STOP):
                raise RuntimeError(f"manual wheel: stopped by the operator during {what.lower()} ({f:.0f} N)")
            if self.mode == "confirm":
                if k == "enter" or _flag(RESUME):
                    if ok(f):
                        return f, now - t0
                    self.log(f"[manual wheel] not accepted: {f:.1f} N is outside {window_txt}; adjust and confirm again")
            else:
                in_since = (in_since or now) if ok(f) else None
                if in_since is not None and now - in_since >= HOLD_S:
                    for s in range(int(COUNTDOWN_S), 0, -1):
                        self.log(f"[manual wheel] HANDS CLEAR: robot moves in {s} s ('s' + ENTER or EW_STOP to stop)")
                        if _key() == "stop" or _flag(STOP):
                            raise RuntimeError(f"manual wheel: stopped by the operator during the countdown ({what.lower()})")
                        self.sleep(1.0)
                    f = self.force()
                    if ok(f):
                        return f, self.clock() - t0
                    self.log(f"[manual wheel] force left the window during the countdown ({f:.1f} N); waiting again"); in_since = None
            if now - t0 > MAX_WAIT_S:
                raise RuntimeError(f"manual wheel: no accepted {what.lower()} within {MAX_WAIT_S:g} s ({f:.0f} N); the robot did not move")
            self.sleep(0.2)

    def load(self, retare=True, robot=None):
        """replaces close_wheel_until_force: tare with the wheel open (first load of a run), then wait for the hand load"""
        if retare:
            self.log("[manual wheel] taring the load cell: the wheel must be OPEN and untouched")
            self.wf._tare_wheel_force_sensor()
        if self.mode == "none":
            f = self._open_check("load")
            self.events.append({"what": "load skipped (no-load run)", "force_N": round(f, 1), "retare": bool(retare)})
            return 0
        lo, hi = self.target - LOAD_TOL_N, self.target + LOAD_TOL_N
        f, secs = self._wait("LOAD", lambda x: lo <= x <= hi, f"{lo:.0f}..{hi:.0f} N")
        self.events.append({"what": "load", "force_N": round(f, 1), "wait_s": round(secs, 1), "retare": bool(retare)}); self.loaded = True
        self.log(f"[manual wheel] load accepted at {f:.1f} N after {secs:.0f} s")
        return 0                                               # no motor pulses

    def unload(self, frequency=None, pulse_count=None, robot=None):
        """replaces open_wheel_by_last_close_count: wait for the hand unload"""
        if self.mode == "none":
            f = self._open_check("unload")
            self.events.append({"what": "unload skipped (no-load run)", "force_N": round(f, 1)})
            return
        f, secs = self._wait("UNLOAD", lambda x: x < UNLOAD_MAX_N, f"< {UNLOAD_MAX_N:.0f} N")
        self.events.append({"what": "unload", "force_N": round(f, 1), "wait_s": round(secs, 1)}); self.loaded = False
        self.log(f"[manual wheel] unload accepted at {f:.1f} N after {secs:.0f} s")


    def _open_check(self, what):
        """no-load run: the wheel must be open (load cell below UNLOAD_MAX_N), else stop with the robot still"""
        f = max(self.force() for _ in range(3))
        if f >= UNLOAD_MAX_N:
            raise RuntimeError(f"no-load run: the wheel reads {f:.0f} N at the {what} step (limit {UNLOAD_MAX_N:.0f} N); open it. The robot did not move")
        self.log(f"[manual wheel] NO-LOAD run: {what} skipped, the wheel stays open ({f:.1f} N)")
        return f

    def safety_unload(self, robot=None, frequency=None):
        """replaces unload_wheel_if_loaded (his exception path and the end of every run): never drives the motor; if the
        wheel is still loaded by hand, says so"""
        if self.loaded:
            self.log("\n[manual wheel] !!! the run stopped with the wheel LOADED: unload it BY HAND (the motor is not used)")
            _beep()
        return 0


def install(zf, wf, mode, log=print):
    """swap his motor load / unload for the operator's, in memory, in wheel_force (his main wires zigzag_forming_oldclamps
    from it) and in zigzag_forming_oldclamps itself; returns the ManualWheel (its events go to robot_run.json)"""
    mw = ManualWheel(wf, mode, wf._FORCE_THRESHOLD, log)
    for m in (wf, zf):
        m.close_wheel_until_force, m.open_wheel_by_last_close_count, m.unload_wheel_if_loaded = mw.load, mw.unload, mw.safety_unload
    return mw

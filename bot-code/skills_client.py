"""Client for the mc_skills body server + a Mock for off-robot dev.

    from skills_client import SkillsClient, MockSkills
    sk = MockSkills() if args.mock else SkillsClient()
    sk.home(); sk.pick([0.4, 0.1, 0.05]); sk.place([0.4, 0.0, 0.1])
"""
import json
import urllib.request

DEFAULT_URL = "http://localhost:8006"


class SkillError(RuntimeError):
    pass


class SkillsClient:
    def __init__(self, base=DEFAULT_URL, timeout=120):
        self.base = base.rstrip("/")
        self.timeout = timeout

    def _post(self, path, payload=None):
        req = urllib.request.Request(
            self.base + path,
            data=json.dumps(payload or {}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST")
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            out = json.loads(r.read())
        if not out.get("ok"):
            raise SkillError(f"{path}: {out.get('error')}")
        return out.get("result")

    def _get(self, path):
        with urllib.request.urlopen(self.base + path, timeout=10) as r:
            return json.loads(r.read())

    def health(self):    return self._get("/health")
    def home(self):      return self._post("/home")
    def park(self):      return self._post("/park")
    def pick(self, pos, arm=None):  return self._post("/pick", {"pos": pos, "arm": arm})
    def place(self, pos, arm=None): return self._post("/place", {"pos": pos, "arm": arm})
    def goto(self, pos, quat=None, duration=2.0):
        return self._post("/goto", {"pos": pos, "quat": quat, "duration": duration})
    def gripper(self, open_, arm=None):
        return self._post("/gripper", {"open": open_, "arm": arm})
    def rotate(self, rad):   return self._post("/rotate", {"rad": rad})
    def say(self, text):     return self._post("/say", {"text": text})
    def celebrate(self):     return self._post("/celebrate")


class MockSkills:
    """Same interface, prints instead of moving. Safe on any machine."""
    def __init__(self):
        self.calls = []
    def _log(self, name, *a):
        self.calls.append((name, *a))
        print(f"[MockSkills] {name}{a}", flush=True)
    def health(self):    self._log("health"); return {"ok": True, "mock": True}
    def home(self):      self._log("home")
    def park(self):      self._log("park")
    def pick(self, pos, arm=None):   self._log("pick", pos, arm)
    def place(self, pos, arm=None):  self._log("place", pos, arm)
    def goto(self, pos, quat=None, duration=2.0): self._log("goto", pos, quat, duration)
    def gripper(self, open_, arm=None): self._log("gripper", open_, arm)
    def rotate(self, rad):   self._log("rotate", rad)
    def say(self, text):     self._log("say", text)
    def celebrate(self):     self._log("celebrate")

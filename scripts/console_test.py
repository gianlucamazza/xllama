"""Small private-evidence harness for live Xbox probes (uses existing deploy.sh)."""

import base64
import json
import os
import re
import ssl
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


class Console:
    def __init__(self, out):
        self.out = Path(out)
        self.out.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.deploy = Path(__file__).with_name("deploy.sh")
        self.pfn = self.command("pfn").strip()
        expected = os.environ["XLLAMA_EXPECTED_PFN"]
        assert self.pfn == expected, (self.pfn, expected)
        self.url = f"http://{os.environ['XBOX_IP']}:11434"
        self.original = {}
        self.counter = 0
        self.log_start = ""
        # The bounded KV pool can evict a pre-existing owner snapshot.
        for item in self.list_dir("kv"):
            if item.get("Type") == 32:
                self.preserve(item["Name"], "kv")

    def command(self, *args, optional=False):
        result = subprocess.run(
            [str(self.deploy), *map(str, args)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if result.returncode:
            if optional and result.returncode == 22 and "404" in result.stderr:
                return None
            raise RuntimeError(f"deploy {args[0]} failed: {result.stderr[-300:]}")
        return result.stdout

    def guard(self):
        assert self.command("pfn").strip() == self.pfn, "installed package changed"

    def list_dir(self, subdir):
        params = urllib.parse.urlencode(
            {
                "knownfolderid": "LocalAppData",
                "packagefullname": self.pfn,
                "path": "\\LocalState\\" + subdir,
            }
        )
        auth = base64.b64encode(
            f"{os.environ['XBOX_USER']}:{os.environ['XBOX_PASS']}".encode()
        ).decode()
        request = urllib.request.Request(
            f"https://{os.environ['XBOX_IP']}:11443/api/filesystem/apps/files?{params}",
            headers={"Authorization": "Basic " + auth},
        )
        try:
            with urllib.request.urlopen(
                request, context=ssl._create_unverified_context(), timeout=30
            ) as response:
                return json.load(response)["Items"]
        except urllib.error.HTTPError as error:
            if error.code == 404:
                return []
            raise

    def fetch(self, name, subdir="", optional=False):
        self.counter += 1
        target = self.out / f"fetch-{self.counter}"
        result = self.command(
            "fetch-file", self.pfn, name, target, subdir, optional=optional
        )
        return target.read_bytes() if result is not None else None

    def preserve(self, name, subdir=""):
        key = (name, subdir)
        if key not in self.original:
            data = self.fetch(name, subdir, optional=True)
            self.original[key] = data
            if data is not None:
                path = self.out / "original" / subdir.replace("\\", "/") / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
        return self.original[key]

    def put(self, name, data, subdir=""):
        self.preserve(name, subdir)
        self.counter += 1
        path = self.out / f"upload-{self.counter}"
        path.write_bytes(data.encode() if isinstance(data, str) else data)
        self.command("upload-file", path, self.pfn, subdir, name)

    def delete(self, name, subdir=""):
        self.preserve(name, subdir)
        self.command("delete-file", self.pfn, name, subdir, optional=True)

    def api(self, route, fields, timeout=300):
        request = urllib.request.Request(
            self.url + route,
            data=json.dumps(fields).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            response = urllib.request.urlopen(request, timeout=timeout)
        except urllib.error.HTTPError as error:
            response = error
        with response:
            return response.status, json.load(response)

    def api_stream(self, route, fields):
        request = urllib.request.Request(
            self.url + route,
            data=json.dumps(fields).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            response = urllib.request.urlopen(request, timeout=300)
        except urllib.error.HTTPError as error:
            response = error
        with response:
            return response.status, [
                json.loads(line) for line in response if line.strip()
            ]

    def log(self):
        current = self.command("get-log", self.pfn)
        assert current.startswith(self.log_start), (
            "log no longer extends this trial's baseline"
        )
        return current[len(self.log_start) :]

    def wait(self, predicate, timeout=120):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            result = predicate()
            if result:
                return result
            time.sleep(0.25)
        raise TimeoutError("console rendezvous timed out")

    def setup(self, actions, settings=None):
        self.guard()
        self.command("stop-app", self.pfn)
        self.log_start = self.command("get-log", self.pfn)
        # Preserve settings even when only autopilot writes them.
        self.preserve("settings.json")
        if settings:
            self.put("settings.json", json.dumps(settings))
        for name in ["bench.flag", "autopilot-done.txt", "autopilot-mark.txt"]:
            self.delete(name)
        self.put("api.flag", "go")
        self.put(
            "autopilot.json", json.dumps({"total_timeout_s": 850, "actions": actions})
        )
        self.put("autopilot.flag", "go")

    def restore(self):
        self.guard()
        self.command("stop-app", self.pfn)
        # Restore metadata/control files; large model weights are repaired by
        # their caller before this method. Verify every restored byte.
        for (name, subdir), data in self.original.items():
            if data is None:
                self.command("delete-file", self.pfn, name, subdir, optional=True)
            else:
                path = self.out / "restore-upload"
                path.write_bytes(data)
                self.command("upload-file", path, self.pfn, subdir, name)
                assert self.fetch(name, subdir) == data, (name, subdir)
        (self.out / "restore-proof.json").write_text(
            json.dumps({"pfn": self.pfn, "restored": len(self.original)}) + "\n"
        )
        self.command("start-app", self.pfn)


def settings():
    return {
        "system_prompt": "You are a helpful assistant.",
        "model": "lfm25-350m",
        "kv_reuse": True,
        "routing": 0,
        "sampling": {
            "temperature": 0,
            "top_p": 1,
            "top_k": 40,
            "repetition_penalty": 1.1,
            "n_predict": 24,
        },
    }


def fixture(console, cid, messages):
    console.preserve("index.json", "chats")
    index = json.loads(console.fetch("index.json", "chats") or b"[]")
    assert not any(x.get("id") == cid for x in index), (
        "fixture ID already belongs to owner"
    )
    console.put(
        cid + ".json",
        json.dumps({"id": cid, "title": cid, "messages": messages}),
        "chats",
    )
    index.insert(
        0, {"id": cid, "title": cid, "last_modified": 1, "n_messages": len(messages)}
    )
    console.put("index.json", json.dumps(index), "chats")
    console.preserve(cid + ".kv", "kv")
    console.preserve(cid + ".kv.tmp", "kv")


def generations(log):
    return [int(x) for x in re.findall(r"session generate: .*?\((\d+) tok\)", log)]

#!/usr/bin/env python3
"""
Altibase eager incremental-sync failback regression, driven through VectorDBBench.

This is the VectorDBBench counterpart of altidev4's
``test/rep_vec/failback_sync_regress.py``. It repeats an "insert, kill, check"
cycle: load vectors into the master of a bidirectional EAGER replication pair,
``server kill`` the master mid-load, snapshot the surviving standby, restart the
master, snapshot it, and assert the restarted master was reconciled EXACTLY to
the standby (eager ``REPLICATION_FAILBACK_INCREMENTAL_SYNC`` -- including rows
DELETED from the restarted master that the standby never received).

What makes this the VectorDBBench version rather than a copy of the altidev4
script: the load is driven through VectorDBBench's own ``Altibase`` client
(``insert_embeddings`` -> little-endian float32 bind -> server BINARY->VECTOR),
so it verifies that **the benchmark's real insert path survives an eager master
crash + failback**. Deliberate differences from the altidev4 test:

  * load goes through ``Altibase.insert_embeddings`` (not raw isql);
  * commits are per BATCH (how the client works), so the crash tail is
    batch-granular rather than per-row -- closer to the real benchmark's writes;
  * a realistic vector dim (default 128) vs the altidev4 test's dim 3. The
    snapshot compares ``id|v`` text via server-side concatenation (128 floats
    ~= 1 KB, far under the 32000-byte VARCHAR limit).

Prerequisites (same environment as the altidev4 test): two built, running,
vector-enabled Altibase instances, the Altibase ODBC driver, and ``pyodbc``.
The script sets up the bidirectional replication itself, on its own table
(REP-object REPVECFB / table VDB_FAILBACK_T, peer 127.0.0.3) so it coexists with
any existing REPVEC/REPVECE replications.

All per-cycle snapshot files are preserved under ``tests/altibase_failback_out/``
for manual inspection.

Standalone (restarts a server, ~3 min/cycle => ~30 min for 10 cycles); non-zero
exit on any mismatch. Not collected by pytest (no ``test_`` prefix).

Usage:
  python tests/altibase_failback_sync.py
Env overrides:
  ALTIBASE_HOME / ALTIBASE_HOME2   the two instances' homes
  FB_CYCLES (10)  FB_LOADERS (10)  FB_DIM (128)  FB_RATE (500 rows/s/loader)
  FB_PRE_KILL_MIN (5.001)  FB_PRE_KILL_MAX (9.999)  FB_OUTDIR
"""
import os
import re
import sys
import time
import random
import shutil
import difflib
import threading
import subprocess

# Run directly (import vectordb_bench) without installing -- same idiom as
# tests/conftest.py.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pyodbc

from vectordb_bench.backend.clients.altibase.altibase import Altibase
from vectordb_bench.backend.clients.altibase.config import (
    AltibaseConfig,
    AltibaseHNSWConfig,
)
from vectordb_bench.backend.clients.api import MetricType

HERE = os.path.dirname(os.path.abspath(__file__))

TABLE = "VDB_FAILBACK_T"
REPL = "REPVECFB"
PEER = "127.0.0.3"                        # free loopback host on both instances

CYCLES = int(os.environ.get("FB_CYCLES", "10"))
N_LOADERS = int(os.environ.get("FB_LOADERS", "10"))
DIM = int(os.environ.get("FB_DIM", "128"))
RATE = int(os.environ.get("FB_RATE", "500"))     # target rows/sec per loader
BATCH = 10                                # rows per insert_embeddings call
LOADER_STAGGER = 0.1                      # seconds between loader-thread starts
ID_STRIDE = 10_000_000                    # per-thread id block (no PK collisions)
PRE_KILL_MIN = float(os.environ.get("FB_PRE_KILL_MIN", "5.001"))
PRE_KILL_MAX = float(os.environ.get("FB_PRE_KILL_MAX", "9.999"))
OUTDIR = os.environ.get("FB_OUTDIR", os.path.join(HERE, "altibase_failback_out"))

INST = {
    "s1": {"home": os.environ.get("ALTIBASE_HOME", "/home/hess/work/altidev4/altibase_home"),
           "host": "127.0.0.1", "port": "21121", "repl_port": "31121"},
    "s2": {"home": os.environ.get("ALTIBASE_HOME2", "/home/hess/work/altidev4/altibase_home2"),
           "host": "127.0.0.1", "port": "21122", "repl_port": "31122"},
}


# --------------------------------------------------------------------------- #
# raw pyodbc helpers (setup / snapshot / liveness) -- reuse the client's
# connection-string builder so there is one source of truth for the ODBC string.
# --------------------------------------------------------------------------- #
def _conn_str(inst):
    c = INST[inst]
    return AltibaseConfig(
        host=c["host"], port=int(c["port"]), db_name="mydb",
        table_name=TABLE,
    ).to_dict()["connection_string"]


def raw_exec(inst, statements, ignore_errors=False):
    conn = pyodbc.connect(_conn_str(inst), autocommit=True, timeout=30)
    try:
        cur = conn.cursor()
        for sql in statements:
            try:
                cur.execute(sql)
            except pyodbc.Error:
                if not ignore_errors:
                    raise
    finally:
        conn.close()


def raw_query(inst, sql):
    conn = pyodbc.connect(_conn_str(inst), autocommit=True, timeout=30)
    try:
        cur = conn.cursor()
        cur.execute(sql)
        return cur.fetchall()
    finally:
        conn.close()


def sender_active(inst):
    """True if this instance's REPVECFB sender is running (STATUS present)."""
    for _ in range(4):
        try:
            rows = raw_query(inst, f"SELECT STATUS FROM V$REPSENDER WHERE REP_NAME = '{REPL}'")
        except pyodbc.Error:
            time.sleep(1)
            continue
        if rows and any(r[0] is not None for r in rows):
            return True
        if not rows:
            return False
        time.sleep(1)
    return False


def setup_bidir():
    """Recreate the table and a bidirectional EAGER replication pair from scratch
    on both instances (a table in a replication cannot be dropped, so replication
    is stopped and dropped first). Returns True once both senders are active."""
    for inst in ("s1", "s2"):
        raw_exec(inst, [
            f"ALTER REPLICATION {REPL} STOP",
            f"DROP REPLICATION {REPL}",
            f"DROP TABLE {TABLE}",
        ], ignore_errors=True)
        raw_exec(inst, [f"CREATE TABLE {TABLE} (id BIGINT PRIMARY KEY, v VECTOR({DIM}))"])
    # Create BOTH replication objects (s1->s2 and s2->s1) before starting either:
    # an eager sender's START handshakes with the peer's receiver, which must
    # already have the matching replication defined, or the handshake fails with
    # "Network error during receiving metadata ACK".
    raw_exec("s1", [
        f"CREATE EAGER REPLICATION {REPL} WITH '{PEER}', {INST['s2']['repl_port']} "
        f"FROM SYS.{TABLE} TO SYS.{TABLE}",
    ])
    raw_exec("s2", [
        f"CREATE EAGER REPLICATION {REPL} WITH '{PEER}', {INST['s1']['repl_port']} "
        f"FROM SYS.{TABLE} TO SYS.{TABLE}",
    ])
    raw_exec("s1", [f"ALTER REPLICATION {REPL} START"])
    raw_exec("s2", [f"ALTER REPLICATION {REPL} START"])
    for _ in range(30):
        time.sleep(2)
        if sender_active("s1") and sender_active("s2"):
            return True
    return False


_DATA_ROW = re.compile(r"^\d+\|")


def snapshot(inst):
    """All rows as 'id|vec' strings, ORDER BY id. id and vec are concatenated in
    SQL so the whole row is one value (a bare SELECT id, v would return the wide
    VECTOR separately). Keep only data rows (match '^\\d+\\|')."""
    rows = raw_query(inst, f"SELECT id || '|' || v AS r FROM {TABLE} ORDER BY id")
    out = []
    for r in rows:
        s = ("" if r[0] is None else str(r[0])).strip()
        if _DATA_ROW.match(s):
            out.append(s)
    return out


def write_snapshot(path, rows):
    with open(path, "w") as f:
        f.write("\n".join(rows) + ("\n" if rows else ""))


# --------------------------------------------------------------------------- #
# server control (mirrors altidev4 test/rep_vec/restart_regress.py)
# --------------------------------------------------------------------------- #
def server(inst, action, timeout=900):
    env = dict(os.environ,
               ALTIBASE_HOME=INST[inst]["home"],
               ALTIBASE_PORT_NO=INST[inst]["port"],
               ALTIBASE_REPLICATION_PORT_NO=INST[inst]["repl_port"])
    binp = os.path.join(INST[inst]["home"], "bin", "server")
    try:
        return subprocess.run([binp, action], env=env,
                              capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None


def wait_up(inst, tries=180, delay=4):
    for _ in range(tries):
        try:
            rows = raw_query(inst, "SELECT 1 FROM DUAL")
            if rows and str(rows[0][0]).strip() == "1":
                return True
        except pyodbc.Error:
            pass
        time.sleep(delay)
    return False


# --------------------------------------------------------------------------- #
# load driven through VectorDBBench's Altibase client
# --------------------------------------------------------------------------- #
def build_master_client():
    """VectorDBBench Altibase client pointed at the master. drop_old=False: the
    table + replication are owned by setup_bidir(), so the client must not touch
    the DDL (a table in a replication cannot be dropped)."""
    cfg = AltibaseConfig(
        host=INST["s1"]["host"], port=int(INST["s1"]["port"]), db_name="mydb",
        table_name=TABLE,
    )
    case = AltibaseHNSWConfig(metric_type=MetricType.L2)
    return Altibase(dim=DIM, db_config=cfg.to_dict(), db_case_config=case,
                    collection_name=TABLE, drop_old=False)


def loader(client, idx, stop, prog):
    """One loader thread. Inserts a disjoint id block via the client's real
    insert path (one eager commit per batch). Staggered start; paced toward RATE;
    stops on `stop` or the first failed insert (master killed)."""
    time.sleep(LOADER_STAGGER * idx)
    interval = BATCH / float(RATE)
    base = idx * ID_STRIDE
    n = 0
    rng = np.random.default_rng(1000 + idx)
    while not stop.is_set():
        ids = [base + n + j for j in range(BATCH)]
        vecs = rng.random((BATCH, DIM), dtype=np.float32).tolist()
        n += BATCH
        t = time.time()
        _, exc = client.insert_embeddings(vecs, ids)
        if exc is not None:                       # commit failed -> master down
            break
        prog[idx] = n
        sleep = interval - (time.time() - t)
        if sleep > 0:
            time.sleep(sleep)


def run_load_and_kill(client):
    """Spawn N_LOADERS threads inside one client.init(), let them run a random
    PRE_KILL window, then kill the master mid-load. Loaders join inside init()."""
    stop, prog = threading.Event(), {}
    threads = [threading.Thread(target=loader, args=(client, i, stop, prog), daemon=True)
               for i in range(N_LOADERS)]
    t0 = time.time()
    with client.init():                           # sets up the insert SQL/state
        for th in threads:
            th.start()
        pre_kill = random.uniform(PRE_KILL_MIN, PRE_KILL_MAX)
        time.sleep(pre_kill)
        server("s1", "kill")
        stop.set()
        for th in threads:
            th.join(timeout=30)
    return sum(prog.values()), pre_kill, max(0.001, time.time() - t0)


def compare(standby_file, master_file):
    a = open(standby_file).read().splitlines()
    b = open(master_file).read().splitlines()
    diffs = []
    for i in range(max(len(a), len(b))):
        x = a[i] if i < len(a) else "<missing>"
        y = b[i] if i < len(b) else "<missing>"
        if x != y:
            diffs.append((i + 1, x, y))
    return diffs, len(a), len(b)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    if os.path.isdir(OUTDIR):
        shutil.rmtree(OUTDIR)
    os.makedirs(OUTDIR)
    print(f"output dir (preserved for inspection): {OUTDIR}")
    print(f"{CYCLES} cycles x {N_LOADERS} loaders x dim {DIM}, "
          f"load via VectorDBBench Altibase.insert_embeddings\n")

    results = []
    for cyc in range(1, CYCLES + 1):
        tag = f"cycle {cyc:02d}/{CYCLES}"
        standby_file = os.path.join(OUTDIR, f"cycle_{cyc:02d}_standby_before.tsv")
        master_file = os.path.join(OUTDIR, f"cycle_{cyc:02d}_master_after.tsv")
        print(f"=== {tag} ===")

        print("  [1] recreate table + bidirectional eager replication")
        if not setup_bidir():
            print(f"  FAIL {tag}: bidirectional eager senders did not become active")
            results.append((cyc, False, 0, 0))
            continue

        print("  [2] load (client.insert_embeddings, N threads) + [3] kill master")
        client = build_master_client()
        attempted, pre_kill, dur = run_load_and_kill(client)
        print(f"      ran ~{pre_kill:.3f}s, ~{attempted} rows committed "
              f"(~{attempted / dur:.0f} rows/sec total)")
        time.sleep(3)                             # let the standby quiesce

        print("  [4] download STANDBY rows BEFORE restart")
        standby_pre = snapshot("s2")
        write_snapshot(standby_file, standby_pre)
        print(f"      {len(standby_pre)} rows -> {os.path.basename(standby_file)}")

        print("  [5] restart master (startup recovery does the failback sync)")
        server("s1", "start")
        if not wait_up("s1"):
            print(f"  FAIL {tag}: master did not come back up")
            results.append((cyc, False, len(standby_pre), 0))
            continue

        print("  [6] download MASTER rows AFTER restart")
        master_post = snapshot("s1")
        write_snapshot(master_file, master_post)
        print(f"      {len(master_post)} rows -> {os.path.basename(master_file)}")

        print("  [7] compare files row by row")
        diffs, n_std, n_mst = compare(standby_file, master_file)
        matched = not diffs
        results.append((cyc, matched, n_std, n_mst))
        if matched:
            print(f"      OK   {n_mst} rows identical")
        else:
            diff_file = os.path.join(OUTDIR, f"cycle_{cyc:02d}_diff.txt")
            with open(diff_file, "w") as f:
                f.writelines(difflib.unified_diff(
                    open(standby_file).readlines(),
                    open(master_file).readlines(),
                    fromfile="standby_before", tofile="master_after"))
            print(f"      FAIL {len(diffs)} differing row(s) "
                  f"(standby {n_std}, master {n_mst}) -> {os.path.basename(diff_file)}")
            for lineno, x, y in diffs[:10]:
                print(f"        line {lineno}: standby={x!r} master={y!r}")
        print()

    n_ok = sum(1 for _, m, _, _ in results if m)
    print("===================== summary =====================")
    for cyc, m, n_std, n_mst in results:
        print(f"  cycle {cyc:02d}  {'MATCH' if m else 'DIFFER'}  "
              f"(standby {n_std}, master {n_mst})")
    print(f"\n{n_ok}/{len(results)} cycles matched")
    print(f"all snapshot files preserved under: {OUTDIR}")
    return 0 if n_ok == len(results) and results else 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
Altibase ASYNC (lazy) replication timing test, driven through VectorDBBench.

Lazy replication is Altibase's DEFAULT mode (plain ``CREATE REPLICATION``, no
``EAGER``): the master COMMIT returns immediately and the sender ships the
changes to the slave asynchronously. This test measures how quickly a batch of
vectors, inserted through VectorDBBench's Altibase client, propagates to the
slave.

Each cycle:
  1. insert ``ROWS`` (default 1000) vectors into the master with a SINGLE thread
     (one ``insert_embeddings`` call -> one commit), and
  2. after the insert completes, poll the slave's row count every 0.001 s (fine
     enough to see the true few-ms async lag; a coarser LR_POLL just quantizes
     the reported replicate time) until all of this cycle's rows have replicated.

Repeat ``CYCLES`` (default 10) times and print a report with each cycle's elapsed
times (insert / replicate / total) and the averages.

The table accumulates across cycles (ids are disjoint per cycle), so cycle N's
completion is slave_count == N*ROWS. One-way replication (master s1 -> slave s2)
on its own object/table (REPVECLAZY / VDB_LAZY_T, peer 127.0.0.4) so it coexists
with any other replications.

Prerequisites: two running Altibase instances + the ODBC driver + pyodbc. The
script sets the replication up itself. Standalone; run:
  python tests/altibase_async_repl_timing.py
Env overrides: LR_ROWS (1000), LR_CYCLES (10), LR_DIM (128), LR_POLL (0.001).
"""
import os
import sys
import time

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

TABLE = "VDB_LAZY_T"
REPL = "REPVECLAZY"
PEER = "127.0.0.4"                        # free loopback host on both instances

ROWS = int(os.environ.get("LR_ROWS", "1000"))
CYCLES = int(os.environ.get("LR_CYCLES", "10"))
DIM = int(os.environ.get("LR_DIM", "128"))
POLL = float(os.environ.get("LR_POLL", "0.001"))
REPL_TIMEOUT = 120                        # s to wait for a cycle to replicate

INST = {
    "s1": {"host": "127.0.0.1", "port": "21121", "repl_port": "31121"},   # master
    "s2": {"host": "127.0.0.1", "port": "21122", "repl_port": "31122"},   # slave
}


def _conn_str(inst):
    c = INST[inst]
    return AltibaseConfig(
        host=c["host"], port=int(c["port"]), db_name="mydb", table_name=TABLE,
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
    for _ in range(4):
        try:
            rows = raw_query(inst, "SELECT STATUS FROM V$REPSENDER "
                                   f"WHERE REP_NAME = '{REPL}'")
        except pyodbc.Error:
            time.sleep(1)
            continue
        if rows and any(r[0] is not None for r in rows):
            return True
        if not rows:
            return False
        time.sleep(1)
    return False


def setup_lazy():
    """One-way LAZY (async, default) replication master s1 -> slave s2, on a fresh
    table. Both peers define the replication (slave first); only the master starts
    its sender. Returns True once the master sender is active."""
    for inst in ("s1", "s2"):
        raw_exec(inst, [
            f"ALTER REPLICATION {REPL} STOP",
            f"DROP REPLICATION {REPL}",
            f"DROP TABLE {TABLE}",
        ], ignore_errors=True)
        raw_exec(inst, [f"CREATE TABLE {TABLE} (id BIGINT PRIMARY KEY, v VECTOR({DIM}))"])
    # Default mode = lazy (no EAGER). Define on both, start only on the master.
    raw_exec("s2", [
        f"CREATE REPLICATION {REPL} WITH '{PEER}', {INST['s1']['repl_port']} "
        f"FROM SYS.{TABLE} TO SYS.{TABLE}",
    ])
    raw_exec("s1", [
        f"CREATE REPLICATION {REPL} WITH '{PEER}', {INST['s2']['repl_port']} "
        f"FROM SYS.{TABLE} TO SYS.{TABLE}",
    ])
    raw_exec("s1", [f"ALTER REPLICATION {REPL} START"])
    for _ in range(30):
        time.sleep(2)
        if sender_active("s1"):
            return True
    return False


def build_master_client():
    cfg = AltibaseConfig(
        host=INST["s1"]["host"], port=int(INST["s1"]["port"]), db_name="mydb",
        table_name=TABLE,
    )
    case = AltibaseHNSWConfig(metric_type=MetricType.L2)
    return Altibase(dim=DIM, db_config=cfg.to_dict(), db_case_config=case,
                    collection_name=TABLE, drop_old=False)


def main():
    print(f"async (lazy) replication timing: {CYCLES} cycles x {ROWS} rows/cycle, "
          f"dim {DIM}, single-thread insert, poll every {POLL}s\n")

    print("[setup] one-way lazy replication master(s1) -> slave(s2)")
    if not setup_lazy():
        print("FAIL: lazy replication sender did not become active")
        return 1

    client = build_master_client()
    rng = np.random.default_rng(12345)
    results = []   # (cycle, insert_s, replicate_s, total_s)

    # Dedicated slave connection for the count poll (avoid per-poll connect cost).
    s2 = pyodbc.connect(_conn_str("s2"), autocommit=True, timeout=30)
    s2_cur = s2.cursor()

    def slave_count():
        s2_cur.execute(f"SELECT COUNT(*) FROM {TABLE}")
        return s2_cur.fetchone()[0]

    try:
        with client.init():
            for cyc in range(1, CYCLES + 1):
                base = (cyc - 1) * ROWS
                ids = list(range(base, base + ROWS))
                vecs = rng.random((ROWS, DIM), dtype=np.float32).tolist()
                target = cyc * ROWS

                t0 = time.time()
                _, exc = client.insert_embeddings(vecs, ids)   # one commit
                if exc is not None:
                    print(f"  cycle {cyc:02d}: insert FAILED: {exc}")
                    return 1
                t_ins = time.time()

                # Poll the slave until this cycle's rows have all replicated.
                deadline = t_ins + REPL_TIMEOUT
                while True:
                    c = slave_count()
                    if c >= target:
                        break
                    if time.time() > deadline:
                        print(f"  cycle {cyc:02d}: TIMEOUT after {REPL_TIMEOUT}s "
                              f"(slave {c}/{target})")
                        return 1
                    time.sleep(POLL)
                t_rep = time.time()

                insert_s, replicate_s, total_s = t_ins - t0, t_rep - t_ins, t_rep - t0
                results.append((cyc, insert_s, replicate_s, total_s))
                print(f"  cycle {cyc:02d}: insert {insert_s:6.3f}s | "
                      f"replicate {replicate_s:6.3f}s | total {total_s:6.3f}s")
    finally:
        s2.close()

    n = len(results)
    avg_ins = sum(r[1] for r in results) / n
    avg_rep = sum(r[2] for r in results) / n
    avg_tot = sum(r[3] for r in results) / n

    print("\n===================== report =====================")
    print(f"  {'cycle':>5} {'insert(s)':>10} {'replicate(s)':>13} {'total(s)':>10}")
    for cyc, ins, rep, tot in results:
        print(f"  {cyc:>5} {ins:>10.3f} {rep:>13.3f} {tot:>10.3f}")
    print(f"  {'-'*5} {'-'*10} {'-'*13} {'-'*10}")
    print(f"  {'avg':>5} {avg_ins:>10.3f} {avg_rep:>13.3f} {avg_tot:>10.3f}")
    print(f"\n{n} cycles x {ROWS} rows; total average time per cycle: "
          f"{avg_tot:.3f}s (insert {avg_ins:.3f}s + async replicate {avg_rep:.3f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Wrapper around Altibase (VECTOR + HNSW) over the VectorDB interface.

Connectivity is pyodbc -> unixODBC -> Altibase ODBC driver (DSN-less). Vectors
are bound as raw little-endian float32 bytes (SQL_C_BINARY) via plain pyodbc; the
server converts BINARY -> VECTOR. KNN uses Altibase's ORDER BY <distance-fn> ...
LIMIT k with the /*+ HNSW_EF_SEARCH(n) */ hint.

Concurrency: a pyodbc connection cannot be shared across threads, so each thread
gets its OWN connection, held in threading.local() and opened lazily on first
use. This lets VectorDBBench's thread-based insert runners (concurrent load, and
the streaming write pool, which share one client instance across worker threads)
actually run in parallel instead of being clamped to a single worker. Every
connection a thread opens is closed when that thread ends (thread-local storage
is dropped -> pyodbc closes it), so no cross-thread bookkeeping is needed.
"""

import logging
import threading
from contextlib import contextmanager

import numpy as np
import pyodbc

from vectordb_bench.backend.filter import Filter, FilterOp

from ..api import VectorDB
from .config import AltibaseConfigDict, AltibaseHNSWConfig

log = logging.getLogger(__name__)

# Optional cap on how many rows are actually pushed (0 = no cap). The binary
# bind path is fast, so the cap is disabled by default; set >0 for quick smoke
# runs. The cap is per client instance/connection (per-process under concurrency).
MAX_UPLOAD_ROWS = 0


class Altibase(VectorDB):
    # Each worker thread uses its own pyodbc connection (threading.local), so the
    # client is safe to share across threads and the insert runners parallelize.
    thread_safe: bool = True
    supported_filter_types: list[FilterOp] = [
        FilterOp.NonFilter,
        FilterOp.NumGE,
    ]

    def __init__(
        self,
        dim: int,
        db_config: AltibaseConfigDict,
        db_case_config: AltibaseHNSWConfig,
        collection_name: str = "vdbbench",
        drop_old: bool = False,
        **kwargs,
    ):
        self.name = "Altibase"
        self.connection_string = db_config["connection_string"]
        self.table_name = db_config.get("table_name") or collection_name
        self.case_config = db_case_config
        self.dim = dim

        self._index_name = f"{self.table_name}_hnsw"
        self._primary_field = "id"
        self._vector_field = "v"
        self.where_clause = ""
        self._inserted_total = 0

        # Per-thread connection state lives here; _thread_conn() populates it.
        # A worker thread's connection is released when the thread itself ends
        # (thread-local storage is dropped -> pyodbc closes it), so no cross-
        # thread bookkeeping is needed -- which also keeps cleanup safe under
        # both runner shapes (the load pool joins inside init(); the streaming
        # pool joins outside it).
        self._local = threading.local()

        # Bootstrap DDL on a throwaway connection (not stored on the instance).
        conn, cursor = self._create_connection(self.connection_string)
        try:
            if drop_old:
                self._drop_table(conn, cursor)
                self._create_table(conn, cursor, dim)
        finally:
            cursor.close()
            conn.close()

    def __getstate__(self):
        # VectorDBBench pickles the client to hand it to subprocesses (the load
        # task and each search process). threading.local() and live pyodbc
        # connections are not picklable, so drop the per-thread state; the
        # unpickled copy re-opens its own connections lazily via _thread_conn().
        state = self.__dict__.copy()
        state.pop("_local", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._local = threading.local()

    @staticmethod
    def _create_connection(connection_string: str):
        conn = pyodbc.connect(connection_string, autocommit=False, timeout=30)
        cursor = conn.cursor()
        assert conn is not None, "Connection is not initialized"
        assert cursor is not None, "Cursor is not initialized"
        return conn, cursor

    def _thread_conn(self):
        """Return this thread's (conn, cursor), opening it lazily.

        Worker threads in the concurrent-load and streaming-write pools never call
        init() themselves, so the connection must be created on first use here.
        """
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn, cursor = self._create_connection(self.connection_string)
            self._local.conn = conn
            self._local.cursor = cursor
        return self._local.conn, self._local.cursor

    def _close_thread(self):
        """Close the calling thread's own connection and reset its state."""
        for closer in (getattr(self._local, "cursor", None), getattr(self._local, "conn", None)):
            if closer is not None:
                try:
                    closer.close()
                except Exception:
                    log.debug("error closing connection/cursor", exc_info=True)
        self._local.conn = None
        self._local.cursor = None

    @staticmethod
    def _vec_bytes(vector) -> bytes:
        """Raw little-endian float32 bytes; server converts BINARY -> VECTOR."""
        return np.asarray(vector, dtype="<f4").tobytes()

    def _drop_table(self, conn, cursor):
        # No "IF EXISTS" in Altibase; ignore "table not found".
        try:
            cursor.execute(f"DROP TABLE {self.table_name}")
            conn.commit()
        except pyodbc.Error:
            conn.rollback()

    def _create_table(self, conn, cursor, dim: int):
        cursor.execute(
            f"CREATE TABLE {self.table_name} "
            f"({self._primary_field} BIGINT PRIMARY KEY, {self._vector_field} VECTOR({dim}))"
        )
        conn.commit()

    @contextmanager
    def init(self):
        # Connection-independent query parameters, shared by all threads.
        search_param = self.case_config.search_param()
        ef = search_param["ef_search"]
        self._hint = f"/*+ HNSW_EF_SEARCH({ef}) */ " if ef else ""
        self._distance_fn = search_param["distance_fn"]
        self._insert_sql = (
            f"INSERT INTO {self.table_name} "
            f"({self._primary_field}, {self._vector_field}) VALUES (?, ?)"
        )
        log.info(f"{self.name}: thread_safe (per-thread connections)")

        # Open the calling thread's connection eagerly so single-threaded callers
        # (search processes, DDL) have one immediately.
        self._thread_conn()
        try:
            yield
        finally:
            self._close_thread()

    def insert_embeddings(
        self,
        embeddings: list[list[float]],
        metadata: list[int],
        **kwargs,
    ) -> tuple[int, Exception | None]:
        conn, cursor = self._thread_conn()

        # TEMPORARY row cap: persist at most MAX_UPLOAD_ROWS, but still report the
        # whole batch as inserted so the runner's per-batch accounting
        # (already_insert_count == len(metadata)) stays consistent.
        take = len(metadata)
        if MAX_UPLOAD_ROWS > 0:
            remaining = MAX_UPLOAD_ROWS - self._inserted_total
            take = max(0, min(remaining, len(metadata)))

        try:
            if take > 0:
                # Bind raw float32 bytes (SQL_C_BINARY); the server's
                # BINARY -> VECTOR conversion builds the vector. Pure pyodbc.
                rows = [
                    (int(metadata[i]), self._vec_bytes(embeddings[i]))
                    for i in range(take)
                ]
                cursor.executemany(self._insert_sql, rows)
                conn.commit()
            self._inserted_total += take
            if take < len(metadata):
                log.info(
                    f"{self.name}: upload cap {MAX_UPLOAD_ROWS} reached "
                    f"(persisted {self._inserted_total}); skipped {len(metadata) - take} "
                    f"row(s) of this batch"
                )
            return len(metadata), None
        except Exception as e:
            log.warning(f"Failed to insert into Altibase table ({self.table_name}): {e}")
            try:
                conn.rollback()
            except pyodbc.Error:
                pass
            return 0, e

    def optimize(self, data_size: int | None = None):
        """Build the HNSW index after load (blocks until built)."""
        conn, cursor = self._thread_conn()

        index_param = self.case_config.index_param()
        options = [f"DISTANCE='{index_param['distance']}'"]
        if index_param["M"] is not None:
            options.append(f"M={index_param['M']}")
        if index_param["ef_construction"] is not None:
            options.append(f"EFCONSTRUCTION={index_param['ef_construction']}")

        create_sql = (
            f"CREATE INDEX {self._index_name} ON {self.table_name} ({self._vector_field}) "
            f"INDEXTYPE IS HNSW WITH ({', '.join(options)})"
        )
        if index_param["parallel"] is not None:
            create_sql += f" PARALLEL {index_param['parallel']}"

        try:
            cursor.execute(f"DROP INDEX {self._index_name}")
            conn.commit()
        except pyodbc.Error:
            conn.rollback()

        log.info(f"{self.name} creating HNSW index: {create_sql}")
        cursor.execute(create_sql)
        conn.commit()

    def prepare_filter(self, filters: Filter):
        if filters.type == FilterOp.NonFilter:
            self.where_clause = ""
        elif filters.type == FilterOp.NumGE:
            self.where_clause = f"WHERE {self._primary_field} >= {filters.int_value}"
        else:
            msg = f"Filter not supported for Altibase: {filters}"
            raise ValueError(msg)

    def search_embedding(
        self,
        query: list[float],
        k: int = 100,
        **kwargs,
    ) -> list[int]:
        _, cursor = self._thread_conn()

        # Bind the query vector as a parameter: raw little-endian float32 bytes
        # (SQL_VARBINARY), which the server converts BINARY -> VECTOR in the
        # distance-function argument. The SQL text is constant across queries, so
        # pyodbc reuses the prepared statement and the server can cache the plan.
        # A host-variable query vector in the KNN ORDER BY is accepted by the
        # server (qmvOrderBy exemption for vector distance functions) and still
        # runs the HNSW K-NN index scan.
        sql = (
            f"SELECT {self._hint}{self._primary_field} FROM {self.table_name} "
            f"{self.where_clause} "
            f"ORDER BY {self._distance_fn}({self._vector_field}, ?) LIMIT {int(k)}"
        )
        cursor.execute(sql, self._vec_bytes(query))
        return [row[0] for row in cursor.fetchall()]

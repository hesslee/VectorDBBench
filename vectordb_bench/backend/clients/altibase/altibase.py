"""Wrapper around Altibase (VECTOR + HNSW) over the VectorDB interface.

Connectivity is pyodbc -> unixODBC -> Altibase ODBC driver. Vectors are bound
as text literals '[v, v, ...]' (the path proven to work through pyodbc); the
binary SQL_C_VECTOR fast path is not reachable from pyodbc and is left as a
follow-on. KNN uses Altibase's ORDER BY <distance-fn> ... LIMIT k pattern with
the /*+ HNSW_EF_SEARCH(n) */ hint.
"""

import logging
from contextlib import contextmanager

import pyodbc

from vectordb_bench.backend.filter import Filter, FilterOp

from ..api import VectorDB
from .config import AltibaseConfigDict, AltibaseHNSWConfig

log = logging.getLogger(__name__)


class Altibase(VectorDB):
    # pyodbc connections are not shareable across threads; the runner will
    # deep-copy this instance and call init() once per process instead.
    thread_safe: bool = False
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

        self.conn, self.cursor = self._create_connection(self.connection_string)
        if drop_old:
            self._drop_table()
            self._create_table(dim)
        self.cursor.close()
        self.conn.close()
        self.cursor = None
        self.conn = None

    @staticmethod
    def _create_connection(connection_string: str):
        conn = pyodbc.connect(connection_string, autocommit=False, timeout=30)
        cursor = conn.cursor()
        assert conn is not None, "Connection is not initialized"
        assert cursor is not None, "Cursor is not initialized"
        return conn, cursor

    @staticmethod
    def _vec_literal(vector: list[float]) -> str:
        """Render a vector as an Altibase text literal '[v, v, ...]'."""
        return "[" + ",".join(map(str, vector)) + "]"

    def _drop_table(self):
        # No "IF EXISTS" in Altibase; ignore "table not found".
        try:
            self.cursor.execute(f"DROP TABLE {self.table_name}")
            self.conn.commit()
        except pyodbc.Error:
            self.conn.rollback()

    def _create_table(self, dim: int):
        self.cursor.execute(
            f"CREATE TABLE {self.table_name} "
            f"({self._primary_field} BIGINT PRIMARY KEY, {self._vector_field} VECTOR({dim}))"
        )
        self.conn.commit()

    @contextmanager
    def init(self):
        self.conn, self.cursor = self._create_connection(self.connection_string)

        search_param = self.case_config.search_param()
        ef = search_param["ef_search"]
        self._hint = f"/*+ HNSW_EF_SEARCH({ef}) */ " if ef else ""
        self._distance_fn = search_param["distance_fn"]
        self._insert_sql = (
            f"INSERT INTO {self.table_name} "
            f"({self._primary_field}, {self._vector_field}) VALUES (?, ?)"
        )
        try:
            yield
        finally:
            self.cursor.close()
            self.conn.close()
            self.cursor = None
            self.conn = None

    def insert_embeddings(
        self,
        embeddings: list[list[float]],
        metadata: list[int],
        **kwargs,
    ) -> tuple[int, Exception | None]:
        assert self.conn is not None, "Connection is not initialized"
        assert self.cursor is not None, "Cursor is not initialized"

        try:
            rows = [
                (int(metadata[i]), self._vec_literal(embeddings[i]))
                for i in range(len(metadata))
            ]
            self.cursor.executemany(self._insert_sql, rows)
            self.conn.commit()
            return len(metadata), None
        except Exception as e:
            log.warning(f"Failed to insert into Altibase table ({self.table_name}): {e}")
            try:
                self.conn.rollback()
            except pyodbc.Error:
                pass
            return 0, e

    def optimize(self, data_size: int | None = None):
        """Build the HNSW index after load (blocks until built)."""
        assert self.conn is not None, "Connection is not initialized"
        assert self.cursor is not None, "Cursor is not initialized"

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
            self.cursor.execute(f"DROP INDEX {self._index_name}")
            self.conn.commit()
        except pyodbc.Error:
            self.conn.rollback()

        log.info(f"{self.name} creating HNSW index: {create_sql}")
        self.cursor.execute(create_sql)
        self.conn.commit()

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
        assert self.conn is not None, "Connection is not initialized"
        assert self.cursor is not None, "Cursor is not initialized"

        q = self._vec_literal(query)
        sql = (
            f"SELECT {self._hint}{self._primary_field} FROM {self.table_name} "
            f"{self.where_clause} "
            f"ORDER BY {self._distance_fn}({self._vector_field}, '{q}') LIMIT {int(k)}"
        )
        self.cursor.execute(sql)
        return [row[0] for row in self.cursor.fetchall()]

import os
from typing import ClassVar, TypedDict

from pydantic import BaseModel, SecretStr

from ..api import DBCaseConfig, DBConfig, IndexType, MetricType

# Default ODBC driver location, derived from ALTIBASE_HOME when set.
# Override with --odbc-driver (CLI) or the config field.
_ALTIBASE_HOME = os.environ.get("ALTIBASE_HOME", "/home/hess/work/altidev4/altibase_home")
DEFAULT_ALTIBASE_ODBC_DRIVER = f"{_ALTIBASE_HOME}/lib/libaltibase_odbc-64bit-ul64.so"


class AltibaseConfigDict(TypedDict):
    """Resolved values handed to the Altibase client."""

    connection_string: str
    table_name: str
    tablespace: str


class AltibaseConfig(DBConfig):
    """Connection settings for Altibase over pyodbc / unixODBC (DSN-less).

    A DSN-less connection string is built from host/port/credentials plus the
    ODBC driver .so path.
    """

    user_name: SecretStr = SecretStr("SYS")
    password: SecretStr = SecretStr("MANAGER")
    host: str = "127.0.0.1"
    port: int = 21121
    db_name: str = "mydb"
    odbc_driver: str = DEFAULT_ALTIBASE_ODBC_DRIVER
    table_name: str = "vdbbench"
    # Tablespace for the data table. Empty -> the default (memory) tablespace,
    # which routes to the in-memory HNSW (smnh). Set to a disk data tablespace
    # (e.g. SYS_TBS_DISK_DATA) to exercise the on-disk HNSW (sdnh) path.
    tablespace: str = ""

    def to_dict(self) -> AltibaseConfigDict:
        user = self.user_name.get_secret_value()
        pwd = self.password.get_secret_value()
        connection_string = (
            f"DRIVER={self.odbc_driver};"
            f"SERVER={self.host};PORT={self.port};"
            f"UID={user};PWD={pwd};"
            f"DATABASE={self.db_name};ServerType=Altibase"
        )
        return {
            "connection_string": connection_string,
            "table_name": self.table_name,
            "tablespace": self.tablespace,
        }


class AltibaseIndexConfig(BaseModel, DBCaseConfig):
    """Shared metric handling for Altibase HNSW."""

    metric_type: MetricType | None = None

    # Altibase HNSW DISTANCE option value (see qdx.cpp parseDistanceMetric).
    _DISTANCE: ClassVar[dict] = {
        MetricType.L2: "L2",
        MetricType.COSINE: "COSINE",
        MetricType.IP: "INNER",
    }
    # Matching scalar distance function used in ORDER BY.
    _DISTANCE_FN: ClassVar[dict] = {
        MetricType.L2: "L2_DISTANCE",
        MetricType.COSINE: "COSINE_DISTANCE",
        MetricType.IP: "INNER_PRODUCT",
    }

    def parse_distance(self) -> str:
        if self.metric_type not in self._DISTANCE:
            msg = f"Metric type {self.metric_type} is not supported by Altibase HNSW"
            raise ValueError(msg)
        return self._DISTANCE[self.metric_type]

    def distance_fn(self) -> str:
        return self._DISTANCE_FN.get(self.metric_type, "L2_DISTANCE")


class AltibaseHNSWConfig(AltibaseIndexConfig):
    M: int | None = None
    ef_construction: int | None = None
    ef_search: int | None = None
    parallel: int | None = None
    index: IndexType = IndexType.HNSW

    def index_param(self) -> dict:
        return {
            "index_type": self.index.value,
            "distance": self.parse_distance(),
            "M": self.M,
            "ef_construction": self.ef_construction,
            "parallel": self.parallel,
        }

    def search_param(self) -> dict:
        return {
            "distance_fn": self.distance_fn(),
            "ef_search": self.ef_search,
        }


_altibase_case_config = {
    IndexType.HNSW: AltibaseHNSWConfig,
}

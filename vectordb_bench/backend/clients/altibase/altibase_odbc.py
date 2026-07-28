"""Binary VECTOR insert path for Altibase via ctypes.

pyodbc cannot bind Altibase's custom SQL_C_VECTOR C type, so the fast ingest
path calls the Altibase CLI shared library (libodbccli_sl.so) directly:
SQLBindParameter(..., SQL_C_VECTOR, SQL_VECTOR, dim, ...) sends a raw float[dim]
array and the server builds the vector without any text parsing. This mirrors
ut/sample/SQLCLI/demo_vector.cpp.
"""

import ctypes as C

import numpy as np

# --- ODBC constants (Altibase sqlcli.h) ---
SQL_HANDLE_ENV, SQL_HANDLE_DBC, SQL_HANDLE_STMT = 1, 2, 3
SQL_ATTR_ODBC_VERSION, SQL_OV_ODBC3 = 200, 3
SQL_ATTR_AUTOCOMMIT, SQL_AUTOCOMMIT_OFF = 102, 0
SQL_PARAM_INPUT = 1
SQL_C_SBIGINT, SQL_BIGINT = -25, -5
SQL_C_VECTOR, SQL_VECTOR = 20004, 20004
SQL_NTS = -3
SQL_SUCCESS, SQL_SUCCESS_WITH_INFO = 0, 1
SQL_COMMIT = 0
SQL_DRIVER_NOPROMPT = 0

SQLLEN = C.c_ssize_t
SQLULEN = C.c_size_t

_LIB_CACHE: dict = {}


def _load(lib_path: str):
    """Load the CLI lib once per path and configure ctypes signatures."""
    if lib_path in _LIB_CACHE:
        return _LIB_CACHE[lib_path]
    lib = C.CDLL(lib_path)

    def sig(name, restype, argtypes):
        fn = getattr(lib, name)
        fn.restype = restype
        fn.argtypes = argtypes
        return fn

    ns = {
        "AllocHandle": sig("SQLAllocHandle", C.c_short, [C.c_short, C.c_void_p, C.POINTER(C.c_void_p)]),
        "SetEnvAttr": sig("SQLSetEnvAttr", C.c_short, [C.c_void_p, C.c_int, C.c_void_p, C.c_int]),
        "SetConnectAttr": sig("SQLSetConnectAttr", C.c_short, [C.c_void_p, C.c_int, C.c_void_p, C.c_int]),
        "DriverConnect": sig("SQLDriverConnect", C.c_short,
                             [C.c_void_p, C.c_void_p, C.c_char_p, C.c_short, C.c_char_p, C.c_short,
                              C.POINTER(C.c_short), C.c_ushort]),
        "ExecDirect": sig("SQLExecDirect", C.c_short, [C.c_void_p, C.c_char_p, C.c_int]),
        "Prepare": sig("SQLPrepare", C.c_short, [C.c_void_p, C.c_char_p, C.c_int]),
        "BindParameter": sig("SQLBindParameter", C.c_short,
                             [C.c_void_p, C.c_ushort, C.c_short, C.c_short, C.c_short,
                              SQLULEN, C.c_short, C.c_void_p, SQLLEN, C.POINTER(SQLLEN)]),
        "Execute": sig("SQLExecute", C.c_short, [C.c_void_p]),
        "EndTran": sig("SQLEndTran", C.c_short, [C.c_short, C.c_void_p, C.c_short]),
        "FreeHandle": sig("SQLFreeHandle", C.c_short, [C.c_short, C.c_void_p]),
        "Disconnect": sig("SQLDisconnect", C.c_short, [C.c_void_p]),
        "GetDiagRec": sig("SQLGetDiagRec", C.c_short,
                          [C.c_short, C.c_void_p, C.c_short, C.c_char_p, C.POINTER(C.c_int),
                           C.c_char_p, C.c_short, C.POINTER(C.c_short)]),
    }
    _LIB_CACHE[lib_path] = ns
    return ns


def _ok(rc: int) -> bool:
    return rc in (SQL_SUCCESS, SQL_SUCCESS_WITH_INFO)


class BinaryVectorInserter:
    """Owns a dedicated CLI connection that inserts vectors via SQL_C_VECTOR."""

    def __init__(self, lib_path: str, conn_str: str, table: str, dim: int,
                 id_field: str = "id", vec_field: str = "v"):
        self._api = _load(lib_path)
        self.dim = dim
        self.henv = C.c_void_p()
        self.hdbc = C.c_void_p()
        self.hstmt = C.c_void_p()

        self._check(self._api["AllocHandle"](SQL_HANDLE_ENV, None, C.byref(self.henv)), None, "alloc env")
        self._api["SetEnvAttr"](self.henv, SQL_ATTR_ODBC_VERSION, C.c_void_p(SQL_OV_ODBC3), 0)
        self._check(self._api["AllocHandle"](SQL_HANDLE_DBC, self.henv, C.byref(self.hdbc)),
                    self.henv, "alloc dbc", SQL_HANDLE_ENV)
        rc = self._api["DriverConnect"](self.hdbc, None, conn_str.encode(), SQL_NTS,
                                        None, 0, None, SQL_DRIVER_NOPROMPT)
        self._check(rc, self.hdbc, "connect", SQL_HANDLE_DBC)
        self._api["SetConnectAttr"](self.hdbc, SQL_ATTR_AUTOCOMMIT, C.c_void_p(SQL_AUTOCOMMIT_OFF), 0)

        self._check(self._api["AllocHandle"](SQL_HANDLE_STMT, self.hdbc, C.byref(self.hstmt)),
                    self.hdbc, "alloc stmt", SQL_HANDLE_DBC)
        insert_sql = f"INSERT INTO {table} ({id_field}, {vec_field}) VALUES (?, ?)".encode()
        self._check(self._api["Prepare"](self.hstmt, insert_sql, SQL_NTS), self.hstmt, "prepare")

        # bound buffers reused across every SQLExecute
        self._id = C.c_int64(0)
        self._id_ind = SQLLEN(0)
        self._vec = (C.c_float * dim)()
        self._vec_ind = SQLLEN(dim * 4)
        self._check(self._api["BindParameter"](self.hstmt, 1, SQL_PARAM_INPUT, SQL_C_SBIGINT, SQL_BIGINT,
                                               0, 0, C.byref(self._id), 0, C.byref(self._id_ind)),
                    self.hstmt, "bind id")
        self._check(self._api["BindParameter"](self.hstmt, 2, SQL_PARAM_INPUT, SQL_C_VECTOR, SQL_VECTOR,
                                               dim, 0, self._vec, dim * 4, C.byref(self._vec_ind)),
                    self.hstmt, "bind vec")

    def _diag(self, h, htype: int) -> str:
        state = C.create_string_buffer(6)
        native = C.c_int()
        msg = C.create_string_buffer(1024)
        txtlen = C.c_short()
        self._api["GetDiagRec"](htype, h, 1, state, C.byref(native), msg, 1024, C.byref(txtlen))
        return f"[{state.value.decode(errors='replace')}] {msg.value.decode(errors='replace')}"

    def _check(self, rc: int, h, ctx: str, htype: int = SQL_HANDLE_STMT):
        if not _ok(rc):
            detail = self._diag(h, htype) if h is not None else ""
            raise RuntimeError(f"Altibase binary insert {ctx} failed (rc={rc}) {detail}")

    def insert(self, ids: list[int], embeddings: list[list[float]]) -> int:
        """Insert one batch; commit; return row count."""
        mat = np.ascontiguousarray(embeddings, dtype=np.float32)
        vec_addr = C.addressof(self._vec)
        row_bytes = self.dim * 4
        exec_fn = self._api["Execute"]
        for i in range(len(ids)):
            self._id.value = int(ids[i])
            C.memmove(vec_addr, mat[i].ctypes.data, row_bytes)
            self._check(exec_fn(self.hstmt), self.hstmt, f"execute row {i}")
        self._check(self._api["EndTran"](SQL_HANDLE_DBC, self.hdbc, SQL_COMMIT), self.hdbc, "commit", SQL_HANDLE_DBC)
        return len(ids)

    def close(self):
        try:
            if self.hstmt:
                self._api["FreeHandle"](SQL_HANDLE_STMT, self.hstmt)
            if self.hdbc:
                self._api["Disconnect"](self.hdbc)
                self._api["FreeHandle"](SQL_HANDLE_DBC, self.hdbc)
            if self.henv:
                self._api["FreeHandle"](SQL_HANDLE_ENV, self.henv)
        except Exception:  # noqa: BLE001 - close must never raise
            pass
        self.hstmt = self.hdbc = self.henv = C.c_void_p()

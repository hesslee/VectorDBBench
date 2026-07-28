from typing import Annotated, Unpack

import click
from pydantic import SecretStr

from vectordb_bench.backend.clients import DB

from ....cli.cli import (
    CommonTypedDict,
    cli,
    click_parameter_decorators_from_typed_dict,
    run,
)
from .config import DEFAULT_ALTIBASE_ODBC_DRIVER


class AltibaseTypedDict(CommonTypedDict):
    user_name: Annotated[
        str,
        click.option("--username", type=str, help="Db username", default="SYS", show_default=True),
    ]
    password: Annotated[
        str,
        click.option("--password", type=str, help="Db password", default="MANAGER", show_default=True),
    ]
    host: Annotated[
        str,
        click.option("--host", type=str, help="Db host", default="127.0.0.1", show_default=True),
    ]
    port: Annotated[
        int,
        click.option("--port", type=int, help="Db port", default=21121, show_default=True),
    ]
    db_name: Annotated[
        str,
        click.option("--db-name", type=str, help="Database name", default="mydb", show_default=True),
    ]
    dsn: Annotated[
        str,
        click.option(
            "--dsn",
            type=str,
            help="unixODBC DSN name (from /etc/odbc.ini); if set, overrides --odbc-driver/host/port",
            default="",
        ),
    ]
    odbc_driver: Annotated[
        str,
        click.option(
            "--odbc-driver",
            type=str,
            help="Path to the Altibase ODBC driver .so (used when --dsn is not given)",
            default=DEFAULT_ALTIBASE_ODBC_DRIVER,
            show_default=True,
        ),
    ]


class AltibaseHNSWTypedDict(AltibaseTypedDict):
    m: Annotated[
        int | None,
        click.option("--m", type=int, help="HNSW M (max connections per layer)", required=False),
    ]
    ef_construction: Annotated[
        int | None,
        click.option("--ef-construction", type=int, help="HNSW EFCONSTRUCTION", required=False),
    ]
    ef_search: Annotated[
        int | None,
        click.option("--ef-search", type=int, help="HNSW ef search (HNSW_EF_SEARCH hint)", required=False),
    ]
    parallel: Annotated[
        int | None,
        click.option("--parallel", type=int, help="Parallel degree for index build", required=False),
    ]


@cli.command()
@click_parameter_decorators_from_typed_dict(AltibaseHNSWTypedDict)
def AltibaseHNSW(**parameters: Unpack[AltibaseHNSWTypedDict]):
    from .config import AltibaseConfig, AltibaseHNSWConfig

    run(
        db=DB.Altibase,
        db_config=AltibaseConfig(
            db_label=parameters["db_label"],
            user_name=SecretStr(parameters["user_name"]),
            password=SecretStr(parameters["password"]),
            host=parameters["host"],
            port=parameters["port"],
            db_name=parameters["db_name"],
            dsn=parameters["dsn"],
            odbc_driver=parameters["odbc_driver"],
        ),
        db_case_config=AltibaseHNSWConfig(
            M=parameters["m"],
            ef_construction=parameters["ef_construction"],
            ef_search=parameters["ef_search"],
            parallel=parameters["parallel"],
        ),
        **parameters,
    )

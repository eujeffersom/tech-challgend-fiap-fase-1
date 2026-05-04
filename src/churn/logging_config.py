"""Configuracao unica de logging estruturado para todo o pacote."""

import logging
import sys

import structlog


def configure_logging() -> None:
    # Logs em JSON facilitam leitura local e coleta em producao.
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=logging.INFO,
    )
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str):
    """Retorna logger JSON ja configurado."""

    configure_logging()
    return structlog.get_logger(name)

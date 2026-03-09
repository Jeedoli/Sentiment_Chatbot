"""
core/logging.py
────────────────
loguru 기반 로거 설정.
- INFO 이상은 콘솔 출력
- WARNING 이상은 logs/app.log 파일에 누적 (30일 rotation)
"""

import sys
from loguru import logger


def setup_logger() -> None:
    logger.remove()   # 기본 핸들러 제거

    # ── 콘솔 ────────────────────────────────────────────────────────────
    logger.add(
        sys.stdout,
        level="INFO",
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
               "<level>{level: <8}</level> | "
               "<cyan>{name}</cyan>:<cyan>{line}</cyan> — {message}",
        colorize=True,
    )

    # ── 파일 ────────────────────────────────────────────────────────────
    logger.add(
        "logs/app.log",
        level="WARNING",
        rotation="1 day",
        retention="30 days",
        compression="zip",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{line} — {message}",
        encoding="utf-8",
    )


# 모듈 임포트 시 자동 적용
setup_logger()

__all__ = ["logger"]

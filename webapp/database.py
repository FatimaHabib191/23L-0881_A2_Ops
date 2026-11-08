"""The permanent request trail. Works with PostgreSQL in Compose and SQLite in tests."""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, Integer, String, create_engine, func, select, text
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker


class Base(DeclarativeBase):
    pass


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class RequestLog(Base):
    __tablename__ = "request_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, index=True)
    status: Mapped[str] = mapped_column(String(16), index=True)  # ok | rejected | error
    filename: Mapped[str | None] = mapped_column(String(128))
    sha256: Mapped[str | None] = mapped_column(String(64))
    size_bytes: Mapped[int | None] = mapped_column(Integer)
    width: Mapped[int | None] = mapped_column(Integer)
    height: Mapped[int | None] = mapped_column(Integer)
    label: Mapped[str | None] = mapped_column(String(32))
    confidence: Mapped[float | None] = mapped_column(Float)
    probability_ai: Mapped[float | None] = mapped_column(Float)
    threshold: Mapped[float | None] = mapped_column(Float)
    model_version: Mapped[str | None] = mapped_column(String(128))
    backend: Mapped[str | None] = mapped_column(String(16))
    latency_ms: Mapped[int | None] = mapped_column(Integer)
    error_code: Mapped[str | None] = mapped_column(String(32))


class Database:
    def __init__(self, url: str):
        self.engine = create_engine(url, pool_pre_ping=True, future=True)
        self.sessions = sessionmaker(self.engine, expire_on_commit=False, class_=Session)

    def create_schema(self) -> None:
        Base.metadata.create_all(self.engine)

    def ping(self) -> bool:
        try:
            with self.engine.connect() as connection:
                connection.execute(text("SELECT 1"))
            return True
        except Exception:  # Health checks report failure instead of raising.
            return False

    def add(self, **fields) -> RequestLog:
        record = RequestLog(**fields)
        with self.sessions.begin() as session:
            session.add(record)
        return record

    def get(self, record_id: int) -> RequestLog | None:
        with self.sessions() as session:
            return session.get(RequestLog, record_id)

    def history(self, page: int, page_size: int, status: str | None) -> tuple[list[RequestLog], int]:
        query = select(RequestLog)
        count = select(func.count()).select_from(RequestLog)
        if status:
            query, count = query.where(RequestLog.status == status), count.where(RequestLog.status == status)
        query = query.order_by(RequestLog.id.desc()).offset((page - 1) * page_size).limit(page_size)
        with self.sessions() as session:
            return list(session.scalars(query)), int(session.scalar(count) or 0)

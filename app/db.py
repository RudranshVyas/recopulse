import sqlite3
from pathlib import Path

DB = Path(__file__).resolve().parent.parent / "data" / "recopulse.db"


def connect():
    con = sqlite3.connect(DB, check_same_thread=False)
    con.row_factory = sqlite3.Row
    return con


_con = connect()


def query(sql, params=()):
    return [dict(r) for r in _con.execute(sql, params).fetchall()]


def one(sql, params=()):
    r = _con.execute(sql, params).fetchone()
    return dict(r) if r else None

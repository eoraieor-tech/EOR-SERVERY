# ============================================
# EOR Screening - Merkezi Server (v2)
# ============================================
#
# Bu server 3 shey edir:
#   1) Kriteriyalari (EOR_CRITERIA) Postgres-de saxlayir - "canli menbe".
#      import_excel_data.py --push ile yeni Excel data-si buraya
#      gonderilir, server ANINDA yenilenir - hech bir client-i
#      yeniden paylashmaq lazim deyil.
#   2) XGBoost modelini kriteriyalar her deyishende OZU yeniden qurur
#      (fayl saxlamir - hemishe cari kriteriyalara uygun tezedir).
#   3) Quyu parametrlerini + neticelerini daimi saxlayir (wells cedveli).
#
# MUHIT DEYISHENLERI (Render-de "Environment" bolmesinde teyin et):
#   DATABASE_URL      - Supabase/Neon Postgres connection string
#   EOR_ADMIN_TOKEN    - Excel-den kriteriya "push" etmek ucun sirr acar
#                        (ozun secdiyin istenilen uzun, tesadufi setir)

import os
import json
from typing import Optional, Dict, List

import numpy as np
import psycopg2
import psycopg2.extras
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

DATABASE_URL = os.environ.get("DATABASE_URL")
ADMIN_TOKEN = os.environ.get("EOR_ADMIN_TOKEN", "")

FEATURES = ["permeability", "porosity", "oil_viscosity", "temperature",
            "depth", "api_gravity", "oil_saturation"]

app = FastAPI(title="EOR Screening Server", version="2.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_criteria_cache: Dict = {}
_xgb_model = None
_xgb_classes: List[str] = []


def get_conn():
    if not DATABASE_URL:
        raise HTTPException(500, "DATABASE_URL mühit dəyişəni təyin olunmayıb")
    return psycopg2.connect(DATABASE_URL, sslmode="require")


def init_db():
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS wells (
                    id SERIAL PRIMARY KEY,
                    well_name TEXT UNIQUE NOT NULL,
                    parameters JSONB NOT NULL,
                    best_method TEXT,
                    best_percentage REAL,
                    created_at TIMESTAMPTZ DEFAULT now(),
                    updated_at TIMESTAMPTZ DEFAULT now()
                );
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS criteria_store (
                    id INT PRIMARY KEY DEFAULT 1,
                    data JSONB NOT NULL,
                    updated_at TIMESTAMPTZ DEFAULT now(),
                    CHECK (id = 1)
                );
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS app_version_store (
                    id INT PRIMARY KEY DEFAULT 1,
                    latest_version TEXT NOT NULL DEFAULT '1.0.0',
                    download_url TEXT,
                    notes TEXT,
                    updated_at TIMESTAMPTZ DEFAULT now(),
                    CHECK (id = 1)
                );
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS audit_log (
                    id SERIAL PRIMARY KEY,
                    user_name TEXT,
                    user_email TEXT,
                    action TEXT NOT NULL,
                    details TEXT,
                    created_at TIMESTAMPTZ DEFAULT now()
                );
            """)
        conn.commit()
    finally:
        conn.close()


def load_criteria_from_db():
    global _criteria_cache
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT data FROM criteria_store WHERE id = 1")
            row = cur.fetchone()
        _criteria_cache = row[0] if row else {}
    finally:
        conn.close()


def save_criteria_to_db(criteria: dict):
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO criteria_store (id, data, updated_at) VALUES (1, %s, now())
                ON CONFLICT (id) DO UPDATE SET data = EXCLUDED.data, updated_at = now()
            """, (json.dumps(criteria),))
        conn.commit()
    finally:
        conn.close()


def _global_range(feat):
    los, his = [], []
    for info in _criteria_cache.values():
        if feat in info.get("criteria", {}):
            lo, hi, _ = info["criteria"][feat]
            los.append(lo)
            his.append(hi)
    return (min(los), max(his)) if los else (0.0, 1.0)


def _sample_param(rng, lo, hi, avg, n):
    if lo < avg < hi:
        return rng.triangular(lo, avg, hi, n)
    if lo == hi:
        return np.full(n, lo)
    return rng.uniform(lo, hi, n)


def train_xgb_from_criteria():
    """Kriteriyalar her deyishende suni data ile XGBoost-u yeniden qurur.
    Fayl saxlanmir - server yeniden basladiqda da bu funksiya cagirilir,
    hemishe cari kriteriyalara uygun model olur."""
    global _xgb_model, _xgb_classes
    try:
        from xgboost import XGBClassifier
        from sklearn.preprocessing import LabelEncoder
    except ImportError:
        _xgb_model = None
        return

    if not _criteria_cache:
        _xgb_model = None
        return

    rng = np.random.default_rng(42)
    SAMPLES = 300

    X, y = [], []
    for method, info in _criteria_cache.items():
        crit = info.get("criteria", {})
        cols = {}
        for feat in FEATURES:
            if feat in crit:
                lo, hi, avg = crit[feat]
                cols[feat] = _sample_param(rng, lo, hi, avg, SAMPLES)
            else:
                lo, hi = _global_range(feat)
                cols[feat] = rng.uniform(lo, hi, SAMPLES)
        for i in range(SAMPLES):
            X.append([cols[f][i] for f in FEATURES])
            y.append(method)

    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    model = XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.1,
        objective="multi:softprob", num_class=len(le.classes_),
        eval_metric="mlogloss", random_state=42,
    )
    model.fit(np.array(X), y_enc)
    _xgb_model = model
    _xgb_classes = le.classes_.tolist()


@app.on_event("startup")
def on_startup():
    init_db()
    load_criteria_from_db()
    train_xgb_from_criteria()


def score_parameter(value, lo, hi, avg):
    if value < lo or value > hi:
        return 0.0
    if lo == hi:
        return 100.0
    if value <= avg:
        return 100.0 if avg == lo else 100.0 * (value - lo) / (avg - lo)
    return 100.0 if hi == avg else 100.0 * (hi - value) / (hi - avg)


def rule_based_recommend(user_params: dict):
    results = []
    for method, info in _criteria_cache.items():
        crit = info.get("criteria", {})
        total, used = 0.0, 0
        for feat in FEATURES:
            if feat in user_params and feat in crit:
                lo, hi, avg = crit[feat]
                total += score_parameter(user_params[feat], lo, hi, avg)
                used += 1
        pct = (total / used) if used else 0.0
        results.append({"method": method, "rule_pct": pct, "used": used})
    results.sort(key=lambda r: r["rule_pct"], reverse=True)
    return results


def xgb_predict_probs(user_params: dict):
    if _xgb_model is None:
        return None
    defaults = {}
    for feat in FEATURES:
        vals = [info["criteria"][feat][2] for info in _criteria_cache.values() if feat in info.get("criteria", {})]
        defaults[feat] = sum(vals) / len(vals) if vals else 0.0
    row = [user_params.get(f, defaults[f]) for f in FEATURES]
    proba = _xgb_model.predict_proba(np.array([row]))[0]
    return {cls: float(p) * 100 for cls, p in zip(_xgb_classes, proba)}


def log_action(user_name, user_email, action, details=None):
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO audit_log (user_name, user_email, action, details) VALUES (%s, %s, %s, %s)",
                (user_name, user_email, action, details),
            )
        conn.commit()
    finally:
        conn.close()


class RecommendRequest(BaseModel):
    well_name: Optional[str] = None
    parameters: Dict[str, float]
    save: bool = False
    user_name: Optional[str] = None
    user_email: Optional[str] = None


class CriteriaPush(BaseModel):
    criteria: Dict


class VersionPush(BaseModel):
    latest_version: str
    download_url: Optional[str] = None
    notes: Optional[str] = None


@app.get("/")
def root():
    return {"status": "ok", "methods_loaded": len(_criteria_cache), "xgb_ready": _xgb_model is not None}


@app.get("/version")
def get_version():
    conn = get_conn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT latest_version, download_url, notes FROM app_version_store WHERE id = 1")
            row = cur.fetchone()
        if not row:
            return {"latest_version": "1.0.0", "download_url": None, "notes": None}
        return row
    finally:
        conn.close()


@app.post("/admin/set_version")
def set_version(payload: VersionPush, x_admin_token: str = Header(default="")):
    if not ADMIN_TOKEN or x_admin_token != ADMIN_TOKEN:
        raise HTTPException(403, "Admin token yanlışdır")
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO app_version_store (id, latest_version, download_url, notes, updated_at)
                VALUES (1, %s, %s, %s, now())
                ON CONFLICT (id) DO UPDATE SET
                    latest_version = EXCLUDED.latest_version,
                    download_url = EXCLUDED.download_url,
                    notes = EXCLUDED.notes,
                    updated_at = now()
                """,
                (payload.latest_version, payload.download_url, payload.notes),
            )
        conn.commit()
    finally:
        conn.close()
    return {"status": "ok", "latest_version": payload.latest_version}


@app.get("/criteria")
def get_criteria():
    return {"features": FEATURES, "criteria": _criteria_cache}


@app.post("/admin/update_criteria")
def update_criteria(payload: CriteriaPush, x_admin_token: str = Header(default="")):
    if not ADMIN_TOKEN or x_admin_token != ADMIN_TOKEN:
        raise HTTPException(403, "Admin token yanlışdır")
    save_criteria_to_db(payload.criteria)
    load_criteria_from_db()
    train_xgb_from_criteria()
    log_action(None, None, "criteria_updated", f"{len(_criteria_cache)} metod yükləndi (Excel push)")
    return {"status": "ok", "methods": len(_criteria_cache), "xgb_ready": _xgb_model is not None}


@app.post("/recommend")
def recommend(req: RecommendRequest):
    if not _criteria_cache:
        raise HTTPException(503, "Server-de heç bir kriteriya yüklənməyib (əvvəlcə Excel-i push et)")

    rule_results = rule_based_recommend(req.parameters)
    xgb_scores = xgb_predict_probs(req.parameters)

    combined = []
    for r in rule_results:
        xgb_pct = xgb_scores.get(r["method"]) if xgb_scores else None
        comb = (r["rule_pct"] + xgb_pct) / 2 if xgb_pct is not None else r["rule_pct"]
        combined.append({
            "method": r["method"], "rule_pct": r["rule_pct"],
            "xgb_pct": xgb_pct, "combined_pct": comb,
        })
    combined.sort(key=lambda r: r["combined_pct"], reverse=True)

    if req.save and req.well_name and req.well_name.strip():
        best = combined[0]
        conn = get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO wells (well_name, parameters, best_method, best_percentage, updated_at)
                    VALUES (%s, %s, %s, %s, now())
                    ON CONFLICT (well_name) DO UPDATE SET
                        parameters = EXCLUDED.parameters,
                        best_method = EXCLUDED.best_method,
                        best_percentage = EXCLUDED.best_percentage,
                        updated_at = now()
                    """,
                    (req.well_name.strip(), json.dumps(req.parameters), best["method"], best["combined_pct"]),
                )
            conn.commit()
        finally:
            conn.close()
        log_action(req.user_name, req.user_email, "well_saved",
                   f"'{req.well_name.strip()}' -> {best['method']} ({best['combined_pct']:.1f}%)")

    return {"results": combined}


@app.get("/wells")
def list_wells():
    conn = get_conn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT well_name, best_method, best_percentage, updated_at "
                "FROM wells ORDER BY updated_at DESC"
            )
            return cur.fetchall()
    finally:
        conn.close()


@app.get("/wells/{well_name}")
def get_well(well_name: str):
    conn = get_conn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT well_name, parameters, best_method, best_percentage, updated_at "
                "FROM wells WHERE well_name = %s",
                (well_name,),
            )
            row = cur.fetchone()
        if not row:
            raise HTTPException(404, "Quyu tapılmadı")
        return row
    finally:
        conn.close()


@app.delete("/wells/{well_name}")
def delete_well(well_name: str, user_name: Optional[str] = None, user_email: Optional[str] = None):
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM wells WHERE well_name = %s", (well_name,))
            deleted = cur.rowcount
        conn.commit()
        if deleted == 0:
            raise HTTPException(404, "Quyu tapılmadı")
        log_action(user_name, user_email, "well_deleted", well_name)
        return {"status": "deleted", "well_name": well_name}
    finally:
        conn.close()


@app.get("/audit_log")
def get_audit_log(limit: int = 200):
    conn = get_conn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT user_name, user_email, action, details, created_at "
                "FROM audit_log ORDER BY created_at DESC LIMIT %s",
                (limit,),
            )
            return cur.fetchall()
    finally:
        conn.close()

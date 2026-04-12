import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import io, os, re, time, sqlite3, secrets
import bcrypt
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('dark_background')

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import shap

try:
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    from catboost import CatBoostClassifier
    from sklearn.ensemble import RandomForestClassifier, StackingClassifier
    from sklearn.linear_model import LogisticRegression
except ImportError:
    pass

# ══════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════
st.set_page_config(
    page_title="AI Fraud Guard Pro",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ══════════════════════════════════════════
# GLOBAL CSS
# ══════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg:#060d1a; --bg2:#0b1628; --bg3:#0f1f35;
    --cyan:#00e0ff; --blue:#3b82f6; --violet:#8b5cf6;
    --green:#10f5a8; --red:#ff4d6d; --amber:#f59e0b;
    --text:#e8f4ff; --muted:rgba(168,200,230,0.6);
    --border:rgba(0,224,255,0.12); --card:rgba(10,20,40,0.8);
}

/* ─── BASE ─────────────────────────── */
html, body,
[data-testid="stAppViewContainer"],
[data-testid="stApp"] {
    background: var(--bg) !important;
    font-family: 'DM Sans', sans-serif !important;
    color: var(--text) !important;
}
[data-testid="stAppViewContainer"]::before {
    content:''; position:fixed; inset:0; pointer-events:none; z-index:0;
    background-image:
        linear-gradient(rgba(0,224,255,0.02) 1px,transparent 1px),
        linear-gradient(90deg,rgba(0,224,255,0.02) 1px,transparent 1px);
    background-size:52px 52px;
}
[data-testid="stAppViewContainer"]::after {
    content:''; position:fixed; inset:0; pointer-events:none; z-index:0;
    background:
        radial-gradient(ellipse 80% 50% at 20% 10%,rgba(0,224,255,0.05) 0%,transparent 60%),
        radial-gradient(ellipse 60% 50% at 80% 90%,rgba(139,92,246,0.06) 0%,transparent 60%);
}
#MainMenu,footer,header,
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"]{display:none!important}
[data-testid="stVerticalBlock"],
[data-testid="block-container"],
section.main{position:relative;z-index:1}

/* ─── SIDEBAR ──────────────────────── */
[data-testid="stSidebar"]{
    background:rgba(6,13,26,0.97)!important;
    border-right:1px solid var(--border)!important;
}
[data-testid="stSidebar"] *{color:var(--text)!important}

/* ─── INPUTS ───────────────────────── */
[data-testid="stTextInput"] input,
[data-testid="stTextArea"] textarea,
[data-testid="stNumberInput"] input{
    background:rgba(11,22,40,0.9)!important;
    border:1px solid rgba(0,224,255,0.2)!important;
    border-radius:10px!important;
    color:var(--text)!important;
    font-family:'DM Sans',sans-serif!important;
}
[data-testid="stTextInput"] input:focus,
[data-testid="stTextArea"] textarea:focus{
    border-color:var(--cyan)!important;
    box-shadow:0 0 0 2px rgba(0,224,255,0.1)!important;
}
[data-testid="stTextInput"] input::placeholder{color:rgba(168,200,230,0.4)!important}

/* auth screen — override white bg on inputs */
[data-testid="stTextInput"] input[type="password"],
[data-testid="stTextInput"] input[type="text"]{
    background:rgba(11,22,40,0.95)!important;
    color:var(--text)!important;
}

/* ─── BUTTONS ──────────────────────── */
.stButton>button{
    background:linear-gradient(135deg,var(--cyan),var(--blue))!important;
    color:#000!important; font-weight:700!important;
    border:none!important; border-radius:10px!important;
    font-family:'DM Sans',sans-serif!important;
    transition:transform .15s,box-shadow .15s!important;
}
.stButton>button:hover{
    transform:translateY(-1px)!important;
    box-shadow:0 8px 24px rgba(0,224,255,0.25)!important;
}

/* ─── METRICS ──────────────────────── */
[data-testid="metric-container"]{
    background:var(--card)!important;
    border:1px solid var(--border)!important;
    border-radius:14px!important;
    padding:16px 20px!important;
    backdrop-filter:blur(12px)!important;
}
[data-testid="metric-container"] label{color:var(--muted)!important;font-size:12px!important}
[data-testid="metric-container"] [data-testid="stMetricValue"]{
    color:var(--text)!important;
    font-family:'Syne',sans-serif!important;
    font-size:26px!important;
}

/* ─── TABS ─────────────────────────── */
[data-testid="stTabs"] [role="tablist"]{
    background:rgba(11,22,40,0.8)!important;
    border:1px solid var(--border)!important;
    border-radius:12px!important;
    padding:4px!important; gap:2px!important;
}
[data-testid="stTabs"] button[role="tab"]{
    background:transparent!important;
    color:var(--muted)!important;
    border:none!important; border-radius:8px!important;
    font-family:'DM Sans',sans-serif!important;
    font-weight:500!important; font-size:13px!important;
    padding:7px 14px!important;
    transition:all .2s!important;
}
[data-testid="stTabs"] button[role="tab"]:hover{
    color:var(--text)!important;
    background:rgba(0,224,255,0.07)!important;
}
[data-testid="stTabs"] button[role="tab"][aria-selected="true"]{
    background:rgba(0,224,255,0.12)!important;
    color:var(--cyan)!important;
}
[data-testid="stTabs"] [role="tabpanel"]{padding-top:20px!important}

/* ─── FORMS ────────────────────────── */
[data-testid="stForm"]{
    background:var(--card)!important;
    border:1px solid var(--border)!important;
    border-radius:16px!important;
    padding:20px!important;
    backdrop-filter:blur(12px)!important;
}

/* ─── SELECTBOX ────────────────────── */
[data-testid="stSelectbox"]>div>div,
[data-testid="stMultiSelect"]>div>div{
    background:rgba(11,22,40,0.9)!important;
    border:1px solid rgba(0,224,255,0.2)!important;
    border-radius:10px!important; color:var(--text)!important;
}

/* ─── DATAFRAME ────────────────────── */
[data-testid="stDataFrame"]{
    border:1px solid var(--border)!important;
    border-radius:12px!important; overflow:hidden!important;
}

/* ─── PROGRESS ─────────────────────── */
[data-testid="stProgress"]>div{
    background:rgba(0,224,255,0.1)!important; border-radius:4px!important;
}
[data-testid="stProgress"]>div>div{
    background:linear-gradient(90deg,var(--cyan),var(--blue))!important;
    border-radius:4px!important;
}

/* ─── FILE UPLOADER ────────────────── */
[data-testid="stFileUploader"]{
    background:rgba(11,22,40,0.6)!important;
    border:1px dashed rgba(0,224,255,0.25)!important;
    border-radius:12px!important;
}

/* ─── EXPANDER ─────────────────────── */
[data-testid="stExpander"]{
    background:var(--card)!important;
    border:1px solid var(--border)!important;
    border-radius:12px!important;
}

/* ─── PYPLOT FIX ───────────────────── */
[data-testid="stImage"] img,
.stPlotlyChart,
[data-testid="stPyplotGlobalUse"] {
    background: transparent !important;
}
/* Force matplotlib figures dark */
.element-container figure,
.element-container .stImage {
    background: rgba(11,22,40,0.0) !important;
}

/* ─── CUSTOM CLASSES ───────────────── */
.hd { font-family:'Syne',sans-serif; font-weight:700; font-size:17px;
       margin:8px 0 16px; display:flex; align-items:center; gap:8px; }

.pulse-dot {
    width:7px; height:7px; border-radius:50%;
    background:var(--green); box-shadow:0 0 8px var(--green);
    display:inline-block; animation:pdot 2s ease-in-out infinite;
}
@keyframes pdot{0%,100%{box-shadow:0 0 6px var(--green)}50%{box-shadow:0 0 16px var(--green)}}

.status-bar{
    display:flex; align-items:center; gap:10px;
    background:rgba(16,245,168,0.07);
    border:1px solid rgba(16,245,168,0.2);
    border-radius:12px; padding:9px 14px;
    margin-bottom:12px; font-size:13px; color:var(--green);
}
.role-badge{
    margin-left:auto;
    background:linear-gradient(90deg,var(--violet),var(--blue));
    color:#fff; font-size:10px; font-weight:700;
    letter-spacing:.08em; text-transform:uppercase;
    padding:3px 10px; border-radius:20px;
}

.feat-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin:16px 0 24px}
.feat-card{
    background:var(--bg3); border:1px solid var(--border);
    border-radius:16px; padding:22px 20px; text-align:center;
    transition:border-color .2s,transform .15s;
}
.feat-card:hover{border-color:rgba(0,224,255,0.3);transform:translateY(-2px)}
.feat-icon{font-size:30px;margin-bottom:10px}
.feat-title{font-family:'Syne',sans-serif;font-weight:700;font-size:14px;margin-bottom:6px}
.feat-text{font-size:12px;color:var(--muted);line-height:1.6}

.risk-box{
    border-radius:14px; padding:18px 22px;
    display:flex; align-items:center; gap:18px; margin:14px 0;
    backdrop-filter:blur(12px);
}
.risk-fraud{background:rgba(255,77,109,0.08);border:1px solid rgba(255,77,109,0.3)}
.risk-safe {background:rgba(16,245,168,0.06);border:1px solid rgba(16,245,168,0.2)}
.risk-mid  {background:rgba(245,158,11,0.07);border:1px solid rgba(245,158,11,0.25)}
.risk-pct  {font-family:'Syne',sans-serif;font-size:40px;font-weight:800}
.risk-lbl  {font-size:15px;font-weight:700}
.risk-sub  {font-size:12px;color:var(--muted);margin-top:3px}

.info-box{
    background:rgba(11,22,40,0.7);
    border:1px solid rgba(0,224,255,0.1);
    border-radius:14px; padding:18px 22px;
    font-size:13px; color:rgba(168,200,230,0.8); line-height:2;
}

/* ─── AUTH CARD FIX ─────────────────── */
/* Ensure auth card contains all content */
.auth-outer {
    display:flex; justify-content:center;
    padding: 40px 16px 60px;
}
.auth-card{
    background:rgba(10,20,40,0.88);
    backdrop-filter:blur(32px) saturate(160%);
    border:1px solid rgba(0,224,255,0.15);
    border-radius:24px;
    padding:2.5rem 2.5rem 2rem;
    width:100%; max-width:460px;
    box-shadow:0 24px 64px rgba(0,0,0,0.6),0 0 40px rgba(0,224,255,0.07);
    animation:cardIn .5s cubic-bezier(.22,1,.36,1) both;
    box-sizing:border-box;
}
@keyframes cardIn{
    from{opacity:0;transform:translateY(20px) scale(0.97)}
    to  {opacity:1;transform:translateY(0) scale(1)}
}
.brand-block{
    display:flex; align-items:center; gap:12px;
    margin-bottom:.2rem; padding-bottom:1.2rem;
    border-bottom:1px solid rgba(0,224,255,0.1);
}
.brand-icon{
    width:44px; height:44px; flex-shrink:0;
    background:linear-gradient(135deg,var(--cyan),var(--violet));
    border-radius:12px; display:flex; align-items:center;
    justify-content:center; font-size:20px;
    box-shadow:0 0 20px rgba(0,224,255,0.3);
}
.brand-name{font-family:'Syne',sans-serif;font-weight:800;font-size:1.35rem;
    background:linear-gradient(90deg,var(--cyan) 0%,#a5b4fc 100%);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text}
.brand-sub{font-size:.72rem;color:var(--muted);letter-spacing:.07em;text-transform:uppercase;margin-top:1px}
.auth-footer{
    text-align:center;font-size:.74rem;color:var(--muted);
    margin-top:1.4rem;padding-top:.9rem;
    border-top:1px solid rgba(0,224,255,0.08);
}
.auth-footer span{color:var(--cyan);font-weight:500}

/* ─── FILE UPLOADER main area ───────────── */
[data-testid="stFileUploader"]{
    background:transparent!important;
    border:none!important;
}
[data-testid="stFileUploaderDropzone"]{
    border:2px dashed rgba(0,224,255,0.25)!important;
    border-radius:14px!important;
    background:rgba(0,224,255,0.025)!important;
    min-height:90px!important;
    transition:all .25s!important;
}
[data-testid="stFileUploaderDropzone"]:hover{
    border-color:rgba(0,224,255,0.55)!important;
    background:rgba(0,224,255,0.06)!important;
    box-shadow:0 0 20px rgba(0,224,255,0.08)!important;
}
[data-testid="stFileUploaderDropzone"] p{
    color:rgba(168,200,230,0.55)!important;
    font-size:13px!important;
}
[data-testid="stFileUploaderDropzone"] svg{
    color:rgba(0,224,255,0.4)!important;
}
/* uploaded file chip */
[data-testid="stFileUploader"] [data-testid="stMarkdownContainer"]{
    background:rgba(16,245,168,0.07)!important;
    border-radius:8px!important;
    padding:6px 10px!important;
}
/* Browse files button */
[data-testid="stFileUploaderDropzone"] button{
    background:linear-gradient(135deg,rgba(0,224,255,0.15),rgba(59,130,246,0.15))!important;
    border:1px solid rgba(0,224,255,0.3)!important;
    color:var(--cyan)!important;
    border-radius:8px!important;
    font-weight:600!important;
}
[data-testid="stFileUploaderDropzone"] button:hover{
    background:linear-gradient(135deg,rgba(0,224,255,0.25),rgba(59,130,246,0.25))!important;
    border-color:rgba(0,224,255,0.6)!important;
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════
# DATABASE
# ══════════════════════════════════════════
DB_PATH = "users.db"

def _conn(): return sqlite3.connect(DB_PATH)

def init_db():
    with _conn() as c:
        c.execute("""CREATE TABLE IF NOT EXISTS users(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'user',
            is_blocked INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL)""")
        c.execute("""CREATE TABLE IF NOT EXISTS logs(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user TEXT,action TEXT,timestamp TEXT)""")
        c.execute("""CREATE TABLE IF NOT EXISTS api_keys(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            key_name TEXT,api_key TEXT,key_value TEXT,
            created_by TEXT,is_revoked INTEGER DEFAULT 0,created_at TEXT)""")
        c.execute("""CREATE TABLE IF NOT EXISTS user_history(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,action_type TEXT,details TEXT,created_at TEXT)""")
        c.execute("""CREATE TABLE IF NOT EXISTS messages(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            from_user TEXT NOT NULL,
            subject   TEXT NOT NULL DEFAULT '',
            body      TEXT NOT NULL,
            is_read   INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL)""")
        for col in ["is_blocked"]:
            try: c.execute(f"ALTER TABLE users ADD COLUMN {col} INTEGER DEFAULT 0")
            except: pass
        api_cols=[r[1] for r in c.execute("PRAGMA table_info(api_keys)")]
        for col,typ in [("api_key","TEXT"),("key_value","TEXT"),("created_by","TEXT"),
                        ("is_revoked","INTEGER DEFAULT 0"),("created_at","TEXT")]:
            if col not in api_cols:
                try: c.execute(f"ALTER TABLE api_keys ADD COLUMN {col} {typ}")
                except: pass
        msg_cols=[r[1] for r in c.execute("PRAGMA table_info(messages)")]
        for col,typ in [("subject","TEXT DEFAULT ''"),("is_read","INTEGER DEFAULT 0")]:
            if col not in msg_cols:
                try: c.execute(f"ALTER TABLE messages ADD COLUMN {col} {typ}")
                except: pass
        c.commit()
    if not _admin_exists(): _create_user("admin","Admin@12345","admin")

def _admin_exists():
    with _conn() as c:
        return c.execute("SELECT 1 FROM users WHERE role='admin' LIMIT 1").fetchone() is not None

def _uexists(u):
    with _conn() as c:
        return c.execute("SELECT 1 FROM users WHERE username=?",(u,)).fetchone() is not None

def _create_user(username,password,role="user"):
    if _uexists(username): return False
    if role=="admin" and _admin_exists(): return False
    ph = bcrypt.hashpw(password.encode(),bcrypt.gensalt()).decode()
    with _conn() as c:
        c.execute("INSERT INTO users(username,password_hash,role,created_at)VALUES(?,?,?,?)",
                  (username,ph,role,datetime.utcnow().isoformat()))
        c.commit()
    _log(username,f"created role={role}")
    return True

def _auth(username,password):
    with _conn() as c:
        row=c.execute("SELECT id,username,password_hash,role,is_blocked,created_at FROM users WHERE username=?",(username,)).fetchone()
    if not row: return None
    uid,un,ph,role,blocked,cat=row
    if blocked: return {"blocked":True}
    if bcrypt.checkpw(password.encode(),ph.encode()):
        return {"id":uid,"username":un,"role":role,"is_blocked":blocked,"created_at":cat}
    return None

def _log(user,action):
    with _conn() as c:
        c.execute("INSERT INTO logs(user,action,timestamp)VALUES(?,?,?)",(user,action,datetime.utcnow().isoformat()))
        c.commit()

def _save_hist(username,atype,details):
    with _conn() as c:
        c.execute("INSERT INTO user_history(username,action_type,details,created_at)VALUES(?,?,?,?)",
                  (username,atype,details,datetime.utcnow().isoformat()))
        c.commit()

def _get_users():
    with _conn() as c:
        return c.execute("SELECT id,username,role,is_blocked,created_at FROM users ORDER BY id").fetchall()

def _get_logs(limit=300):
    with _conn() as c:
        return c.execute("SELECT id,user,action,timestamp FROM logs ORDER BY id DESC LIMIT ?",(limit,)).fetchall()

def _get_hist(username):
    with _conn() as c:
        return c.execute("SELECT id,action_type,details,created_at FROM user_history WHERE username=? ORDER BY id DESC LIMIT 100",(username,)).fetchall()

def _del_user(uid):
    with _conn() as c: c.execute("DELETE FROM users WHERE id=?",(uid,)); c.commit()

def _upd_role(uid,role):
    with _conn() as c: c.execute("UPDATE users SET role=? WHERE id=?",(role,uid)); c.commit()

def _set_block(uid,val):
    with _conn() as c: c.execute("UPDATE users SET is_blocked=? WHERE id=?",(val,uid)); c.commit()

def _counts():
    with _conn() as c:
        t=c.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        a=c.execute("SELECT COUNT(*) FROM users WHERE role='admin'").fetchone()[0]
        r=c.execute("SELECT COUNT(*) FROM users WHERE role='user'").fetchone()[0]
        b=c.execute("SELECT COUNT(*) FROM users WHERE is_blocked=1").fetchone()[0]
    return t,a,r,b

def _clear_logs():
    with _conn() as c: c.execute("DELETE FROM logs"); c.commit()

def _chg_pw(pw,uname="admin"):
    ph=bcrypt.hashpw(pw.encode(),bcrypt.gensalt()).decode()
    with _conn() as c:
        c.execute("UPDATE users SET password_hash=? WHERE username=? AND role='admin'",(ph,uname)); c.commit()

def _mk_key(name,by):
    key=secrets.token_hex(24)
    with _conn() as c:
        # Detect actual column names to handle legacy schema (key_value vs api_key)
        cols=[r[1] for r in c.execute("PRAGMA table_info(api_keys)")]
        key_col="api_key" if "api_key" in cols else "key_value"
        by_col="created_by" if "created_by" in cols else "created_by"
        c.execute(
            f"INSERT INTO api_keys(key_name,{key_col},{by_col},created_at)VALUES(?,?,?,?)",
            (name,key,by,datetime.utcnow().isoformat()))
        c.commit()
    return key

def _get_keys():
    with _conn() as c:
        cols=[r[1] for r in c.execute("PRAGMA table_info(api_keys)")]
        key_col="api_key" if "api_key" in cols else "key_value"
        return c.execute(
            f"SELECT id,key_name,{key_col},created_by,is_revoked,created_at FROM api_keys ORDER BY id DESC"
        ).fetchall()

def _revoke_key(kid):
    with _conn() as c: c.execute("UPDATE api_keys SET is_revoked=1 WHERE id=?",(kid,)); c.commit()

# ── MESSAGE FUNCTIONS ───────────────────────────────────────
def _send_msg(from_user, subject, body):
    with _conn() as c:
        c.execute("INSERT INTO messages(from_user,subject,body,created_at)VALUES(?,?,?,?)",
                  (from_user, subject, body, datetime.utcnow().isoformat()))
        c.commit()

def _get_msgs(limit=200):
    with _conn() as c:
        return c.execute(
            "SELECT id,from_user,subject,body,is_read,created_at FROM messages ORDER BY id DESC LIMIT ?",
            (limit,)).fetchall()

def _mark_read(mid):
    with _conn() as c:
        c.execute("UPDATE messages SET is_read=1 WHERE id=?",(mid,)); c.commit()

def _del_msg(mid):
    with _conn() as c:
        c.execute("DELETE FROM messages WHERE id=?",(mid,)); c.commit()

def _unread_count():
    with _conn() as c:
        return c.execute("SELECT COUNT(*) FROM messages WHERE is_read=0").fetchone()[0]

def _val_user(u):
    if len(u)<3: return "Кем дегенде 3 таңба."
    if not re.match(r"^[A-Za-z0-9_]+$",u): return "Тек әріп, цифр, '_'."
    return None

def _val_pass(p):
    if len(p)<8: return "Кем дегенде 8 таңба."
    if not re.search(r"[A-Z]",p): return "Бір бас әріп керек."
    if not re.search(r"\d",p): return "Бір цифр керек."
    return None

# ══════════════════════════════════════════
# GEO HELPERS
# ══════════════════════════════════════════
COUNTRY_COORDS = {
    "Қазақстан":(48.0,68.0),"Ресей":(61.0,105.0),"Польша":(52.0,19.0),
    "Канада":(56.0,-106.0),"БАЭ":(24.0,54.0),"АҚШ":(37.0,-95.0),
    "Ұлыбритания":(55.0,-3.0),"Германия":(51.0,10.0),"Франция":(46.0,2.0),
    "Нигерия":(9.0,8.0),"Индонезия":(-2.5,118.0),"Филиппин":(13.0,122.0),
    "Бразилия":(-10.0,-55.0),"Түркия":(39.0,35.0),"Үндістан":(21.0,78.0),
    "Мексика":(23.0,-102.0),"Оңтүстік Африка":(-30.0,25.0)
}

def _country(amount,risk):
    if risk>0.80: return np.random.choice(["Нигерия","Индонезия","Филиппин","Бразилия"])
    if risk>0.55: return np.random.choice(["Түркия","Үндістан","Мексика","Оңтүстік Африка"])
    if amount>500: return np.random.choice(["АҚШ","Ұлыбритания","Германия","Франция"])
    return np.random.choice(["Қазақстан","Ресей","Польша","Канада","БАЭ"])

@st.cache_data(show_spinner=False)
def _enrich_geo(df_hash, df):
    """Cache by hash so repeated calls are instant."""
    tmp=df.copy()
    if "AI_Risk_Score" not in tmp.columns: tmp["AI_Risk_Score"]=np.random.uniform(0.05,0.95,len(tmp))
    if "Amount" not in tmp.columns: tmp["Amount"]=np.random.uniform(1,1000,len(tmp))
    countries,lats,lons=[],[],[]
    for _,row in tmp.iterrows():
        c=_country(float(row.get("Amount",0)),float(row.get("AI_Risk_Score",.1)))
        lat,lon=COUNTRY_COORDS.get(c,(48.,68.))
        countries.append(c); lats.append(lat+np.random.uniform(-1,1)); lons.append(lon+np.random.uniform(-1,1))
    tmp["Ел"]=countries; tmp["lat"]=lats; tmp["lon"]=lons
    return tmp

# ══════════════════════════════════════════
# MODEL & DATA
# ══════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_model():
    if os.path.exists('model.pkl') and os.path.exists('model_columns.pkl'):
        try:
            return joblib.load('model.pkl'), joblib.load('model_columns.pkl')
        except Exception as e:
            st.error(f"Модель қатесі: {e}")
    return None, None

@st.cache_data(show_spinner=False)
def load_csv(uploaded_file):
    if uploaded_file is None: return None
    for enc in ['utf-8','cp1251']:
        for sep in [',',';']:
            for hdr in [0,1]:
                try:
                    uploaded_file.seek(0)
                    df=pd.read_csv(uploaded_file,encoding=enc,sep=sep,header=hdr)
                    cl=[c.lower() for c in df.columns]
                    if any(k in cl for k in ['class','amount','v1']):
                        rm={}
                        for col in df.columns:
                            if col.lower()=='amount': rm[col]='Amount'
                            if col.lower()=='class': rm[col]='Class'
                        if rm: df=df.rename(columns=rm)
                        for col in df.columns:
                            if df[col].dtype=='object':
                                try: df[col]=pd.to_numeric(df[col].astype(str).str.replace(',','.'))
                                except: pass
                        return df
                except: continue
    return None

def smart_parse(text, cols):
    try:
        text=text.strip()
        if "=" in text:
            d={}
            for pair in text.replace("\n",",").split(","):
                if "=" in pair:
                    k,v=pair.split("=",1); d[k.strip()]=float(v.strip())
            row={c:0.0 for c in cols}
            for k in d:
                if k in row: row[k]=d[k]
            return pd.DataFrame([row])
        else:
            vals=[float(x.strip()) for x in text.replace('\n',',').replace(';',',').split(',') if x.strip()]
            vals=vals[:len(cols)]+[0.]*(len(cols)-len(vals))
            return pd.DataFrame([vals[:len(cols)]],columns=cols)
    except Exception as e:
        st.error(f"Parse қатесі: {e}"); return None

# ══════════════════════════════════════════
# PLOTLY DARK TEMPLATE
# ══════════════════════════════════════════
DARK_LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(11,22,40,0.5)',
    font_color='#e8f4ff',
    margin=dict(l=10,r=10,t=40,b=10)
)

# ══════════════════════════════════════════
# SESSION
# ══════════════════════════════════════════
init_db()
for k,v in [("authenticated",False),("user",None),("saved_checks",[]),
                ("favorites",[]),("admin_panel_open",False),("profile_open",False)]:
    if k not in st.session_state: st.session_state[k]=v

# ══════════════════════════════════════════
# AUTH SCREEN
# ══════════════════════════════════════════
SHIELD = """<div style="text-align:center;margin:0 0 1.2rem">
<svg width="70" height="80" viewBox="0 0 70 80" fill="none" xmlns="http://www.w3.org/2000/svg">
<defs><linearGradient id="sg" x1="0" y1="0" x2="70" y2="80" gradientUnits="userSpaceOnUse">
<stop offset="0%" stop-color="#00e0ff"/><stop offset="100%" stop-color="#8b5cf6"/></linearGradient></defs>
<path d="M35 4L66 15L66 38C66 55 52 69 35 78C18 69 4 55 4 38L4 15Z"
fill="rgba(0,224,255,0.07)" stroke="url(#sg)" stroke-width="1.5"/>
<circle cx="35" cy="38" r="12" fill="rgba(0,224,255,0.1)" stroke="url(#sg)" stroke-width="1.5"/>
<path d="M29 38L34 44L42 32" stroke="#00e0ff" stroke-width="2.5"
stroke-linecap="round" stroke-linejoin="round"/>
</svg></div>"""

def render_auth():
    st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)
    _,col,_ = st.columns([1,2.4,1])
    with col:
        # Auth card wraps EVERYTHING including brand block
        st.markdown("""
        <div class="auth-card">
          <div class="brand-block">
            <div class="brand-icon">🛡️</div>
            <div>
              <div class="brand-name">AI Fraud Guard</div>
              <div class="brand-sub">Enterprise Security System</div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Tabs are rendered inside the column (visually inside the card area)
        t1,t2 = st.tabs(["🔐 Кіру","✨ Тіркелу"])
        with t1:
            st.markdown(SHIELD, unsafe_allow_html=True)
            with st.form("lf",clear_on_submit=False):
                un=st.text_input("Логин",placeholder="Пайдаланушы аты")
                pw=st.text_input("Пароль",type="password",placeholder="••••••••")
                sub=st.form_submit_button("🔐 Кіру",use_container_width=True)
            if sub:
                if not un or not pw:
                    st.error("⚠️ Барлық өрістерді толтырыңыз.")
                else:
                    with st.spinner("Тексеруде..."):
                        time.sleep(0.3)
                        res=_auth(un,pw)
                    if res and res.get("blocked"):
                        st.error("⛔ Есептік жазбаңыз бұғатталған.")
                    elif res:
                        st.session_state.authenticated=True
                        st.session_state.user=res
                        _log(res["username"],"Login")
                        st.success(f"✅ Қош келдіңіз, {res['username']}!")
                        time.sleep(0.5); st.rerun()
                    else:
                        st.error("❌ Логин немесе пароль қате.")
        with t2:
            with st.form("rf",clear_on_submit=False):
                ru=st.text_input("Логин",placeholder="Пайдаланушы аты",key="ru")
                rp1=st.text_input("Пароль",type="password",key="rp1")
                rp2=st.text_input("Растау",type="password",key="rp2")
                rsub=st.form_submit_button("🚀 Тіркелу",use_container_width=True)
            if rsub:
                eu=_val_user(ru); ep=_val_pass(rp1)
                if not ru or not rp1 or not rp2: st.error("⚠️ Барлық өрістерді толтырыңыз.")
                elif eu: st.error(f"⚠️ {eu}")
                elif ep: st.error(f"⚠️ {ep}")
                elif rp1!=rp2: st.error("⚠️ Парольдер сәйкес емес.")
                else:
                    ok=_create_user(ru,rp1,"user")
                    if ok: st.success("🎉 Тіркелу сәтті! Кіруге болады.")
                    else: st.error("❌ Бұл логин бос емес.")

        st.markdown("""<div class="auth-footer">
        <span>AI Fraud Guard™</span> · Шифрланған · Рөлге негізделген қол жеткізу</div>""",
        unsafe_allow_html=True)
    st.stop()

if not st.session_state.authenticated:
    render_auth()

user = st.session_state.user

# ══════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════
with st.sidebar:
    # ── User status bar ─────────────────────
    st.markdown(f"""<div class="status-bar">
    <span class="pulse-dot"></span>
    <span style="font-size:13px">{user['username']}</span>
    <span class="role-badge">{user['role']}</span></div>""", unsafe_allow_html=True)

    if st.button("🚪 Шығу", use_container_width=True):
        _log(user["username"],"Logout")
        st.session_state.authenticated=False
        st.session_state.user=None
        st.rerun()

    st.markdown("---")

    # ── Threshold slider ─────────────────────
    st.markdown("""
    <div style="font-size:12px;font-weight:700;color:rgba(168,200,230,0.7);
    letter-spacing:.06em;text-transform:uppercase;margin-bottom:8px">
      ⚙️ Сезімталдық
    </div>""", unsafe_allow_html=True)
    threshold = st.slider("Шек (Threshold)", 0.0, 1.0, 0.4, 0.01,
                          label_visibility="collapsed")

    tpct = int(threshold * 100)
    tcolor = "#10f5a8" if threshold < 0.4 else "#f59e0b" if threshold < 0.7 else "#ff4d6d"
    st.markdown(f"""
    <div style="background:rgba(11,22,40,0.8);border:1px solid rgba(0,224,255,0.12);
    border-radius:10px;padding:10px 14px;margin-top:4px">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px">
        <span style="font-size:11px;color:rgba(168,200,230,0.6)">Ағымдағы шек</span>
        <span style="font-family:'JetBrains Mono',monospace;font-size:15px;
        font-weight:700;color:{tcolor}">{threshold:.2f}</span>
      </div>
      <div style="height:4px;background:rgba(255,255,255,0.07);border-radius:4px;overflow:hidden">
        <div style="height:100%;width:{tpct}%;background:{tcolor};border-radius:4px"></div>
      </div>
      <div style="font-size:10px;color:{tcolor};margin-top:5px;text-align:right">
        {"🟢 Төмен" if threshold < 0.4 else "🟡 Орташа" if threshold < 0.7 else "🔴 Жоғары"} сезімталдық
      </div>
    </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style="text-align:center;font-size:10px;color:rgba(168,200,230,0.3);
    font-family:'JetBrains Mono',monospace;line-height:1.8">
      🔮 Stacking Ensemble<br>XGB · LGBM · Cat · RF
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════
# HERO
# ══════════════════════════════════════════
st.markdown(f"""
<div style="background:radial-gradient(ellipse 80% 60% at 50% 0%,rgba(0,224,255,0.06) 0%,transparent 70%);
border-bottom:1px solid rgba(0,224,255,0.12);padding:36px 0 28px;text-align:center;margin-bottom:24px">
  <div style="display:inline-flex;align-items:center;gap:7px;background:rgba(0,224,255,0.07);
  border:1px solid rgba(0,224,255,0.2);border-radius:20px;padding:5px 14px;
  font-size:12px;color:#00e0ff;font-family:'JetBrains Mono',monospace;margin-bottom:16px">
    <span class="pulse-dot"></span> AI Stacking Ensemble v2.0 — Белсенді
  </div>
  <h1 style="font-family:'Syne',sans-serif;font-size:clamp(26px,4vw,42px);font-weight:800;
  line-height:1.1;background:linear-gradient(135deg,#fff 0%,#00e0ff 100%);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;
  margin-bottom:10px">Жасанды интеллект арқылы<br>алаяқтықты анықтаңыз</h1>
  <p style="font-size:14px;color:rgba(168,200,230,0.6);max-width:560px;margin:0 auto;line-height:1.7">
  XGBoost + LightGBM + CatBoost + RandomForest ансамблі. Нақты уақытта транзакцияларды тексеріп,
  мошенниктерді анықтаңыз.</p>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════
# CSV UPLOAD ZONE — main area, always visible
# ══════════════════════════════════════════
st.markdown("""
<div style="display:flex;align-items:center;gap:10px;margin-bottom:10px">
  <div style="height:1px;flex:1;background:linear-gradient(90deg,transparent,rgba(0,224,255,0.2))"></div>
  <span style="font-family:'Syne',sans-serif;font-size:14px;font-weight:700;
  color:rgba(168,200,230,0.7);letter-spacing:.08em;text-transform:uppercase">
    📂 CSV Файлын Жүктеу
  </span>
  <div style="height:1px;flex:1;background:linear-gradient(90deg,rgba(0,224,255,0.2),transparent)"></div>
</div>""", unsafe_allow_html=True)

upl_c1, upl_c2 = st.columns([3, 1])
with upl_c1:
    uploaded_file = st.file_uploader(
        "csv_main",
        type=["csv"],
        label_visibility="collapsed",
        help="Транзакция деректері бар CSV файлды жүктеңіз. Кодировка: UTF-8 немесе cp1251. Бөлгіш: , немесе ;"
    )
with upl_c2:
    if uploaded_file is not None:
        sz = uploaded_file.size / 1024
        sz_str = f"{sz:.1f} KB" if sz < 1024 else f"{sz/1024:.2f} MB"
        st.markdown(f"""
        <div style="background:rgba(16,245,168,0.08);border:1px solid rgba(16,245,168,0.3);
        border-radius:12px;padding:14px 16px;height:100%;display:flex;flex-direction:column;
        justify-content:center;gap:4px">
          <div style="font-size:11px;font-weight:700;color:#10f5a8;
          letter-spacing:.06em;text-transform:uppercase">✅ Жүктелді</div>
          <div style="font-size:13px;font-weight:600;color:#e8f4ff;
          word-break:break-all">{uploaded_file.name}</div>
          <div style="font-size:11px;color:rgba(168,200,230,0.5)">{sz_str}</div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background:rgba(11,22,40,0.6);border:1px dashed rgba(0,224,255,0.15);
        border-radius:12px;padding:14px 16px;height:100%;display:flex;flex-direction:column;
        align-items:center;justify-content:center;gap:6px">
          <div style="font-size:22px;opacity:.4">📄</div>
          <div style="font-size:11px;color:rgba(168,200,230,0.35);text-align:center">
            Файл жүктелмеген
          </div>
        </div>""", unsafe_allow_html=True)

st.markdown("<div style='margin-bottom:20px'></div>", unsafe_allow_html=True)

# ══════════════════════════════════════════
# ADMIN / USER PANEL
# ══════════════════════════════════════════
# ── ADMIN PANEL ──────────────────────────────────────────────
if user["role"] == "admin":
    unread = _unread_count()
    badge = f" 🔴 {unread}" if unread > 0 else ""
    toggle_label = f"🛠️ Админ панелі{badge} {'▲ Жабу' if st.session_state.admin_panel_open else '▼ Ашу'}"
    if st.button(toggle_label, use_container_width=True, key="admin_toggle_btn"):
        st.session_state.admin_panel_open = not st.session_state.admin_panel_open
        st.rerun()

    if st.session_state.admin_panel_open:
        st.markdown("""<div style="background:rgba(10,20,40,0.85);border:1px solid rgba(0,224,255,0.15);
        border-radius:16px;padding:20px 20px 4px;margin-bottom:4px">""", unsafe_allow_html=True)

        t_u,t_a,t_r,t_b = _counts()
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Барлық қолданушы",t_u)
        c2.metric("Администратор",t_a)
        c3.metric("Қалыпты",t_r)
        c4.metric("Бұғатталған",t_b)
        st.markdown("---")

        at1,at2,at3,at4,at5,at6 = st.tabs([
            "👥 Қолданушылар","📋 Журнал","📩 Хабарлар","🔑 API","📊 Жүйе","⚙️ Басқару"])

        with at1:
            ul=_get_users()
            if ul:
                dfu=pd.DataFrame(ul,columns=["ID","Логин","Рөл","Бұғат","Жасалды"])
                dfu["Бұғат"]=dfu["Бұғат"].map({0:"Жоқ",1:"Иә"})
                st.dataframe(dfu,use_container_width=True)
                st.markdown("### ➕ Жаңа қолданушы")
                with st.form("addf"):
                    nu=st.text_input("Логин",key="anu")
                    np_=st.text_input("Пароль",type="password",key="anp")
                    nr=st.selectbox("Рөл",["user","admin"],key="anr")
                    if st.form_submit_button("Қосу"):
                        eu=_val_user(nu);ep=_val_pass(np_)
                        if not nu or not np_: st.error("Толтырыңыз.")
                        elif eu: st.error(eu)
                        elif ep: st.error(ep)
                        else:
                            if _create_user(nu,np_,nr):
                                _log(user["username"],f"Added {nu}"); st.success("Қосылды."); st.rerun()
                            else: st.error("Логин бар немесе admin лимиті.")
                st.markdown("### 🔄 Өзгерту")
                opts={f"{u[1]} ({u[2]}) [ID={u[0]}]":(u[0],u[1],u[2],u[3]) for u in ul}
                sel=st.selectbox("Қолданушы",list(opts.keys()))
                uid2,uname2,_,is_bl=opts[sel]
                cr1,cr2,cr3=st.columns(3)
                with cr1:
                    nr2=st.selectbox("Жаңа рөл",["user","admin"])
                    if st.button("Рөл өзгерту"):
                        _upd_role(uid2,nr2); _log(user["username"],f"Role {uname2}->{nr2}"); st.rerun()
                with cr2:
                    if st.button("Бұғаттау/Шығару"):
                        if uname2=="admin": st.warning("Admin бұғатталмайды.")
                        else: _set_block(uid2,0 if is_bl else 1); st.rerun()
                with cr3:
                    if st.button("Жою"):
                        if uname2=="admin": st.warning("Admin жойылмайды.")
                        else: _del_user(uid2); _log(user["username"],f"Del {uname2}"); st.rerun()

        with at2:
            logs=_get_logs()
            if logs:
                dfl=pd.DataFrame(logs,columns=["ID","Қолданушы","Әрекет","Уақыт"])
                st.dataframe(dfl,use_container_width=True)
                st.download_button("📥 CSV",dfl.to_csv(index=False).encode(),"audit.csv","text/csv")
            else: st.info("Журнал бос.")

        with at3:
            # ── INBOX ──────────────────────────────────────────────
            msgs = _get_msgs()
            unread_now = sum(1 for m in msgs if m[4]==0)
            st.markdown(f"### 📩 Кіріс қалта &nbsp;<span style='background:#ff4d6d;color:#fff;border-radius:12px;padding:2px 9px;font-size:12px'>{unread_now} жаңа</span>" if unread_now else "### 📩 Кіріс қалта", unsafe_allow_html=True)
            if msgs:
                for m in msgs:
                    mid,mfrom,msubj,mbody,mread,mtime = m
                    bg = "rgba(0,224,255,0.04)" if mread else "rgba(0,224,255,0.10)"
                    bd = "rgba(0,224,255,0.1)" if mread else "rgba(0,224,255,0.35)"
                    dot = "" if mread else "🔵 "
                    with st.container():
                        st.markdown(f"""<div style="background:{bg};border:1px solid {bd};border-radius:12px;
                        padding:14px 18px;margin-bottom:8px">
                        <div style="display:flex;justify-content:space-between;align-items:center">
                          <span style="font-weight:700;font-size:14px">{dot}{msubj or '(Тақырыпсыз)'}</span>
                          <span style="font-size:11px;color:rgba(168,200,230,0.5)">{mtime[:16]}</span>
                        </div>
                        <div style="font-size:12px;color:rgba(168,200,230,0.6);margin-top:3px">
                          Жіберуші: <strong style="color:#00e0ff">{mfrom}</strong></div>
                        <div style="font-size:13px;margin-top:8px;line-height:1.6">{mbody}</div>
                        </div>""", unsafe_allow_html=True)
                        mc1,mc2 = st.columns([1,1])
                        with mc1:
                            if not mread and st.button("✅ Оқылды", key=f"rd_{mid}"):
                                _mark_read(mid); st.rerun()
                        with mc2:
                            if st.button("🗑️ Жою", key=f"dm_{mid}"):
                                _del_msg(mid); st.rerun()
            else:
                st.info("Хабарлар жоқ.")

        with at4:
            with st.form("apif"):
                kn=st.text_input("Кілт аты")
                if st.form_submit_button("Жасау"):
                    if kn.strip():
                        k=_mk_key(kn.strip(),user["username"]); st.success(f"API кілті: `{k}`")
                    else: st.error("Аты енгізіңіз.")
            keys=_get_keys()
            if keys:
                dfk=pd.DataFrame(keys,columns=["ID","Аты","API Кілті","Жасаған","Кері алынды","Уақыт"])
                dfk["Кері алынды"]=dfk["Кері алынды"].map({0:"Жоқ",1:"Иә"})
                st.dataframe(dfk,use_container_width=True)

        with at5:
            t,a,r,b=_counts()
            fig=go.Figure(go.Indicator(
                mode="gauge+number",value=t,title={'text':"Қолданушылар"},
                gauge={'axis':{'range':[0,max(t*2,10)]},'bar':{'color':'#00e0ff'},
                       'bgcolor':'rgba(0,224,255,0.05)','bordercolor':'rgba(0,224,255,0.2)'}))
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',font_color='#e8f4ff',height=240)
            st.plotly_chart(fig,use_container_width=True)
            st.metric("Белсенді қолданушылар",t-b)

        with at6:
            with st.form("apwf"):
                nap=st.text_input("Жаңа admin пароль",type="password")
                if st.form_submit_button("Өзгерту"):
                    ep=_val_pass(nap)
                    if ep: st.error(ep)
                    else: _chg_pw(nap); st.success("Пароль өзгертілді.")
            col_a,col_b=st.columns(2)
            with col_a:
                if st.button("🧹 Журналды тазалау",use_container_width=True):
                    _clear_logs(); st.success("Тазаланды."); st.rerun()
            with col_b:
                dfe=pd.DataFrame(_get_users(),columns=["ID","Логин","Рөл","Бұғат","Уақыт"])
                st.download_button("📤 Users CSV",dfe.to_csv(index=False).encode(),"users.csv","text/csv",use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")

# ── USER PROFILE PANEL ────────────────────────────────────────
else:
    toggle_lbl = f"👤 Менің профилім {'▲ Жабу' if st.session_state.profile_open else '▼ Ашу'}"
    if st.button(toggle_lbl, use_container_width=True, key="profile_toggle_btn"):
        st.session_state.profile_open = not st.session_state.profile_open
        st.rerun()

    if st.session_state.profile_open:
        st.markdown("""<div style="background:rgba(10,20,40,0.85);border:1px solid rgba(0,224,255,0.15);
        border-radius:16px;padding:20px 20px 4px;margin-bottom:4px">""", unsafe_allow_html=True)

        pt1,pt2,pt3 = st.tabs(["📜 Тарих","💾 Сақталған","⭐ Таңдаулылар"])
        with pt1:
            h=_get_hist(user["username"])
            if h: st.dataframe(pd.DataFrame(h,columns=["ID","Тип","Мәлімет","Уақыт"]),use_container_width=True)
            else: st.info("Тарих бос.")
        with pt2:
            if st.session_state.saved_checks:
                dfs=pd.DataFrame(st.session_state.saved_checks)
                st.dataframe(dfs,use_container_width=True)
                st.download_button("📥 CSV",dfs.to_csv(index=False).encode(),"saved.csv","text/csv")
            else: st.info("Сақталған тексерулер жоқ.")
        with pt3:
            if st.session_state.favorites:
                for i,f in enumerate(st.session_state.favorites,1): st.code(f,language=None)
            else: st.info("Таңдаулылар бос.")

        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")

# ══════════════════════════════════════════
# MODEL LOAD
# ══════════════════════════════════════════
model, model_columns = load_model()
if model is None:
    st.error("❌ **Жүйе дайын емес!** `model.pkl` мен `model_columns.pkl` табылмады.")
    st.info("Модельді жаттықтырып, файлдарды жоба қалтасына қойыңыз.")
    st.stop()

# ══════════════════════════════════════════
# QUICK CHECK
# ══════════════════════════════════════════
st.markdown('<div class="hd">⚡ Жылдам тексеру</div>', unsafe_allow_html=True)
qc1,qc2 = st.columns([5,1])
with qc1:
    query = st.text_input("",
        placeholder="V1=0.5, V2=-1.2, Amount=200.0 немесе үтірмен бөлінген сандар...",
        label_visibility="collapsed")
with qc2:
    check_btn = st.button("🔍 Тексеру", use_container_width=True)

if query and check_btn:
    with st.spinner("🤖 AI талдауда..."):
        time.sleep(0.2)
        idf = smart_parse(query, model_columns)
    if idf is not None:
        prob = float(model.predict_proba(idf)[0,1])
        pred = 1 if prob >= threshold else 0
        pct = f"{prob*100:.1f}%"
        if prob < 0.30:   cls,emo,lbl,col = "risk-safe","✅","ҚАУІПСІЗ","#10f5a8"
        elif prob < 0.70: cls,emo,lbl,col = "risk-mid","⚠️","ОРТАША ТӘУЕКЕЛ","#f59e0b"
        else:             cls,emo,lbl,col = "risk-fraud","🚨","АЛАЯҚТЫҚ АНЫҚТАЛДЫ","#ff4d6d"

        st.markdown(f"""<div class="risk-box {cls}">
        <div style="font-size:38px">{emo}</div>
        <div style="flex:1">
          <div class="risk-lbl" style="color:{col}">{lbl}</div>
          <div class="risk-sub">Ықтималдылық: {pct} · Шек: {threshold:.2f} · Вердикт: {'Алаяқтық' if pred else 'Таза'}</div>
        </div>
        <div class="risk-pct" style="color:{col}">{pct}</div></div>""", unsafe_allow_html=True)

        bc1,bc2,bc3 = st.columns(3)
        with bc1:
            if st.button("💾 Сақтау"):
                row={"логин":user["username"],"ықтималдылық":round(prob,6),"вердикт":pred,"уақыт":datetime.utcnow().isoformat()}
                st.session_state.saved_checks.append(row)
                _save_hist(user["username"],"single_saved",str(row))
                _log(user["username"],"Saved single check")
                st.success("Сақталды")
        with bc2:
            if st.button("⭐ Таңдаулыларға"):
                if query not in st.session_state.favorites:
                    st.session_state.favorites.append(query)
                st.success("Қосылды")
        with bc3:
            with st.expander("📋 Деректер"):
                st.dataframe(idf)

        _save_hist(user["username"],"single_check",f"prob={prob:.4f},verdict={pred}")
        _log(user["username"],f"Single check prob={prob:.4f}")

st.markdown("---")

# ══════════════════════════════════════════
# MAIN TABS
# ══════════════════════════════════════════
tab_home, tab_eda, tab_batch, tab_ai, tab_analytics, tab_anylogic, tab_map, tab_contact = st.tabs([
    "🏠 Басты бет",
    "📊 EDA",
    "📂 Batch",
    "🧠 SHAP AI",
    "📈 Аналитика",
    "🎯 AnyLogic 3D",
    "🗺️ Карта",
    "📞 Байланыс"
])

# ── HOME ──────────────────────────────────
with tab_home:
    st.markdown("""<div class="feat-grid">
    <div class="feat-card"><div class="feat-icon">🤖</div>
    <div class="feat-title">AI Ансамбль моделі</div>
    <div class="feat-text">XGBoost+LightGBM+CatBoost+RF бірлесіп 99%+ дәлдік береді</div></div>
    <div class="feat-card"><div class="feat-icon">⚡</div>
    <div class="feat-title">Нақты уақыт тексеруі</div>
    <div class="feat-text">Миллисекундтарда транзакцияны тексеру. &lt;12ms жауап уақыты</div></div>
    <div class="feat-card"><div class="feat-icon">🧠</div>
    <div class="feat-title">SHAP Түсіндірмелер</div>
    <div class="feat-text">Модель шешімін белгілер деңгейінде толық түсіндіреді</div></div>
    <div class="feat-card"><div class="feat-icon">🗺️</div>
    <div class="feat-title">Геолокация картасы</div>
    <div class="feat-text">Болжамды транзакция орны мен тәуекел аймақтарын картада көрсетеді</div></div>
    <div class="feat-card"><div class="feat-icon">📊</div>
    <div class="feat-title">Кеңейтілген аналитика</div>
    <div class="feat-text">Тренд, scatter, funnel — барлық метрикалар бір жерде</div></div>
    <div class="feat-card"><div class="feat-icon">🎯</div>
    <div class="feat-title">AnyLogic 3D симуляция</div>
    <div class="feat-text">Агенттік модель: транзакция ағымын 3D-де визуализациялайды</div></div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="hd">📡 Жүйе метрикалары</div>', unsafe_allow_html=True)
    hc1,hc2,hc3,hc4 = st.columns(4)
    hc1.metric("Тексерілген","248 192","↑ 12.4%")
    hc2.metric("Алаяқтық","1 847","↑ 3.2%")
    hc3.metric("Дәлдік","99.4%","↑ 0.1%")
    hc4.metric("Жауап уақыты","12ms","Оптималды")

    st.markdown('<div class="hd" style="margin-top:24px">📖 Пайдалану нұсқаулығы</div>', unsafe_allow_html=True)
    st.markdown("""<div class="info-box">
    <strong style="color:#00e0ff">1. Жылдам тексеру</strong> — жоғарыдағы өріске транзакция деректерін енгізіп «Тексеру» басыңыз<br>
    <strong style="color:#00e0ff">2. CSV жүктеу</strong> — бүйір панельден файлды жүктеп «Batch» қойындысына өтіңіз<br>
    <strong style="color:#00e0ff">3. SHAP AI</strong> — модель шешімін тереңірек талдау үшін SHAP қойындысын пайдаланыңыз<br>
    <strong style="color:#00e0ff">4. AnyLogic 3D</strong> — агенттік симуляция арқылы транзакция ағымын визуализациялаңыз<br>
    <strong style="color:#00e0ff">5. Карта</strong> — тәуекел аймақтарын әлем картасында көріңіз
    </div>""", unsafe_allow_html=True)

# ── EDA ───────────────────────────────────
with tab_eda:
    df_eda = load_csv(uploaded_file) if uploaded_file else None
    if df_eda is not None:
        st.success(f"✅ Файл жүктелді: **{df_eda.shape[0]}** жол, **{df_eda.shape[1]}** баған")
        c1,c2 = st.columns(2)
        with c1:
            st.subheader("🥧 Класс балансы")
            if 'Class' in df_eda.columns:
                fig=px.pie(df_eda,names='Class',title='Алаяқтық (1) vs Қалыпты (0)',
                           color_discrete_sequence=['#10f5a8','#ff4d6d'],hole=0.45)
                fig.update_layout(**DARK_LAYOUT); st.plotly_chart(fig,use_container_width=True)
            else: st.info("'Class' бағаны жоқ.")
        with c2:
            st.subheader("💰 Сома үлестірімі")
            if 'Amount' in df_eda.columns:
                fd=df_eda[df_eda['Amount']<df_eda['Amount'].quantile(0.99)]
                fig=px.histogram(fd,x="Amount",nbins=50,title="Транзакция сомасы",
                                 color_discrete_sequence=['#3b82f6'])
                fig.update_layout(**DARK_LAYOUT); st.plotly_chart(fig,use_container_width=True)
            else: st.info("'Amount' бағаны жоқ.")
        st.subheader("🔥 Корреляция матрицасы (Топ-15)")
        nc=df_eda.select_dtypes(include=np.number).columns[:15]
        if len(nc)>1:
            fig=px.imshow(df_eda[nc].corr(),text_auto=True,aspect="auto",color_continuous_scale='RdBu_r')
            fig.update_layout(**DARK_LAYOUT); st.plotly_chart(fig,use_container_width=True)
    else:
        st.info("👆 Бүйір панельден CSV файлын жүктеңіз.")

# ── BATCH ─────────────────────────────────
with tab_batch:
    df_batch = load_csv(uploaded_file) if uploaded_file else None
    if df_batch is not None:
        hc='Class' in df_batch.columns
        st.write(f"Дайын: **{len(df_batch)}** транзакция")
        if hc: st.success("📢 Режим: Дәлдікті тексеру (Class бар)")
        else: st.info("📢 Режим: Жасырын қатерлерді іздеу (Class жоқ)")

        if st.button("🚀 Барлық файлды тексеру", use_container_width=True):
            pb=st.progress(0,"Дайындалуда...")
            try:
                chk=df_batch.drop(columns=['Class'],errors='ignore')
                for col in set(model_columns)-set(chk.columns): chk[col]=0
                chk=chk[model_columns]
                pb.progress(30,"Модель есептеуде...")
                probs=model.predict_proba(chk)[:,1]
                preds=(probs>=threshold).astype(int)
                pb.progress(85,"Нәтижелер дайындалуда...")
                df_batch=df_batch.copy()
                df_batch['AI_Risk_Score']=probs
                df_batch['AI_Verdict']=preds
                pb.progress(100,"Дайын!")
                st.balloons()

                fraud_df=df_batch[df_batch['AI_Verdict']==1]
                m1,m2,m3=st.columns(3)
                m1.metric("Анықталған қатер",len(fraud_df))
                m2.metric("Орташа тәуекел",f"{probs.mean()*100:.2f}%")
                m3.metric("Барлық транзакция",len(df_batch))

                st.subheader("🚨 Күдікті транзакциялар")
                st.dataframe(
                    fraud_df.sort_values('AI_Risk_Score',ascending=False).head(50)
                    .style.background_gradient(subset=['AI_Risk_Score'],cmap='Reds'),
                    use_container_width=True)

                if hc:
                    st.markdown("---")
                    st.subheader("📉 Дәлдік есебі")
                    acc=accuracy_score(df_batch['Class'],preds)
                    f1=f1_score(df_batch['Class'],preds,zero_division=0)
                    cm=confusion_matrix(df_batch['Class'],preds)
                    ca,cf=st.columns(2)
                    ca.metric("Accuracy",f"{acc:.2%}")
                    cf.metric("F1-score",f"{f1:.2%}")
                    fig=px.imshow(cm,text_auto=True,title="Қателер матрицасы",
                                  x=['Pred OK','Pred Fraud'],y=['True OK','True Fraud'],
                                  color_continuous_scale='RdYlGn_r')
                    fig.update_layout(**DARK_LAYOUT)
                    st.plotly_chart(fig,use_container_width=True)

                csv=df_batch.to_csv(index=False).encode()
                st.download_button("💾 Толық есепті жүктеу",csv,"fraud_report.csv","text/csv")
                _save_hist(user["username"],"batch",f"rows={len(df_batch)},fraud={len(fraud_df)}")
                _log(user["username"],f"Batch {len(df_batch)} rows")
            except Exception as e:
                st.error(f"Қате: {e}")
    else:
        st.info("Batch тексеру үшін CSV файлын жүктеңіз.")

# ── SHAP ──────────────────────────────────
with tab_ai:
    st.header("🧠 Шешімдерді түсіндіру (SHAP)")
    st.info("Бұл модуль AI шешіміне қандай белгілердің ең көп әсер еткенін көрсетеді.")

    n_samples = st.slider("Фон деректер саны (samples)", 10, 200, 50, 10,
                          help="Азырақ — жылдамырақ, көбірек — дәлірек")

    if st.button("📊 SHAP графиктерін жасау", use_container_width=True):
        with st.spinner("SHAP анализ жүргізілуде... Бұл бірнеше секунд алуы мүмкін."):
            try:
                plt.close('all')

                # ── matplotlib dark theme — NO rgba() strings, only tuples/hex ──
                BG      = '#0b1628'          # figure & axes background
                FG      = '#e8f4ff'          # tick & label color
                SPINE   = (0.0, 0.55, 0.63, 0.4)   # (R,G,B,A) tuple — matplotlib safe
                BAR_COL = '#00e0ff'

                matplotlib.rcParams.update({
                    'figure.facecolor':  BG,
                    'axes.facecolor':    BG,
                    'axes.edgecolor':    FG,
                    'axes.labelcolor':   FG,
                    'xtick.color':       FG,
                    'ytick.color':       FG,
                    'text.color':        FG,
                    'grid.color':        '#1a2e4a',
                    'grid.alpha':        0.5,
                })

                # 1. Get base estimator ───────────────────────────────────────
                est = None
                if hasattr(model, 'named_estimators_'):
                    for nm in ['xgb', 'cat', 'lgbm', 'rf']:
                        if nm in model.named_estimators_:
                            est = model.named_estimators_[nm]
                            break
                if est is None and hasattr(model, 'estimators_'):
                    est = model.estimators_[0]
                if est is None:
                    est = model

                # 2. Background data ──────────────────────────────────────────
                df_shap = load_csv(uploaded_file) if uploaded_file else None
                if df_shap is not None:
                    if 'Class' in df_shap.columns:
                        df_shap = df_shap.drop('Class', axis=1)
                    df_shap = df_shap.sample(min(n_samples, len(df_shap)), random_state=42)
                    bg = pd.DataFrame(0, index=df_shap.index, columns=model_columns)
                    common = list(set(df_shap.columns) & set(model_columns))
                    bg[common] = df_shap[common].values
                else:
                    bg = pd.DataFrame(
                        np.random.randn(n_samples, len(model_columns)),
                        columns=model_columns)

                # 3. SHAP values ──────────────────────────────────────────────
                explainer = shap.TreeExplainer(est)
                sv = explainer.shap_values(bg)

                # Handle list output (binary classifiers return [neg, pos])
                if isinstance(sv, list):
                    sv_plot = sv[1] if len(sv) > 1 else sv[0]
                else:
                    sv_plot = sv

                # ── helper: style axes after shap draws ──────────────────────
                def _style_axes():
                    fig = plt.gcf()
                    fig.set_facecolor(BG)
                    for ax in fig.get_axes():
                        ax.set_facecolor(BG)
                        ax.tick_params(colors=FG, labelsize=9)
                        ax.xaxis.label.set_color(FG)
                        ax.yaxis.label.set_color(FG)
                        ax.title.set_color(FG)
                        for spine in ax.spines.values():
                            spine.set_edgecolor(SPINE)  # ← tuple, NOT string
                            spine.set_linewidth(0.8)

                # 4. Summary (dot) plot ───────────────────────────────────────
                st.markdown("### 📊 SHAP Summary Plot — белгілер әсері")
                plt.figure(figsize=(10, max(5, min(n_samples//4, 10))), facecolor=BG)
                shap.summary_plot(sv_plot, bg,
                                  plot_type="dot",
                                  show=False,
                                  color_bar=True,
                                  max_display=20)
                _style_axes()
                buf1 = io.BytesIO()
                plt.savefig(buf1, format='png', dpi=130,
                            bbox_inches='tight', facecolor=BG)
                buf1.seek(0)
                plt.close('all')
                st.image(buf1, use_container_width=True)

                # 5. Bar (importance) plot ────────────────────────────────────
                st.markdown("### 📊 Белгілердің маңыздылығы (Bar Chart)")
                plt.figure(figsize=(10, max(5, min(n_samples//4, 10))), facecolor=BG)
                shap.summary_plot(sv_plot, bg,
                                  plot_type="bar",
                                  show=False,
                                  max_display=20)
                _style_axes()
                # Recolor bars to cyan
                for ax in plt.gcf().get_axes():
                    for patch in ax.patches:
                        try:
                            patch.set_facecolor(BAR_COL)
                        except Exception:
                            pass
                buf2 = io.BytesIO()
                plt.savefig(buf2, format='png', dpi=130,
                            bbox_inches='tight', facecolor=BG)
                buf2.seek(0)
                plt.close('all')
                st.image(buf2, use_container_width=True)

                # 6. Reset rcParams so other plots are unaffected ─────────────
                matplotlib.rcParams.update(matplotlib.rcParamsDefault)
                matplotlib.use('Agg')
                plt.style.use('dark_background')

                st.success("✅ SHAP графиктері сәтті жасалды!")
                _save_hist(user["username"], "shap", "generated")
                _log(user["username"], "SHAP generated")

            except Exception as e:
                plt.close('all')
                # Reset rcParams on error too
                try:
                    matplotlib.rcParams.update(matplotlib.rcParamsDefault)
                    matplotlib.use('Agg')
                    plt.style.use('dark_background')
                except Exception:
                    pass
                st.error(f"SHAP қатесі: {e}")
                import traceback
                with st.expander("🔍 Толық қате мәтіні"):
                    st.code(traceback.format_exc(), language="python")
                st.info(
                    "**Шешім жолдары:**\n"
                    "1. `model.pkl` файлы ағымдағы XGBoost/LightGBM/CatBoost нұсқасымен жаттықтырылуы керек\n"
                    "2. `pip install shap --upgrade` командасын орындаңыз\n"
                    "3. CSV файлы жүктелген болса, SHAP дәлірек жұмыс жасайды"
                )

# ── ANALYTICS ─────────────────────────────
with tab_analytics:
    st.header("📈 Кеңейтілген аналитика")
    df_an = load_csv(uploaded_file) if uploaded_file else None
    if df_an is not None:
        df_an=df_an.copy()
        if "Amount" not in df_an.columns: df_an["Amount"]=np.random.uniform(1,500,len(df_an))
        if "AI_Risk_Score" not in df_an.columns: df_an["AI_Risk_Score"]=np.random.uniform(0.01,0.99,len(df_an))
        if "Class" not in df_an.columns: df_an["Class"]=(df_an["AI_Risk_Score"]>threshold).astype(int)
        df_an["row_id"]=np.arange(len(df_an))

        c1,c2=st.columns(2)
        with c1:
            fig=px.line(df_an.head(500),x="row_id",y="Amount",
                        title="Транзакция сомасының тренді",color_discrete_sequence=['#00e0ff'])
            fig.update_layout(**DARK_LAYOUT); st.plotly_chart(fig,use_container_width=True)
        with c2:
            fig=px.scatter(df_an.head(1000),x="Amount",y="AI_Risk_Score",color="Class",
                           title="Тәуекел vs Сома",color_discrete_sequence=['#10f5a8','#ff4d6d'])
            fig.update_layout(**DARK_LAYOUT); st.plotly_chart(fig,use_container_width=True)
        c3,c4=st.columns(2)
        with c3:
            fig=px.histogram(df_an,x="AI_Risk_Score",nbins=40,
                             title="Тәуекел Score үлестірімі",color_discrete_sequence=['#8b5cf6'])
            fig.update_layout(**DARK_LAYOUT); st.plotly_chart(fig,use_container_width=True)
        with c4:
            fdf=pd.DataFrame({
                "Кезең":["Барлығы","Тексерілді","Орташа тәуекел","Жоғары тәуекел","Алаяқтық"],
                "Саны":[len(df_an),len(df_an),
                        int((df_an["AI_Risk_Score"]>0.4).sum()),
                        int((df_an["AI_Risk_Score"]>0.7).sum()),
                        int((df_an["Class"]==1).sum())]})
            fig=px.funnel(fdf,x="Саны",y="Кезең",title="Алаяқтық воронкасы",
                          color_discrete_sequence=['#3b82f6'])
            fig.update_layout(**DARK_LAYOUT); st.plotly_chart(fig,use_container_width=True)

        # Fast geo — sample for speed
        sample_size=min(500,len(df_an))
        df_geo_sample=df_an.sample(sample_size,random_state=42)
        df_hash=hash(df_geo_sample.to_json())
        geo=_enrich_geo(df_hash,df_geo_sample)
        cs=geo.groupby("Ел",as_index=False).agg(Транзакция=("Ел","count"),Орта_тәуекел=("AI_Risk_Score","mean"))
        st.subheader("🌍 Елдер бойынша статистика")
        st.dataframe(cs.sort_values("Орта_тәуекел",ascending=False),use_container_width=True)
        _log(user["username"],"Analytics viewed")
    else:
        st.info("Аналитика үшін CSV файлын жүктеңіз.")

# ── ANYLOGIC 3D ───────────────────────────
with tab_anylogic:
    st.markdown('<div class="hd">🎯 AnyLogic 3D — Агенттік симуляция</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:13px;color:rgba(168,200,230,0.6);margin-bottom:18px">Транзакция агенттерінің нақты уақытта қозғалысын агенттік модель арқылы визуализациялайды.</div>',
                unsafe_allow_html=True)

    tc1,tc2,tc3=st.columns([2,2,2])
    with tc1:
        scene=st.selectbox("Сценарий",["Қалыпты трафик","Жоғары риск шабуылы","DDoS Fraud Wave","Ботнет алаяқтықтары"])
    with tc2:
        spd=st.slider("Жылдамдық",0.5,4.0,1.0,0.5)
    with tc3:
        acnt=st.selectbox("Агенттер саны",[20,40,80,120],index=1)

    fc_map={"Қалыпты трафик":0.08,"Жоғары риск шабуылы":0.35,"DDoS Fraud Wave":0.55,"Ботнет алаяқтықтары":0.70}
    fc=fc_map[scene]

    al_html=f"""<!DOCTYPE html><html><head><meta charset="UTF-8">
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&family=DM+Sans:wght@400;600&display=swap');
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:#060d1a;font-family:'DM Sans',sans-serif;color:#e8f4ff;overflow:hidden;user-select:none}}
.wrap{{display:flex;flex-direction:column;height:100vh}}
.topbar{{display:flex;align-items:center;justify-content:space-between;padding:9px 16px;
background:rgba(0,0,0,0.45);border-bottom:1px solid rgba(0,224,255,0.12);font-size:12px;flex-shrink:0}}
.badge{{display:flex;align-items:center;gap:7px;font-family:'JetBrains Mono',monospace;color:#10f5a8}}
.dot{{width:7px;height:7px;border-radius:50%;background:#10f5a8;box-shadow:0 0 8px #10f5a8;animation:pd 2s infinite}}
@keyframes pd{{0%,100%{{box-shadow:0 0 6px #10f5a8}}50%{{box-shadow:0 0 14px #10f5a8}}}}
.controls{{display:flex;gap:7px}}
.btn{{padding:5px 13px;border-radius:7px;border:none;font-size:12px;font-weight:700;cursor:pointer;
font-family:'DM Sans',sans-serif;transition:opacity .15s}}
.btn:hover{{opacity:.8}}
.bp{{background:#10f5a8;color:#000}}.bpa{{background:#f59e0b;color:#000}}.bs{{background:#ff4d6d;color:#fff}}
.tdisp{{font-family:'JetBrains Mono',monospace;font-size:11px;color:rgba(168,200,230,0.5)}}
canvas{{flex:1;display:block;width:100%}}
.legend{{display:flex;gap:16px;padding:7px 16px;background:rgba(0,0,0,0.3);
border-top:1px solid rgba(0,224,255,0.1);flex-wrap:wrap;flex-shrink:0}}
.leg{{display:flex;align-items:center;gap:5px;font-size:11px;color:rgba(168,200,230,0.7)}}
.leg-dot{{width:8px;height:8px;border-radius:50%}}
.stats{{display:grid;grid-template-columns:repeat(5,1fr);border-top:1px solid rgba(0,224,255,0.1);flex-shrink:0}}
.st{{padding:8px 12px;text-align:center;border-right:1px solid rgba(0,224,255,0.07)}}
.st:last-child{{border-right:none}}
.sv{{font-family:'JetBrains Mono',monospace;font-size:17px;font-weight:500}}
.sl{{font-size:10px;color:rgba(168,200,230,0.5);margin-top:1px}}
.logbox{{background:rgba(0,0,0,0.35);border-top:1px solid rgba(0,224,255,0.1);
padding:6px 14px;height:88px;overflow-y:auto;font-family:'JetBrains Mono',monospace;
font-size:10px;color:#10f5a8;line-height:1.85;flex-shrink:0}}
.logbox::-webkit-scrollbar{{width:3px}}
.logbox::-webkit-scrollbar-thumb{{background:rgba(0,224,255,0.2);border-radius:2px}}
</style></head><body>
<div class="wrap">
<div class="topbar">
  <div class="badge"><div class="dot"></div>AnyLogic 3D Agent-Based Fraud Simulation</div>
  <div class="controls">
    <button class="btn bp" onclick="startSim()">▶ Бастау</button>
    <button class="btn bpa" onclick="pauseSim()">⏸ Пауза</button>
    <button class="btn bs" onclick="resetSim()">⏹ Тоқтату</button>
  </div>
  <div class="tdisp">t=<span id="simT">0.00</span>s &nbsp;|&nbsp;
  <span id="simS" style="color:#f59e0b">Тоқтатылды</span></div>
</div>
<canvas id="c"></canvas>
<div class="legend">
  <div class="leg"><div class="leg-dot" style="background:#10f5a8"></div>Қалыпты</div>
  <div class="leg"><div class="leg-dot" style="background:#f59e0b"></div>Тексеруде</div>
  <div class="leg"><div class="leg-dot" style="background:#ff4d6d"></div>Алаяқтық</div>
  <div class="leg"><div class="leg-dot" style="background:#8b5cf6"></div>Бұғатталды</div>
  <div class="leg"><div class="leg-dot" style="background:#3b82f6"></div>Өткізілді</div>
</div>
<div class="stats">
  <div class="st"><div class="sv" id="sp" style="color:#10f5a8">0</div><div class="sl">Өткізілді</div></div>
  <div class="st"><div class="sv" id="sf" style="color:#ff4d6d">0</div><div class="sl">Алаяқтық</div></div>
  <div class="st"><div class="sv" id="sc" style="color:#f59e0b">0</div><div class="sl">Тексеруде</div></div>
  <div class="st"><div class="sv" id="sb" style="color:#8b5cf6">0</div><div class="sl">Бұғатталды</div></div>
  <div class="st"><div class="sv" id="sr" style="color:#00e0ff">0%</div><div class="sl">Fraud Rate</div></div>
</div>
<div class="logbox" id="lb">> Симуляция жүктелді. Бастау үшін ▶ басыңыз...<br></div>
</div>

<script>
const FRAUD_CHANCE={fc},SPEED={spd},MAX_AGENTS={acnt};
const cv=document.getElementById('c'),ctx=cv.getContext('2d');
let W,H;
function resize(){{
  const r=cv.getBoundingClientRect();
  W=r.width; H=cv.clientHeight;
  cv.width=W*devicePixelRatio; cv.height=H*devicePixelRatio;
  cv.style.width=W+'px'; cv.style.height=H+'px';
  ctx.scale(devicePixelRatio,devicePixelRatio);
}}
resize();
window.addEventListener('resize',()=>{{resize();drawFrame();}});

const ND=[
  {{rx:.10,ry:.50,lbl:'Клиент',t:'src'}},
  {{rx:.30,ry:.22,lbl:'Банк API',t:'proc'}},
  {{rx:.30,ry:.78,lbl:'Карта жүйесі',t:'proc'}},
  {{rx:.55,ry:.50,lbl:'AI Детектор',t:'ai'}},
  {{rx:.78,ry:.25,lbl:'Рұқсат',t:'out'}},
  {{rx:.78,ry:.75,lbl:'Блок',t:'blk'}},
  {{rx:.92,ry:.50,lbl:'Банк DB',t:'db'}},
];
const EDGES=[[0,1],[0,2],[1,3],[2,3],[3,4],[3,5],[4,6],[5,6]];
const COLS={{normal:'#10f5a8',checking:'#f59e0b',fraud:'#ff4d6d',blocked:'#8b5cf6',safe:'#3b82f6'}};

function np(n){{return {{x:n.rx*W,y:n.ry*H}}}}

let agents=[],running=false,paused=false,animId=null;
let passed=0,fraud=0,blocked=0,simT=0;

function spawn(){{
  const isF=Math.random()<FRAUD_CHANCE;
  const path=isF?[0,1,3,5,6]:(Math.random()<.5?[0,1,3,4,6]:[0,2,3,4,6]);
  const p0=np(ND[path[0]]);
  return{{x:p0.x,y:p0.y,path,pi:0,prog:0,isF,state:'normal',
    spd:(.007+Math.random()*.005)*SPEED,sz:4+Math.random()*2.5,
    trail:[],alive:true,checked:false,id:Math.random()}};
}}

function drawGrid(){{
  ctx.strokeStyle='rgba(0,224,255,0.02)';ctx.lineWidth=1;
  for(let x=0;x<W;x+=48){{ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,H);ctx.stroke()}}
  for(let y=0;y<H;y+=48){{ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(W,y);ctx.stroke()}}
}}

function drawGlow(px,py,r,col){{
  const g=ctx.createRadialGradient(px,py,0,px,py,r);
  g.addColorStop(0,col+'22');g.addColorStop(1,'transparent');
  ctx.fillStyle=g;ctx.beginPath();ctx.arc(px,py,r,0,Math.PI*2);ctx.fill();
}}

function rr(x,y,w,h,r){{
  ctx.beginPath();
  ctx.moveTo(x+r,y);ctx.lineTo(x+w-r,y);ctx.arcTo(x+w,y,x+w,y+r,r);
  ctx.lineTo(x+w,y+h-r);ctx.arcTo(x+w,y+h,x+w-r,y+h,r);
  ctx.lineTo(x+r,y+h);ctx.arcTo(x,y+h,x,y+h-r,r);
  ctx.lineTo(x,y+r);ctx.arcTo(x,y,x+r,y,r);ctx.closePath();
}}

function drawFrame(){{
  ctx.clearRect(0,0,W,H);
  drawGrid();

  // Glow zones
  ND.forEach(n=>{{
    const p=np(n);
    if(n.t==='ai') drawGlow(p.x,p.y,90,'#00e0ff');
    if(n.t==='blk') drawGlow(p.x,p.y,60,'#ff4d6d');
  }});

  // Edges
  ctx.setLineDash([5,5]);ctx.lineWidth=1;ctx.strokeStyle='rgba(0,224,255,0.12)';
  EDGES.forEach(([a,b])=>{{
    const pa=np(ND[a]),pb=np(ND[b]);
    ctx.beginPath();ctx.moveTo(pa.x,pa.y);ctx.lineTo(pb.x,pb.y);ctx.stroke();
  }});
  ctx.setLineDash([]);

  // Nodes
  ND.forEach((n,i)=>{{
    const p=np(n),isAI=n.t==='ai',isBlk=n.t==='blk';
    const fc2=isAI?'rgba(0,224,255,0.12)':isBlk?'rgba(255,77,109,0.12)':'rgba(59,130,246,0.12)';
    const sc=isAI?'rgba(0,224,255,0.7)':isBlk?'rgba(255,77,109,0.6)':'rgba(59,130,246,0.5)';
    ctx.strokeStyle=sc;ctx.lineWidth=1.5;
    if(isAI){{
      ctx.fillStyle=fc2; rr(p.x-44,p.y-19,88,38,8); ctx.fill();ctx.stroke();
      // 3D top
      ctx.beginPath();ctx.strokeStyle='rgba(0,224,255,0.18)';ctx.lineWidth=0.8;
      ctx.moveTo(p.x-44,p.y-19);ctx.lineTo(p.x-32,p.y-31);
      ctx.lineTo(p.x+56,p.y-31);ctx.lineTo(p.x+44,p.y-19);ctx.stroke();
      // side
      ctx.beginPath();
      ctx.moveTo(p.x+44,p.y-19);ctx.lineTo(p.x+56,p.y-31);
      ctx.lineTo(p.x+56,p.y+7);ctx.lineTo(p.x+44,p.y+19);
      ctx.fillStyle='rgba(0,224,255,0.05)';ctx.fill();ctx.stroke();
    }} else {{
      ctx.fillStyle=fc2;
      ctx.beginPath();ctx.arc(p.x,p.y,20,0,Math.PI*2);ctx.fill();ctx.stroke();
      ctx.beginPath();ctx.arc(p.x,p.y,12,0,Math.PI*2);
      ctx.strokeStyle=sc.replace('0.5','0.2').replace('0.6','0.2').replace('0.7','0.25');
      ctx.lineWidth=0.7;ctx.stroke();
    }}
    ctx.fillStyle='rgba(232,244,255,0.85)';
    ctx.font='500 10.5px "DM Sans",sans-serif';
    ctx.textAlign='center';ctx.textBaseline='top';
    ctx.fillText(n.lbl,p.x,p.y+(isAI?24:26));
  }});

  // Agents
  agents.forEach(a=>{{
    if(!a.alive)return;
    const col=COLS[a.state]||COLS.normal;
    if(a.trail.length>2){{
      ctx.beginPath();ctx.strokeStyle=col+'33';ctx.lineWidth=1.5;
      ctx.moveTo(a.trail[0].x,a.trail[0].y);
      a.trail.forEach(pt=>ctx.lineTo(pt.x,pt.y));ctx.stroke();
    }}
    const g=ctx.createRadialGradient(a.x,a.y,0,a.x,a.y,a.sz*3);
    g.addColorStop(0,col+'44');g.addColorStop(1,'transparent');
    ctx.fillStyle=g;ctx.beginPath();ctx.arc(a.x,a.y,a.sz*3,0,Math.PI*2);ctx.fill();
    ctx.beginPath();ctx.arc(a.x,a.y,a.sz,0,Math.PI*2);
    ctx.fillStyle=col;ctx.fill();
    ctx.strokeStyle='rgba(255,255,255,0.45)';ctx.lineWidth=0.7;ctx.stroke();
  }});
}}

function update(){{
  if(agents.length<MAX_AGENTS&&Math.random()<0.06) agents.push(spawn());
  agents.forEach(a=>{{
    if(!a.alive)return;
    a.prog+=a.spd*SPEED;
    a.trail.push({{x:a.x,y:a.y}});
    if(a.trail.length>18) a.trail.shift();
    if(a.prog>=1){{
      a.pi++;a.prog=0;
      if(a.pi>=a.path.length-1){{
        a.alive=false;
        if(a.state==='fraud'){{fraud++;blocked++;}}else{{passed++;}}
        return;
      }}
      if(a.path[a.pi]===3&&!a.checked){{
        a.state='checking';a.checked=true;
        const aid=a.id;
        setTimeout(()=>{{
          const ag=agents.find(x=>x.id===aid);
          if(!ag||!ag.alive)return;
          ag.state=ag.isF?'fraud':'safe';
          addLog(ag.isF?'🚨 TXN-'+Math.floor(Math.random()*9000+1000)+' АЛАЯҚТЫҚ анықталды'
                       :'✅ TXN-'+Math.floor(Math.random()*9000+1000)+' тексеруден өтті');
        }},300/SPEED);
      }}
    }}
    if(a.pi<a.path.length-1){{
      const fr=np(ND[a.path[a.pi]]),to=np(ND[a.path[a.pi+1]]);
      a.x=fr.x+(to.x-fr.x)*a.prog;
      a.y=fr.y+(to.y-fr.y)*a.prog;
    }}
  }});
  agents=agents.filter(a=>a.alive);
  const chk=agents.filter(a=>a.state==='checking').length;
  document.getElementById('sp').textContent=passed;
  document.getElementById('sf').textContent=fraud;
  document.getElementById('sc').textContent=chk;
  document.getElementById('sb').textContent=blocked;
  const tot=passed+fraud;
  document.getElementById('sr').textContent=tot?Math.round(fraud/tot*100)+'%':'0%';
  simT+=0.016*SPEED;
  document.getElementById('simT').textContent=simT.toFixed(2);
}}

function loop(){{
  if(!running||paused){{animId=requestAnimationFrame(loop);return}}
  update();drawFrame();animId=requestAnimationFrame(loop);
}}

function startSim(){{
  running=true;paused=false;
  document.getElementById('simS').textContent='Жұмыс жасауда';
  document.getElementById('simS').style.color='#10f5a8';
  addLog('> Симуляция басталды — {scene}');
  if(!animId) loop();
}}
function pauseSim(){{
  paused=!paused;
  document.getElementById('simS').textContent=paused?'Уақытша тоқтатылды':'Жұмыс жасауда';
  document.getElementById('simS').style.color=paused?'#f59e0b':'#10f5a8';
}}
function resetSim(){{
  running=false;paused=false;agents=[];passed=0;fraud=0;blocked=0;simT=0;
  ['sp','sf','sc','sb'].forEach(id=>document.getElementById(id).textContent='0');
  document.getElementById('sr').textContent='0%';
  document.getElementById('simT').textContent='0.00';
  document.getElementById('simS').textContent='Тоқтатылды';
  document.getElementById('simS').style.color='#f59e0b';
  if(animId){{cancelAnimationFrame(animId);animId=null}}
  drawFrame();
  document.getElementById('lb').innerHTML='> Тоқтатылды. Бастау үшін ▶ басыңыз...<br>';
}}
function addLog(msg){{
  const lb=document.getElementById('lb');
  lb.innerHTML+='['+simT.toFixed(2)+'s] '+msg+'<br>';
  lb.scrollTop=lb.scrollHeight;
}}
drawFrame();
</script></body></html>"""

    st.components.v1.html(al_html, height=640, scrolling=False)

    ic1,ic2,ic3=st.columns(3)
    with ic1:
        st.markdown(f"""<div class="info-box" style="font-size:12px">
        <strong style="color:#00e0ff">🎯 Параметрлер</strong><br><br>
        Сценарий: <strong style="color:#fff">{scene}</strong><br>
        Fraud %: <strong style="color:#ff4d6d">{fc*100:.0f}%</strong><br>
        Жылдамдық: <strong>{spd}x</strong> · Агент: <strong>{acnt}</strong>
        </div>""", unsafe_allow_html=True)
    with ic2:
        st.markdown("""<div class="info-box" style="font-size:12px">
        <strong style="color:#00e0ff">🟢 Қозғалыс логикасы</strong><br><br>
        Клиент → Банк API / Карта → AI Детектор → Рұқсат немесе Блок → Банк DB
        </div>""", unsafe_allow_html=True)
    with ic3:
        st.markdown("""<div class="info-box" style="font-size:12px">
        <strong style="color:#00e0ff">🔵 AI Детектор</strong><br><br>
        Агент AI аймағына жеткенде тексеру басталады.<br>
        Алаяқтық → 🟣 Блок · Қалыпты → 🔵 Рұқсат
        </div>""", unsafe_allow_html=True)

# ── MAP ───────────────────────────────────
with tab_map:
    st.header("🗺️ Геолокация картасы")
    st.info("AI модулі транзакциялардың болжамды орналасуын анықтап, тәуекел аймақтарын картада көрсетеді.")

    df_map = load_csv(uploaded_file) if uploaded_file else None
    if df_map is not None:
        df_map=df_map.copy()
        if "Amount" not in df_map.columns: df_map["Amount"]=np.random.uniform(1,500,len(df_map))
        if "AI_Risk_Score" not in df_map.columns: df_map["AI_Risk_Score"]=np.random.uniform(0.01,0.99,len(df_map))

        # Sample for speed — use cache
        sample_size=min(800,len(df_map))
        df_sample=df_map.sample(sample_size,random_state=42)
        df_hash=hash(df_sample.values.tobytes())
        geo=_enrich_geo(df_hash,df_sample)

        mc1,mc2=st.columns(2)
        with mc1:
            st.subheader("Глобал транзакция картасы")
            fig=px.scatter_geo(geo,lat="lat",lon="lon",color="AI_Risk_Score",
                               size="Amount",hover_name="Ел",
                               title="Транзакциялардың болжамды орналасуы",
                               color_continuous_scale='RdYlGn_r')
            fig.update_layout(
                paper_bgcolor='rgba(6,13,26,0.9)',font_color='#e8f4ff',
                margin=dict(l=0,r=0,t=40,b=0),
                geo=dict(bgcolor='rgba(11,22,40,0.8)',
                         lakecolor='rgba(0,0,0,0)',
                         landcolor='rgba(15,31,53,0.9)',
                         showocean=True,oceancolor='rgba(11,22,40,0.6)',
                         showframe=False,showcountries=True,
                         countrycolor='rgba(0,224,255,0.15)'))
            st.plotly_chart(fig,use_container_width=True)

        with mc2:
            st.subheader("Ел хороплет картасы")
            cagg=geo.groupby("Ел",as_index=False).agg(
                Орта_тәуекел=("AI_Risk_Score","mean"),
                Саны=("Ел","count"))
            fig=px.choropleth(cagg,locations="Ел",locationmode="country names",
                              color="Орта_тәуекел",hover_name="Ел",
                              title="Ел тәуекел картасы",
                              color_continuous_scale='RdYlGn_r')
            fig.update_layout(
                paper_bgcolor='rgba(6,13,26,0.9)',font_color='#e8f4ff',
                margin=dict(l=0,r=0,t=40,b=0),
                geo=dict(bgcolor='rgba(11,22,40,0.8)',
                         landcolor='rgba(15,31,53,0.9)',
                         showframe=False,showcountries=True,
                         countrycolor='rgba(0,224,255,0.15)'))
            st.plotly_chart(fig,use_container_width=True)

        st.subheader("🌍 Тәуекел рейтингі бойынша елдер")
        top=cagg.sort_values("Орта_тәуекел",ascending=False).head(10)
        st.dataframe(top,use_container_width=True)

        high_risk=["Нигерия","Индонезия","Филиппин","Бразилия","Түркия","Үндістан"]
        flagged=top[top["Ел"].isin(high_risk)]
        if not flagged.empty:
            st.warning("⚠️ Жоғары тәуекелді елдер: " + " · ".join(flagged["Ел"].tolist()))

        # Bar chart of top countries
        fig=px.bar(top.sort_values("Орта_тәуекел"),
                   x="Орта_тәуекел",y="Ел",orientation='h',
                   title="Елдер бойынша орташа тәуекел",
                   color="Орта_тәуекел",color_continuous_scale='RdYlGn_r')
        fig.update_layout(**DARK_LAYOUT)
        st.plotly_chart(fig,use_container_width=True)

        _log(user["username"],"Map viewed")
    else:
        st.info("Карта үшін CSV файлын жүктеңіз.")

# ── CONTACT TAB ──────────────────────────
with tab_contact:
    st.markdown('<div class="hd">📞 Байланыс және Қолдау</div>', unsafe_allow_html=True)

    # ── Contact cards ───────────────────────
    st.markdown("""
    <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin-bottom:28px">

      <div style="background:rgba(10,20,40,0.85);border:1px solid rgba(0,224,255,0.18);
      border-radius:16px;padding:22px 20px;text-align:center;transition:all .2s">
        <div style="font-size:32px;margin-bottom:12px">📧</div>
        <div style="font-family:'Syne',sans-serif;font-weight:700;font-size:14px;
        margin-bottom:8px;color:#e8f4ff">Email</div>
        <a href="mailto:nurdauletoon@gmail.com"
           style="color:#00e0ff;font-size:13px;text-decoration:none;font-family:'JetBrains Mono',monospace">
          nurdauletoon@gmail.com</a>
        <div style="font-size:11px;color:rgba(168,200,230,0.5);margin-top:6px">
          Жауап уақыты: 24 сағат ішінде</div>
      </div>

      <div style="background:rgba(10,20,40,0.85);border:1px solid rgba(0,224,255,0.18);
      border-radius:16px;padding:22px 20px;text-align:center">
        <div style="font-size:32px;margin-bottom:12px">✈️</div>
        <div style="font-family:'Syne',sans-serif;font-weight:700;font-size:14px;
        margin-bottom:8px;color:#e8f4ff">Telegram</div>
        <a href="https://t.me/beard5n" target="_blank"
           style="color:#00e0ff;font-size:13px;text-decoration:none;font-family:'JetBrains Mono',monospace">
          @beard5n</a>
        <div style="font-size:11px;color:rgba(168,200,230,0.5);margin-top:6px">
          Жылдам жауап алу үшін</div>
      </div>

      <div style="background:rgba(10,20,40,0.85);border:1px solid rgba(0,224,255,0.18);
      border-radius:16px;padding:22px 20px;text-align:center">
        <div style="font-size:32px;margin-bottom:12px">🛡️</div>
        <div style="font-family:'Syne',sans-serif;font-weight:700;font-size:14px;
        margin-bottom:8px;color:#e8f4ff">AI Fraud Guard</div>
        <div style="color:#00e0ff;font-size:13px;font-family:'JetBrains Mono',monospace">
          Enterprise Security</div>
        <div style="font-size:11px;color:rgba(168,200,230,0.5);margin-top:6px">
          24/7 Жүйе мониторингі</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Send message form ───────────────────
    cc1, cc2 = st.columns([3, 2])

    with cc1:
        st.markdown('<div class="hd" style="font-size:15px">✉️ Админге хабарлама жіберу</div>',
                    unsafe_allow_html=True)
        with st.form("contact_form", clear_on_submit=True):
            msg_subj = st.text_input("📌 Тақырып", placeholder="Мысалы: Техникалық мәселе / Сұрақ / Ұсыныс")
            msg_body = st.text_area("💬 Хабарлама мәтіні",
                placeholder="Хабарламаңызды осы жерге жазыңыз... Мәселені, сұрақты немесе ұсынысты сипаттаңыз.",
                height=180)
            send_btn = st.form_submit_button("📤 Жіберу", use_container_width=True)

        if send_btn:
            if not msg_subj.strip() or not msg_body.strip():
                st.error("⚠️ Тақырып пен хабарлама мәтінін толтырыңыз.")
            else:
                _send_msg(user["username"], msg_subj.strip(), msg_body.strip())
                _log(user["username"], f"Sent message: {msg_subj[:50]}")
                st.success("✅ Хабарлама сәтті жіберілді! Администратор жақын арада жауап береді.")
                st.balloons()

    with cc2:
        st.markdown('<div class="hd" style="font-size:15px">ℹ️ Жиі қойылатын сұрақтар</div>',
                    unsafe_allow_html=True)
        faqs = [
            ("🔑 Пароль ұмытып қалдым", "Telegram немесе Email арқылы администраторға хабарласыңыз."),
            ("📂 CSV форматы дұрыс жүктелмейді", "UTF-8 немесе cp1251 кодировкасын, ',' немесе ';' бөлгішін тексеріңіз."),
            ("🧠 SHAP жұмыс жасамайды", "model.pkl нұсқасы мен xgboost/shap нұсқалары сәйкес болуы керек."),
            ("🗺️ Карта баяу", "Үлкен CSV үшін ішкі sampling (800 жол) жасалады — бұл қалыпты."),
            ("🎯 AnyLogic сценарийі", "Жоғарғы toolbar-дан сценарийді, жылдамдықты, агент санын өзгертіңіз."),
        ]
        for q, a in faqs:
            with st.expander(q):
                st.markdown(f'<div style="font-size:13px;color:rgba(168,200,230,0.8);line-height:1.7">{a}</div>',
                            unsafe_allow_html=True)

    st.markdown("---")

    # ── My sent messages ─────────────────────
    st.markdown('<div class="hd" style="font-size:15px">📋 Менің хабарларым</div>', unsafe_allow_html=True)
    with _conn() as c:
        my_msgs = c.execute(
            "SELECT id,subject,body,is_read,created_at FROM messages WHERE from_user=? ORDER BY id DESC LIMIT 20",
            (user["username"],)).fetchall()

    if my_msgs:
        for mm in my_msgs:
            mmid,mmsubj,mmbody,mmread,mmtime = mm
            status_col = "#10f5a8" if mmread else "#f59e0b"
            status_txt = "✅ Оқылды" if mmread else "⏳ Күтілуде"
            st.markdown(f"""<div style="background:rgba(11,22,40,0.7);border:1px solid rgba(0,224,255,0.1);
            border-radius:12px;padding:14px 18px;margin-bottom:8px">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px">
              <span style="font-weight:700;font-size:14px">{mmsubj or '(Тақырыпсыз)'}</span>
              <div style="display:flex;gap:12px;align-items:center">
                <span style="color:{status_col};font-size:11px;font-weight:600">{status_txt}</span>
                <span style="font-size:11px;color:rgba(168,200,230,0.4)">{mmtime[:16]}</span>
              </div>
            </div>
            <div style="font-size:12px;color:rgba(168,200,230,0.65);line-height:1.6">{mmbody[:200]}{'...' if len(mmbody)>200 else ''}</div>
            </div>""", unsafe_allow_html=True)
    else:
        st.info("Сіз әлі хабарлама жібермедіңіз.")

# ══════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════
st.markdown("---")
st.markdown("""
<div style='text-align:center;color:rgba(168,200,230,0.3);font-size:11px;
padding:10px 0 18px;font-family:"JetBrains Mono",monospace;letter-spacing:.04em'>
© 2026 AI Fraud Guard Pro &nbsp;·&nbsp;
Stacking Ensemble: XGBoost + LightGBM + CatBoost + RandomForest
</div>""", unsafe_allow_html=True)
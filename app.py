#!/usr/bin/env python3
"""
SQL Storm Model Evaluator — Streamlit frontend

Run with:
    streamlit run app.py
"""

import os
import sys
import re
import time
import queue
import signal
import threading
import subprocess
from pathlib import Path

import pandas as pd
import streamlit as st

# ── Page configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SQL Storm Evaluator",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Constants ─────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
TEMP_DIR = PROJECT_ROOT / "temp_uploads"
PROGRESS_RE = re.compile(r"Completed\s+(\d+)/(\d+)")
GEN_PROGRESS_RE = re.compile(r"Generated\s+(\d+)/(\d+)\s+plans")

MODEL_PRESETS = [
    "openai-community/gpt2",
    "meta-llama/Llama-3.2-3B",
    "meta-llama/Llama-3.2-1B",
    "meta-llama/Meta-Llama-3-8B",
    "mistralai/Mistral-7B-v0.1",
    "microsoft/phi-2",
    "microsoft/Phi-3-mini-4k-instruct",
]

# ── Session state initialisation ──────────────────────────────────────────────
_DEFAULTS: dict = dict(
    uploaded_csv_path=None,
    uploaded_csv_name=None,
    uploaded_file_id=None,
    csv_columns=[],
    is_running=False,
    process=None,
    log_queue=None,
    log_lines=[],
    gen_current=0,
    gen_total=0,
    progress_current=0,
    progress_total=0,
    run_completed=False,
    run_failed=False,
    results_df=None,
    current_output_name="results.csv",
    db_test_result=None,
)

for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ── Helper functions ──────────────────────────────────────────────────────────

def _load_env_defaults() -> dict:
    """Parse .env file and return default DB connection values."""
    path = PROJECT_ROOT / ".env"
    vals = dict(
        host="localhost",
        database="stackoverflow",
        user="cardinal-2026",
        password="Cardinal",
        port="5432",
    )
    if path.exists():
        for line in path.read_text().splitlines():
            line = line.strip()
            if "=" not in line or line.startswith("#"):
                continue
            k, _, v = line.partition("=")
            k, v = k.strip(), v.strip()
            MAP = {
                "POSTGRES_HOST": "host",
                "POSTGRES_DB": "database",
                "POSTGRES_USER": "user",
                "POSTGRES_PASSWORD": "password",
                "POSTGRES_PORT": "port",
            }
            if k in MAP:
                vals[MAP[k]] = v
    return vals


def _test_db(cfg: dict):
    """Attempt a psycopg2 connection; return (success, message)."""
    try:
        import psycopg2
        conn = psycopg2.connect(
            host=cfg["host"],
            database=cfg["database"],
            user=cfg["user"],
            password=cfg["password"],
            port=int(cfg["port"]),
            connect_timeout=5,
        )
        conn.close()
        return True, "Connection successful ✓"
    except Exception as e:
        return False, str(e)


def _save_csv(f) -> Path:
    """Write an uploaded file to temp_uploads/ and return its path."""
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    # Use a safe filename and write in one go to avoid holding the upload buffer long
    safe_name = os.path.basename(f.name) or "uploaded.csv"
    dest = TEMP_DIR / safe_name
    dest.write_bytes(f.getvalue())
    return dest


def _build_cmd(cfg: dict, csv_path: str) -> list:
    """Construct the subprocess argv list for model_evaluator.py."""
    cmd = [
        sys.executable, "-u", "model_evaluator.py",
        csv_path,
        "--output", cfg["output"],
        "--query-column", cfg["query_column"],
        "--model", cfg["model"],
        "--iterations", str(cfg["iterations"]),
        "--workers", str(cfg["workers"]),
    ]
    if cfg["use_hints"]:
        cmd.append("--use-hints")
    if cfg["limit"]:
        cmd += ["--limit", str(cfg["limit"])]
    if cfg["verbose"]:
        cmd.append("--verbose")
    return cmd


def _reader_thread(proc, q: queue.Queue):
    """Background thread: read merged stdout/stderr and enqueue lines."""
    for raw in iter(proc.stdout.readline, b""):
        line = raw.decode("utf-8", errors="replace").replace("\r", "\n").strip()
        if line:
            q.put(("line", line))
    proc.wait()
    q.put(("done", proc.returncode))


def _start(cmd: list, db_cfg: dict, hf_token: str = ""):
    """Launch the pipeline subprocess and wire up the reader thread."""
    env = os.environ.copy()
    # Use project-local Hugging Face cache and disable Xet (avoids 416 / permission errors)
    hf_base = PROJECT_ROOT / ".cache" / "huggingface"
    hf_cache = hf_base / "hub"
    hf_cache.mkdir(parents=True, exist_ok=True)
    env.update({
        "POSTGRES_HOST": db_cfg["host"],
        "POSTGRES_DB": db_cfg["database"],
        "POSTGRES_USER": db_cfg["user"],
        "POSTGRES_PASSWORD": db_cfg["password"],
        "POSTGRES_PORT": str(db_cfg["port"]),
        "PYTHONPATH": str(PROJECT_ROOT / "scripts") + os.pathsep + env.get("PYTHONPATH", ""),
        "HUGGINGFACE_HUB_CACHE": str(hf_cache),
        "HF_HOME": str(hf_base),
        "HF_HUB_DISABLE_XET": "1",
        "HF_HUB_ENABLE_HF_TRANSFER": "0",
    })
    if hf_token:
        env["HF_TOKEN"] = hf_token
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,   # merge stderr into stdout
        start_new_session=True,     # own process group so killpg works
        cwd=str(PROJECT_ROOT),
        env=env,
    )
    q: queue.Queue = queue.Queue()
    threading.Thread(target=_reader_thread, args=(proc, q), daemon=True).start()
    ss = st.session_state
    ss.process = proc
    ss.log_queue = q
    ss.log_lines = []
    ss.gen_current = 0
    ss.gen_total = 0
    ss.progress_current = 0
    ss.progress_total = 0
    ss.is_running = True
    ss.run_completed = False
    ss.run_failed = False
    ss.results_df = None


def _drain():
    """Drain the log queue into session state; update progress + done flag."""
    q = st.session_state.log_queue
    if not q:
        return
    while True:
        try:
            typ, val = q.get_nowait()
        except queue.Empty:
            break
        if typ == "done":
            st.session_state.is_running = False
            st.session_state.run_completed = (val == 0)
            st.session_state.run_failed = (val != 0)
            break
        m = GEN_PROGRESS_RE.search(val)
        if m:
            st.session_state.gen_current = int(m.group(1))
            st.session_state.gen_total = int(m.group(2))
        m = PROGRESS_RE.search(val)
        if m:
            st.session_state.progress_current = int(m.group(1))
            st.session_state.progress_total = int(m.group(2))
        st.session_state.log_lines.append(val)


def _stop():
    """Terminate the pipeline subprocess and all its children."""
    proc = st.session_state.process
    if proc and proc.poll() is None:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except Exception:
            proc.terminate()
    st.session_state.is_running = False
    st.session_state.run_failed = True


# ── Load DB defaults once per session ─────────────────────────────────────────
_DB = _load_env_defaults()


# ═════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.title("⚡ SQL Storm Evaluator")
    st.caption("LLM-powered query plan generation & benchmarking")
    st.divider()

    # ── Input CSV ─────────────────────────────────────────────────────────────
    st.subheader("📂 Input CSV")
    uploaded = st.file_uploader(
        "Upload query CSV", type=["csv"], label_visibility="collapsed"
    )

    if uploaded:
        fid = f"{uploaded.name}_{uploaded.size}"
        if fid != st.session_state.uploaded_file_id:
            try:
                p = _save_csv(uploaded)
                preview_df = pd.read_csv(p, nrows=1)
                st.session_state.uploaded_csv_path = str(p)
                st.session_state.uploaded_csv_name = uploaded.name
                st.session_state.uploaded_file_id = fid
                st.session_state.csv_columns = list(preview_df.columns)
            except Exception as e:
                st.error(f"Failed to process CSV: {e}")
                st.session_state.uploaded_file_id = None
                st.session_state.uploaded_csv_path = None
                st.session_state.uploaded_csv_name = None
                st.session_state.csv_columns = []

    if st.session_state.uploaded_csv_name:
        st.caption(f"✅ **{st.session_state.uploaded_csv_name}**")
        cols_avail = st.session_state.csv_columns
        if cols_avail:
            default_idx = (
                cols_avail.index("sql_text") if "sql_text" in cols_avail else 0
            )
            query_col = st.selectbox(
                "SQL query column", cols_avail, index=default_idx
            )
            with st.expander("Preview (first 5 rows)"):
                try:
                    st.dataframe(
                        pd.read_csv(st.session_state.uploaded_csv_path, nrows=5),
                        use_container_width=True,
                    )
                except Exception as e:
                    st.warning(f"Could not load preview: {e}")
        else:
            query_col = st.text_input("SQL query column name", value="sql_text")
    else:
        st.caption("No file uploaded yet")
        query_col = "sql_text"

    st.divider()

    # ── Model Configuration ────────────────────────────────────────────────────
    st.subheader("🤖 Model")
    preset = st.selectbox(
        "Preset", ["(enter custom ID)"] + MODEL_PRESETS, index=0
    )
    if preset == "(enter custom ID)":
        model_id = st.text_input(
            "HuggingFace model ID",
            placeholder="org/model-name",
            value="meta-llama/Llama-3.2-3B",
        )
    else:
        model_id = preset
        st.caption(f"`{model_id}`")

    hf_token = st.text_input(
        "HuggingFace token (optional)",
        type="password",
        help="Required for gated models (e.g. Llama). Leave blank if already logged in via `huggingface-cli login`.",
    )

    use_hints = st.checkbox(
        "Use pg_hint_plan hints",
        value=True,
        help="Convert generated plans to pg_hint_plan optimization hints before execution",
    )
    verbose = st.checkbox("Verbose output", value=False)

    c1, c2 = st.columns(2)
    with c1:
        iterations = st.number_input(
            "Iterations", min_value=1, max_value=20, value=1,
            help="Benchmark passes per query (results are averaged)",
        )
    with c2:
        workers = st.number_input(
            "Workers", min_value=1, max_value=16, value=4,
            help="Parallel DB execution workers",
        )

    row_limit_val = st.number_input(
        "Row limit (0 = all)", min_value=0, step=10, value=0,
        help="Limit to N queries — useful for quick smoke tests",
    )
    row_limit = int(row_limit_val) if row_limit_val > 0 else None

    output_file = st.text_input("Output CSV filename", value="results.csv")

    st.divider()

    # ── Database Connection ───────────────────────────────────────────────────
    st.subheader("🗄️ Database")
    with st.expander("Connection settings", expanded=False):
        db_host = st.text_input("Host", value=_DB["host"])
        db_name = st.text_input("Database", value=_DB["database"])
        db_user = st.text_input("User", value=_DB["user"])
        db_pass = st.text_input("Password", value=_DB["password"], type="password")
        db_port = st.text_input("Port", value=_DB["port"])

        if st.button("Test Connection", use_container_width=True):
            ok, msg = _test_db(
                dict(host=db_host, database=db_name,
                     user=db_user, password=db_pass, port=db_port)
            )
            st.session_state.db_test_result = (ok, msg)

        if st.session_state.db_test_result:
            ok, msg = st.session_state.db_test_result
            (st.success if ok else st.error)(msg)

    db_cfg = dict(
        host=db_host, database=db_name,
        user=db_user, password=db_pass, port=db_port,
    )

    st.divider()

    # ── Run / Stop ────────────────────────────────────────────────────────────
    can_run = (
        bool(st.session_state.uploaded_csv_path)
        and bool(model_id.strip())
        and not st.session_state.is_running
    )

    if st.session_state.is_running:
        if st.button("⏹ Stop Pipeline", use_container_width=True, type="secondary"):
            _stop()
            st.rerun()
    else:
        if st.button(
            "▶ Run Evaluation Pipeline",
            use_container_width=True,
            type="primary",
            disabled=not can_run,
        ):
            cfg = dict(
                output=output_file,
                query_column=query_col,
                model=model_id.strip(),
                use_hints=use_hints,
                iterations=iterations,
                workers=workers,
                limit=row_limit,
                verbose=verbose,
            )
            st.session_state.current_output_name = output_file
            _start(_build_cmd(cfg, st.session_state.uploaded_csv_path), db_cfg, hf_token)
            st.rerun()

    if not can_run and not st.session_state.is_running:
        missing = []
        if not st.session_state.uploaded_csv_path:
            missing.append("upload a CSV")
        if not model_id.strip():
            missing.append("enter a model ID")
        if missing:
            st.caption(f"⚠ To enable: {' and '.join(missing)}")


# ═════════════════════════════════════════════════════════════════════════════
# MAIN AREA — three tabs
# ═════════════════════════════════════════════════════════════════════════════

tab_log, tab_results, tab_charts = st.tabs(
    ["📋 Live Log", "📊 Results", "📈 Visualizations"]
)

# ── Tab 1: Live Log ───────────────────────────────────────────────────────────
with tab_log:
    ss = st.session_state

    if ss.is_running:
        st.markdown("**Status:** 🟢 Running…")

        # Phase 1: Plan generation
        gen_pct = (ss.gen_current / ss.gen_total) if ss.gen_total else 0.0
        if ss.gen_total > 0:
            gen_label = f"Phase 1 — Generating plans: {ss.gen_current} / {ss.gen_total}"
        else:
            # Show indeterminate spinner text until first "Generated X/Y" line arrives
            gen_label = "Phase 1 — Generating plans (loading model…)"
        st.progress(gen_pct, text=gen_label)

        # Phase 2: Query execution (only shown once execution starts)
        if ss.progress_total > 0:
            exec_pct = ss.progress_current / ss.progress_total
            exec_label = f"Phase 2 — Executing queries: {ss.progress_current} / {ss.progress_total}"
            st.progress(exec_pct, text=exec_label)
    elif ss.run_completed:
        st.success(
            "✅ Pipeline completed successfully! "
            "Switch to the **Results** or **Visualizations** tab."
        )
    elif ss.run_failed:
        st.error("❌ Pipeline failed or was stopped. See the log below for details.")
    else:
        st.info(
            "Configure your settings in the sidebar, then click **▶ Run Evaluation Pipeline**."
        )

    if ss.log_lines:
        # Show the last 300 lines to keep the UI responsive
        log_text = "\n".join(ss.log_lines[-300:])
        st.code(log_text, language=None)
    elif ss.is_running:
        st.caption("Waiting for output…")

# ── Tab 2: Results ────────────────────────────────────────────────────────────
with tab_results:
    ss = st.session_state

    # Load the results CSV once the run finishes
    if ss.run_completed and ss.results_df is None:
        rp = PROJECT_ROOT / ss.current_output_name
        if rp.exists():
            ss.results_df = pd.read_csv(rp)
        else:
            st.warning(
                f"Expected output file `{ss.current_output_name}` not found. "
                "Check the log for errors."
            )

    df = ss.results_df
    if df is not None:
        metric_cols = [
            c for c in ("execution_time_ms", "cpu_time_ms", "peak_memory_kb")
            if c in df.columns
        ]

        if metric_cols:
            mdf = df[metric_cols].dropna()
            LABELS = {
                "execution_time_ms": ("Avg Execution Time", "ms"),
                "cpu_time_ms": ("Avg CPU Time", "ms"),
                "peak_memory_kb": ("Avg Peak Memory", "KB"),
            }
            metric_widget_cols = st.columns(len(metric_cols))
            for widget_col, col_name in zip(metric_widget_cols, metric_cols):
                lbl, unit = LABELS[col_name]
                avg = mdf[col_name].mean()
                mn = mdf[col_name].min()
                mx = mdf[col_name].max()
                widget_col.metric(
                    lbl,
                    f"{avg:.2f} {unit}",
                    delta=f"min {mn:.2f} / max {mx:.2f}",
                    delta_color="off",
                )
            st.caption(
                f"**{len(df)}** queries total · "
                f"**{mdf.shape[0]}** succeeded · "
                f"**{len(df) - mdf.shape[0]}** failed / no metric"
            )
            st.divider()

        st.dataframe(df, use_container_width=True, height=420)

        st.download_button(
            label="⬇ Download Results CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name=ss.current_output_name,
            mime="text/csv",
            type="primary",
        )
    else:
        st.info("Results will appear here after a successful pipeline run.")

# ── Tab 3: Visualizations ──────────────────────────────────────────────────────
with tab_charts:
    df = st.session_state.results_df

    if df is not None:
        try:
            import plotly.express as px
        except ImportError:
            st.error("Plotly is required for charts. Install it: `pip install plotly`")
            df = None  # skip the chart block below

    if df is not None:
        mdf = df[["execution_time_ms", "cpu_time_ms", "peak_memory_kb"]].dropna()
        if mdf.empty:
            st.warning(
                "No numeric metric columns found in results "
                "(all queries may have failed or the columns are missing)."
            )
        else:
            dfi = df.reset_index().rename(columns={"index": "Query #"})

            # Row 1 ────────────────────────────────────────────────────────────
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Execution Time per Query")
                fig = px.bar(
                    dfi.dropna(subset=["execution_time_ms"]),
                    x="Query #",
                    y="execution_time_ms",
                    color="execution_time_ms",
                    color_continuous_scale="Viridis",
                    labels={"execution_time_ms": "Time (ms)"},
                )
                fig.update_layout(coloraxis_showscale=False)
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.subheader("Execution Time Distribution")
                fig = px.histogram(
                    mdf,
                    x="execution_time_ms",
                    nbins=30,
                    labels={"execution_time_ms": "Execution Time (ms)"},
                    color_discrete_sequence=["#636EFA"],
                )
                st.plotly_chart(fig, use_container_width=True)

            # Row 2 ────────────────────────────────────────────────────────────
            col3, col4 = st.columns(2)

            with col3:
                st.subheader("Execution Time vs CPU Time")
                fig = px.scatter(
                    mdf,
                    x="cpu_time_ms",
                    y="execution_time_ms",
                    color="peak_memory_kb",
                    color_continuous_scale="Plasma",
                    opacity=0.8,
                    labels={
                        "cpu_time_ms": "CPU Time (ms)",
                        "execution_time_ms": "Execution Time (ms)",
                        "peak_memory_kb": "Peak Mem (KB)",
                    },
                )
                st.plotly_chart(fig, use_container_width=True)

            with col4:
                st.subheader("Peak Memory per Query")
                fig = px.bar(
                    dfi.dropna(subset=["peak_memory_kb"]),
                    x="Query #",
                    y="peak_memory_kb",
                    color="peak_memory_kb",
                    color_continuous_scale="Reds",
                    labels={"peak_memory_kb": "Memory (KB)"},
                )
                fig.update_layout(coloraxis_showscale=False)
                st.plotly_chart(fig, use_container_width=True)

            # Full-width box plot ──────────────────────────────────────────────
            st.subheader("Metrics Distribution")
            melted = mdf.melt(var_name="Metric", value_name="Value")
            fig = px.box(
                melted,
                x="Metric",
                y="Value",
                color="Metric",
                labels={"Value": "Value (ms or KB)"},
                color_discrete_map={
                    "execution_time_ms": "#636EFA",
                    "cpu_time_ms": "#EF553B",
                    "peak_memory_kb": "#00CC96",
                },
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    elif st.session_state.results_df is None:
        st.info("Visualizations will appear here after a successful pipeline run.")


# ═════════════════════════════════════════════════════════════════════════════
# POLLING LOOP — keep the log streaming while the pipeline runs
# ═════════════════════════════════════════════════════════════════════════════

if st.session_state.is_running:
    _drain()
    time.sleep(0.5)
    st.rerun()

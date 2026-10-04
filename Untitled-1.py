"""Aplikacja Streamlit: analiza biomechaniki ruchu (MVP)."""

from __future__ import annotations
import io
from typing import Callable, Protocol, Sequence
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from plotly.subplots import make_subplots

# --------------------------------------------------------------------------- #
# Konfiguracja
# --------------------------------------------------------------------------- #
PORTFOLIO_URL: str = "przykładowy-link"
PARALLEL_SQUAT_ANGLE: float = 90.0
EPS: float = 1e-9

SQUAT_COLUMNS: list[str] = [
    "Czas", "Biodro_X", "Biodro_Y", "Kolano_X", "Kolano_Y", "Kostka_X", "Kostka_Y",
]
RUNNING_COLUMNS: list[str] = ["Dystans", "Tetno", "GCT"]

# Lista wszystkich numerycznych kolumn, które chcemy wyciągnąć z Garmina
GARMIN_NUMERIC_COLS: list[str] = [
    "Dystans", "Średnie tętno", "Średni czas kontaktu z podłożem", 
    "Średnie odchylenie pionowe", "Średnia kadencja biegu", 
    "Średnia długość kroku", "Suma kalorii", "Maksymalne tętno"
]

class DataLoadError(Exception):
    pass

# --------------------------------------------------------------------------- #
# Warstwa danych
# --------------------------------------------------------------------------- #
class LandmarkSource(Protocol):
    def load(self) -> pd.DataFrame:
        ...

def _decode_bytes(raw: bytes) -> str:
    for encoding in ("utf-8-sig", "cp1250"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1")

def _to_numeric(series: pd.Series) -> pd.Series:
    if series.dtype == object:
        series = series.astype(str).str.strip().str.replace(",", ".", regex=False)
    return pd.to_numeric(series, errors="coerce")

def load_and_validate_csv(raw: bytes, required: Sequence[str]) -> pd.DataFrame:
    if not raw or not raw.strip():
        raise DataLoadError("Plik jest pusty.")
    try:
        df = pd.read_csv(io.StringIO(_decode_bytes(raw)), sep=None, engine="python")
    except Exception as exc: 
        raise DataLoadError(f"Nie udało się odczytać pliku CSV ({exc}).") from exc

    df.columns = [str(c).strip() for c in df.columns]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise DataLoadError(f"Brak wymaganych kolumn: {', '.join(missing)}.")

    df = df[list(required)].apply(_to_numeric)
    df = df.dropna().reset_index(drop=True)
    if len(df) < 2:
        raise DataLoadError("Za mało poprawnych wierszy do analizy (min. 2).")
    return df

def load_garmin_history(raw: bytes) -> pd.DataFrame:
    if not raw or not raw.strip():
        raise DataLoadError("Plik jest pusty.")
    try:
        df = pd.read_csv(io.StringIO(_decode_bytes(raw)), sep=None, engine="python")
    except Exception as exc:
        raise DataLoadError(f"Nie udało się odczytać pliku CSV ({exc}).") from exc

    df.columns = [str(c).strip() for c in df.columns]
    
    if "Data" not in df.columns:
        raise DataLoadError("Brak kolumny 'Data'. Wgraj plik Activities.csv z Garmina.")

    df['Data'] = pd.to_datetime(df['Data'], errors='coerce')
    
    available_cols = [c for c in GARMIN_NUMERIC_COLS if c in df.columns]
    df[available_cols] = df[available_cols].apply(_to_numeric)
    
    return df

class CsvLandmarkSource:
    def __init__(self, raw: bytes) -> None:
        self._raw = raw
    def load(self) -> pd.DataFrame:
        return load_and_validate_csv(self._raw, SQUAT_COLUMNS)

# --------------------------------------------------------------------------- #
# Dane przykładowe (Mock Data - Męskie / Kobiece)
# --------------------------------------------------------------------------- #
def sample_squat_data(gender: str) -> pd.DataFrame:
    rng = np.random.default_rng(42 if gender == "Mężczyzna" else 123)
    t = np.linspace(0, 6, 180)
    # Kobiety częściej mają lepszą mobilność bioder - schodzą niżej w przysiadzie
    min_angle = 80 if gender == "Mężczyzna" else 70 
    amplitude = (170 - min_angle) / 2
    offset = 170 - amplitude
    theta = np.radians(offset + amplitude * np.cos(2 * np.pi * t / 3))
    
    scale, knee_x, knee_y = 500.0, 400.0, 500.0
    shank_len = thigh_len = 0.45 * scale
    noise = lambda: rng.normal(0, 0.8, t.size)
    
    return pd.DataFrame({
        "Czas": t,
        "Biodro_X": knee_x + thigh_len * np.sin(theta) + noise(),
        "Biodro_Y": knee_y + thigh_len * np.cos(theta) + noise(),
        "Kolano_X": np.full_like(t, knee_x),
        "Kolano_Y": np.full_like(t, knee_y),
        "Kostka_X": np.full_like(t, knee_x) + noise(),
        "Kostka_Y": knee_y + shank_len + noise(),
    })

def sample_running_data(gender: str) -> pd.DataFrame:
    rng = np.random.default_rng(7 if gender == "Mężczyzna" else 77)
    dist = np.linspace(0, 10, 500)
    # Kobiety mają fizjologicznie nieco wyższe tętno przy tym samym wysiłku
    base_hr = 140 if gender == "Mężczyzna" else 150
    base_gct = 215 if gender == "Mężczyzna" else 205
    
    return pd.DataFrame({
        "Dystans": dist,
        "Tetno": base_hr + 3.5 * dist + rng.normal(0, 1.5, dist.size),
        "GCT": base_gct + 3.0 * dist + rng.normal(0, 3.0, dist.size),
    })

def sample_garmin_history_data(gender: str) -> pd.DataFrame:
    rng = np.random.default_rng(10 if gender == "Mężczyzna" else 99)
    dates = pd.date_range(start="2026-05-01", periods=30, freq='4D')
    dystans = np.random.uniform(3, 15, size=30)
    
    base_hr = 130 if gender == "Mężczyzna" else 142
    tetno = base_hr + dystans * 2 + rng.normal(0, 5, 30)
    
    base_gct = 220 if gender == "Mężczyzna" else 210
    gct = base_gct + tetno * 0.3 + rng.normal(0, 4, 30)
    
    cadence = 160 + rng.normal(0, 4, 30) if gender == "Mężczyzna" else 168 + rng.normal(0, 4, 30)
    vert_osc = rng.normal(9.0, 0.5, 30) if gender == "Mężczyzna" else rng.normal(8.2, 0.5, 30)
    
    return pd.DataFrame({
        "Data": dates,
        "Dystans": dystans,
        "Średnie tętno": tetno,
        "Średni czas kontaktu z podłożem": gct,
        "Średnia kadencja biegu": cadence,
        "Średnie odchylenie pionowe": vert_osc
    })

# --------------------------------------------------------------------------- #
# Logika analityczna
# --------------------------------------------------------------------------- #
def calculate_knee_angle(df: pd.DataFrame) -> np.ndarray:
    hip = df[["Biodro_X", "Biodro_Y"]].to_numpy(dtype=float)
    knee = df[["Kolano_X", "Kolano_Y"]].to_numpy(dtype=float)
    ankle = df[["Kostka_X", "Kostka_Y"]].to_numpy(dtype=float)

    thigh = hip - knee
    shank = ankle - knee

    dot = np.einsum("ij,ij->i", thigh, shank)
    norms = np.linalg.norm(thigh, axis=1) * np.linalg.norm(shank, axis=1)

    with np.errstate(divide="ignore", invalid="ignore"):
        cos_theta = np.where(norms > EPS, dot / norms, np.nan)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return np.degrees(np.arccos(cos_theta))

def window_mean(series: pd.Series, window: int) -> pd.Series:
    if window <= 1:
        return series
    return series.rolling(window=window, center=True, min_periods=1).mean()

# --------------------------------------------------------------------------- #
# Wykresy
# --------------------------------------------------------------------------- #
def build_knee_angle_figure(time: pd.Series, angle: np.ndarray) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time, y=angle, mode="lines", name="Kąt kolana", line=dict(color="#2563eb", width=3)))
    fig.add_hline(y=PARALLEL_SQUAT_ANGLE, line_dash="dash", line_color="#f97316")
    fig.update_layout(title="Kąt w stawie kolanowym w czasie", template="plotly_white")
    return fig

def build_running_figure(df: pd.DataFrame, hr: pd.Series, gct: pd.Series) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=df["Dystans"], y=hr, mode="lines", name="Tętno [bpm]", line=dict(color="#dc2626")), secondary_y=False)
    fig.add_trace(go.Scatter(x=df["Dystans"], y=gct, mode="lines", name="GCT [ms]", line=dict(color="#2563eb")), secondary_y=True)
    fig.update_layout(title="Tętno i GCT względem dystansu", template="plotly_white")
    return fig

def build_dynamic_garmin_figure(df: pd.DataFrame, x_col: str, y_col: str) -> go.Figure:
    df_clean = df.dropna(subset=[x_col, y_col]).copy()
    df_clean = df_clean.sort_values(by="Data")
    df_clean["Data_str"] = df_clean["Data"].dt.strftime('%Y-%m-%d')
    
    fig = px.scatter(
        df_clean, x=x_col, y=y_col, 
        color="Data", hover_name="Data_str",
        color_continuous_scale=px.colors.sequential.Viridis,
        title=f"Zależność: {y_col} względem {x_col}"
    )
    fig.update_traces(marker=dict(size=10, opacity=0.8))
    fig.update_layout(template="plotly_white", margin=dict(l=20, r=20, t=60, b=20))
    return fig

# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #
def render_home() -> None:
    st.title("🏃 Analiza biomechaniki ruchu")
    st.markdown("Wybierz moduł z paska bocznego.")
    st.link_button("🔗 Przejdź do mojego portfolio", PORTFOLIO_URL)

def render_squat_module() -> None:
    st.title("🏋️ Kinematyka siłowa – przysiad")
    
    data_source = st.radio("Wybierz źródło danych:", ["Wgraj własny plik CSV", "Dane testowe (Mężczyzna)", "Dane testowe (Kobieta)"], horizontal=True)
    
    df = None
    if data_source == "Wgraj własny plik CSV":
        uploaded = st.file_uploader("Wgraj plik CSV z Kinovea", type=["csv"], key="squat")
        if uploaded:
            try:
                df = CsvLandmarkSource(uploaded.getvalue()).load()
            except Exception as exc:
                st.error(f"❌ Błąd: {exc}")
    else:
        gender = "Mężczyzna" if "Mężczyzna" in data_source else "Kobieta"
        df = sample_squat_data(gender)

    if df is not None:
        try:
            angle = calculate_knee_angle(df)
            st.plotly_chart(build_knee_angle_figure(df["Czas"], angle), width="stretch")
            
            min_angle = np.nanmin(angle)
            if min_angle <= PARALLEL_SQUAT_ANGLE:
                st.success(f"Zaliczono głębokość. Kąt minimalny: {min_angle:.1f}°")
            else:
                st.warning(f"Brak pełnego zejścia. Kąt minimalny: {min_angle:.1f}°")
        except Exception as exc:
            st.error(f"❌ Błąd analityczny: {exc}")

def render_running_module() -> None:
    st.title("👟 Biomechanika biegu (Pojedynczy trening)")
    
    data_source = st.radio("Wybierz źródło danych:", ["Wgraj własny plik CSV", "Dane testowe (Mężczyzna)", "Dane testowe (Kobieta)"], horizontal=True)
    
    df = None
    if data_source == "Wgraj własny plik CSV":
        uploaded = st.file_uploader("Wgraj CSV z pojedynczego biegu", type=["csv"], key="run")
        if uploaded:
            try:
                df = load_and_validate_csv(uploaded.getvalue(), RUNNING_COLUMNS)
            except Exception as exc:
                st.error(f"❌ Błąd: {exc}")
    else:
        gender = "Mężczyzna" if "Mężczyzna" in data_source else "Kobieta"
        df = sample_running_data(gender)

    if df is not None:
        try:
            df = df.sort_values("Dystans").reset_index(drop=True)
            st.plotly_chart(build_running_figure(df, df["Tetno"], df["GCT"]), width="stretch")
        except Exception as exc:
            st.error(f"❌ Błąd analityczny: {exc}")

def render_garmin_history_module() -> None:
    st.title("📈 Eksplorator Danych Garmin (Historia)")
    st.markdown("Analizuj dowolne parametry ze swojej historii treningów.")

    data_source = st.radio("Wybierz źródło danych:", ["Wgraj własny plik Activities.csv", "Dane testowe (Mężczyzna)", "Dane testowe (Kobieta)"], horizontal=True)

    df = None
    if data_source == "Wgraj własny plik Activities.csv":
        uploaded = st.file_uploader("Wgraj plik Activities.csv", type=["csv"], key="garmin")
        if uploaded:
            try:
                df = load_garmin_history(uploaded.getvalue())
            except Exception as exc: 
                st.error(f"❌ Błąd: {exc}")
    else:
        gender = "Mężczyzna" if "Mężczyzna" in data_source else "Kobieta"
        df = sample_garmin_history_data(gender)

    if df is not None:
        available_cols = [c for c in GARMIN_NUMERIC_COLS if c in df.columns]
        
        col1, col2 = st.columns(2)
        with col1:
            x_axis = st.selectbox("Wybierz daną dla osi X", available_cols, index=0)
        with col2:
            # Domyślnie na osi Y ustawi GCT, jeśli jest na liście
            default_y = available_cols.index("Średni czas kontaktu z podłożem") if "Średni czas kontaktu z podłożem" in available_cols else 0
            y_axis = st.selectbox("Wybierz daną dla osi Y", available_cols, index=default_y)
            
        st.plotly_chart(build_dynamic_garmin_figure(df, x_axis, y_axis), width="stretch")
        
        with st.expander("Podgląd przefiltrowanych danych"):
            st.dataframe(df, width="stretch")

PAGES: dict[str, Callable[[], None]] = {
    "🏠 Strona główna": render_home,
    "🏋️ Kinematyka siłowa": render_squat_module,
    "👟 Biomechanika biegu": render_running_module,
    "📈 Historia Garmin": render_garmin_history_module,
}

def main() -> None:
    st.set_page_config(page_title="Analiza biomechaniki", page_icon="🏃", layout="wide")
    choice = st.sidebar.radio("Nawigacja", list(PAGES.keys()))
    PAGES[choice]()

if __name__ == "__main__":
    main()
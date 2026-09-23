import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Flight Delay Classifier")


@st.cache_resource
def load():
    return joblib.load("model.joblib")


bundle = load()
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@500;600;700&family=IBM+Plex+Mono:wght@500;600&display=swap');
.stApp {
  background:
    radial-gradient(900px 420px at 15% -10%, rgba(255,176,0,.10), transparent 60%),
    repeating-linear-gradient(0deg, rgba(255,255,255,.025) 0 1px, transparent 1px 44px),
    #0D1014;
}
.block-container { max-width: 1080px; padding-top: 3.5rem; }
header[data-testid="stHeader"] { background: transparent; }
html, body, .stApp, label, p, div { font-family: 'Barlow Condensed', sans-serif; }
.eyebrow, .title, .sub { animation: rise 600ms cubic-bezier(0.23,1,0.32,1) both; }
.eyebrow { color:#FFB000; letter-spacing:.32em; font-size:.95rem; font-weight:600; text-transform:uppercase; }
.title { font-size:clamp(2.6rem,8vw,4.4rem); font-weight:700; line-height:.95; margin:.35rem 0 .6rem; animation-delay:80ms; }
.sub { color:#9A9584; font-size:1.2rem; margin-bottom:2rem; animation-delay:160ms; }
@keyframes rise { from { opacity:0; transform:translateY(10px); } to { opacity:1; transform:none; } }
[data-testid="stWidgetLabel"] p { text-transform:uppercase; letter-spacing:.16em; font-size:.85rem; color:#FFB000; font-weight:600; }
[data-baseweb="select"] > div, [data-testid="stNumberInputContainer"] { background:#161A21; border-radius:6px; border:1px solid #262C36; }
[data-baseweb="select"] div, input { font-family:'IBM Plex Mono', monospace !important; }
.stButton { width:100%; }
.stButton button {
  width:100% !important; margin-top:.6rem; background:#FFB000 !important; color:#0D1014 !important; border:0 !important; border-radius:6px;
  padding:.7rem 0; transition: transform 160ms cubic-bezier(0.23,1,0.32,1), filter 160ms ease-out;
}
.stButton button p { font:700 1.25rem 'Barlow Condensed'; letter-spacing:.22em; text-transform:uppercase; color:#0D1014 !important; }
.stButton button:hover { filter:brightness(1.08); }
.stButton button:active { transform:scale(0.97); }
.board { animation: slidein 420ms cubic-bezier(0.23,1,0.32,1) both; margin-top:1.7rem; padding:1.6rem 1.4rem 1.2rem; background:#10131A; border:1px solid #262C36; border-radius:12px; }
.board .lbl { color:#9A9584; text-transform:uppercase; letter-spacing:.22em; font-size:.85rem; margin-bottom:.9rem; }
.flap {
  display:inline-flex; align-items:center; justify-content:center; position:relative; margin-right:8px;
  width:.78em; height:1.25em; border-radius:8px; color:#FFB000;
  font:600 clamp(52px,7vw,92px)/1 'IBM Plex Mono', monospace;
  background:linear-gradient(#1D222B 50%, #151920 50%);
  transform-origin:50% 0; animation: flip 520ms cubic-bezier(0.23,1,0.32,1) both;
}
.flap::after { content:''; position:absolute; left:0; right:0; top:50%; height:2px; background:#0A0C10; }
@keyframes flip { from { transform:perspective(700px) rotateX(-90deg); opacity:0; } to { transform:perspective(700px) rotateX(0); opacity:1; } }
.status { margin-top:1rem; font-size:1.5rem; font-weight:700; letter-spacing:.14em; text-transform:uppercase; }
.late { color:#FF6B4A; } .ok { color:#8FD694; }
@keyframes slidein { from { opacity:0; transform:translateX(18px); } to { opacity:1; transform:none; } }
.flap.idle { animation:none; color:#3A404B; }
.board.idle { animation:none; }
[data-testid="stColumn"]:nth-child(2) { position:sticky; top:2rem; align-self:flex-start; }
@media (prefers-reduced-motion: reduce) { .board, .flap, .eyebrow, .title, .sub { animation:none; } }
</style>
<div class="eyebrow">Departures · Delay forecast</div>
<div class="title">Will your flight<br>be late?</div>
<div class="sub">Pick a flight. The model estimates its chance of a delay.</div>
""", unsafe_allow_html=True)

left, right = st.columns([3, 2], gap="large")
with left:
    airline = st.selectbox("Airline", bundle["airlines"])
    col1, col2 = st.columns(2)
    origin = col1.selectbox("From", bundle["airports"], index=bundle["airports"].index("ATL"))
    dest = col2.selectbox("To", bundle["airports"], index=bundle["airports"].index("LAX"))
    col3, col4, col5 = st.columns(3)
    day = col3.selectbox("Day of week (1-7)", range(1, 8))
    hour = col4.selectbox("Departure hour", range(24), index=12)
    length = col5.number_input("Length (minutes)", 0, 700, 150)
    go = st.button("Predict", width="stretch")

with right:
    if go and origin == dest:
        st.error("Origin and destination must differ.")
    elif go:
        row = pd.DataFrame([{"Airline": airline, "AirportFrom": origin, "AirportTo": dest,
                             "DayOfWeek": day, "Hour": hour, "Length": length}])
        p = bundle["model"].predict_proba(row)[0, 1]
        tiles = "".join(f'<span class="flap" style="animation-delay:{n * 90}ms">{c}</span>'
                        for n, c in enumerate(f"{p:.0%}"))
        status = ('<div class="status late">Likely delayed</div>' if p >= 0.5
                  else '<div class="status ok">Likely on time</div>')
        st.markdown(f'<div class="board"><div class="lbl">Chance of delay</div>{tiles}{status}</div>',
                    unsafe_allow_html=True)
        st.caption("Baseline: 44.5% of flights in the data are delayed. The model can't see weather or "
                   "day-to-day events, so treat this as a rough estimate.")
    else:
        tiles = "".join(f'<span class="flap idle">{c}</span>' for c in "--%")
        st.markdown(f'<div class="board idle"><div class="lbl">Chance of delay</div>{tiles}'
                    '<div class="status" style="color:#5A5F6A">Awaiting flight</div></div>',
                    unsafe_allow_html=True)

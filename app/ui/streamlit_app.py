import os
import requests
import streamlit as st

st.set_page_config(page_title="Receiper", page_icon="🍳", layout="centered")

API_BASE = os.getenv("API_BASE", "http://localhost:8000")

st.title("Receiper 🍳 — Recipe Q&A with Sources")

with st.sidebar:
    st.header("Settings")
    hyde = st.checkbox("Enable HyDE", value=True)
    rerank = st.checkbox("Enable re-ranking", value=True)
    k = st.slider("Top-k", min_value=3, max_value=12, value=6)
    use_llm = st.checkbox("Use LLM answer", value=True)

query = st.text_input("Ask a cooking question or search recipes:", placeholder="vegan dinner under 30 minutes with chickpeas")

if st.button("Ask") and query:
    try:
        resp = requests.post(
            f"{API_BASE}/query",
            json={
                "query": query,
                "options": {"hyde": hyde, "rerank": rerank, "k": k, "llm": use_llm},
            },
        )
        data = resp.json()
        st.markdown("### Answer")
        st.write(data.get("answer", "No answer."))
        sources = data.get("sources", [])
        if sources:
            st.markdown("### Sources")
            for i, s in enumerate(sources, 1):
                st.markdown(f"[{i}] [{s.get('title','source')}]({s.get('url','#')}) — {s.get('snippet','')} ")
        with st.expander("Diagnostics"):
            st.json(data.get("diagnostics", {}))
    except Exception as e:
        st.error(f"API error: {e}")

st.divider()

if st.button("Check API health"):
    try:
        r = requests.get(f"{API_BASE}/healthz").json()
        st.success(r)
    except Exception as e:
        st.error(str(e))

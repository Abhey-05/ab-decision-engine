"""
Smoke test: load every page for every experiment with Streamlit's AppTest
and make sure nothing raises. Slow-ish (trains uplift models), ~30-60 s.
"""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from engine.data import EXPERIMENTS

APP = str(Path(__file__).resolve().parent.parent / "app.py")
PAGES = ["pages/home.py", "pages/trust.py", "pages/results.py", "pages/segments.py",
         "pages/uplift.py", "pages/analyst.py", "pages/plan.py", "pages/how_it_works.py"]


@pytest.mark.parametrize("key", list(EXPERIMENTS))
def test_every_page_loads(key, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)   # exercise the fallback path
    at = AppTest.from_file(APP, default_timeout=120)
    at.session_state["experiment"] = key
    at.run()
    assert not at.exception, at.exception
    for page in PAGES:
        at.switch_page(page)
        at.run()
        assert not at.exception, (page, at.exception)
        if page == "pages/analyst.py":
            at.button[0].click()
            at.run()
            assert not at.exception
            assert any("All numbers verified" in s.value for s in at.success), "template readout not verified"

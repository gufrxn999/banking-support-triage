"""Entry point for `streamlit run streamlit_app.py`.

Streamlit executes a script as `__main__` rather than as a package module, so a
file inside the package cannot use relative imports when run directly. This one
line wrapper keeps `banking_triage.demo` a normal module.
"""

from banking_triage.demo import main

main()

"""Launch the DeepGen assessment app: python -m streamlit run app/app.py."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.submission_app import main
main()

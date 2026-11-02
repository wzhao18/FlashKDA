set -e
pip install -e .
pip install "flash-linear-attention>=0.5.0" matplotlib pytest
python tests/test_fwd.py
python -m pytest tests/test_vsplit.py -x -q
python -m pytest tests/test_chunk_parallel.py -x -q

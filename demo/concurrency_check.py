"""Two threads sharing one SourceMap: the loser must fail before writing."""
import sys
import threading
from io import StringIO

sys.path.insert(0, "/home/node/.local/lib/python3.11/site-packages")
sys.path.insert(0, "/workspace")

from pdfminer.high_level import extract_text, extract_text_to_fp
from pdfminer.layout import LAParams
from pdfminer.pdfexceptions import SourceMapInUseError
from pdfminer.sourcemap import SourceMap

sm = SourceMap()
sm.begin_extraction()  # extraction #1 still "in progress"

errors = []
out = StringIO()
try:
    with open("/workspace/samples/simple1.pdf", "rb") as fp:
        extract_text_to_fp(fp, out, laparams=LAParams(), source_map=sm)
except SourceMapInUseError as e:
    errors.append(str(e))

assert len(errors) == 1, errors
assert out.getvalue() == "", f"stream must stay empty, got {out.getvalue()!r}"
assert sm.text is None  # never committed
sm.rollback_extraction()

# Independent maps in parallel must both complete.
maps = [SourceMap(), SourceMap()]
texts = [None, None]
barrier = threading.Barrier(2)

def worker(i):
    barrier.wait()
    texts[i] = extract_text(
        "/workspace/samples/simple%d.pdf" % (1 if i == 0 else 4),
        source_map=maps[i],
    )

ts = [threading.Thread(target=worker, args=(i,)) for i in range(2)]
for t in ts:
    t.start()
for t in ts:
    t.join()
assert maps[0].text == texts[0] and maps[1].text == texts[1]
assert maps[0].document_id != maps[1].document_id
print("busy-map rejection before any write: OK ->", errors[0])
print("parallel independent maps: OK, ids",
      maps[0].document_id, maps[1].document_id)

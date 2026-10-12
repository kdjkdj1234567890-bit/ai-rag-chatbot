#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI RAG 챗봇 — 기업 문서 Q&A (시연용 프로토타입)
- 외부 LLM API 없이 로컬에서 완전 동작
- TF-IDF + 코사인 유사도 기반 검색, 발췌형 답변 + 출처 표시
"""
import os
import re
import math
import sqlite3
import time
from datetime import datetime
from flask import Flask, request, jsonify, render_template, g

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "chatbot.db")
SAMPLE_DIR = os.path.join(BASE_DIR, "sample_docs")
ALLOWED_EXT = {".txt", ".md"}
CHUNK_SIZE = 600          # 청크 최대 글자 수
CHUNK_OVERLAP = 80        # 청크 겹침
TOP_K = 3                 # 참조할 상위 청크 수
MIN_SCORE = 0.06          # 이 미만이면 "찾을 수 없음"
MAX_UPLOAD_MB = 5

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024

# ---------------- DB ----------------
def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(DB_PATH)
        g.db.row_factory = sqlite3.Row
    return g.db

@app.teardown_appcontext
def close_db(exc):
    db = g.pop("db", None)
    if db is not None:
        db.close()

def init_db():
    db = sqlite3.connect(DB_PATH)
    db.execute("""CREATE TABLE IF NOT EXISTS docs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        filename TEXT NOT NULL,
        content TEXT NOT NULL,
        n_chunks INTEGER NOT NULL DEFAULT 0,
        uploaded_at TEXT NOT NULL)""")
    db.execute("""CREATE TABLE IF NOT EXISTS chunks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        doc_id INTEGER NOT NULL REFERENCES docs(id) ON DELETE CASCADE,
        idx INTEGER NOT NULL,
        text TEXT NOT NULL)""")
    db.commit()
    db.close()

# ---------------- 텍스트 처리 ----------------
_word_re = re.compile(r"[가-힣a-zA-Z0-9]+")

def tokenize(text):
    """단어 + 문자 바이그램(한국어 조사/어미 대응) 토큰화"""
    toks = []
    for w in _word_re.findall(text.lower()):
        toks.append(w)
        if len(w) >= 2:
            for i in range(len(w) - 1):
                toks.append(w[i:i+2])
    return toks

def chunk_text(text):
    """문단 기준 분할 후 CHUNK_SIZE 이하로 병합"""
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks, cur = [], ""
    for p in paras:
        if len(cur) + len(p) + 2 <= CHUNK_SIZE:
            cur = (cur + "\n\n" + p).strip()
        else:
            if cur:
                chunks.append(cur)
            if len(p) > CHUNK_SIZE:
                # 긴 문단은 문장 단위로 자름
                sents = re.split(r"(?<=[.!?。])\s+", p)
                buf = ""
                for s in sents:
                    if len(buf) + len(s) <= CHUNK_SIZE:
                        buf += (" " if buf else "") + s
                    else:
                        if buf:
                            chunks.append(buf)
                        buf = s
                cur = buf
            else:
                cur = p
    if cur:
        chunks.append(cur)
    return chunks

# ---------------- TF-IDF 인덱스 (인메모리) ----------------
_index = {"chunk_ids": [], "vocab": {}, "idf": [], "matrix": [], "texts": []}

def rebuild_index():
    global _index
    db = sqlite3.connect(DB_PATH)
    db.row_factory = sqlite3.Row
    rows = db.execute(
        "SELECT c.id, c.text FROM chunks c ORDER BY c.doc_id, c.idx").fetchall()
    db.close()
    chunk_ids = [r["id"] for r in rows]
    texts = [r["text"] for r in rows]
    # 문서 빈도
    df = {}
    tok_lists = []
    for t in texts:
        toks = tokenize(t)
        tok_lists.append(toks)
        for tok in set(toks):
            df[tok] = df.get(tok, 0) + 1
    vocab = {tok: i for i, tok in enumerate(df)}
    n = len(texts)
    idf = [math.log((n + 1) / (df_t + 1)) + 1.0 for df_t in
           (df[tok] for tok in vocab)]
    matrix = []
    for toks in tok_lists:
        tf = {}
        for tok in toks:
            tf[tok] = tf.get(tok, 0) + 1
        vec = [0.0] * len(vocab)
        for tok, c in tf.items():
            i = vocab[tok]
            vec[i] = (1.0 + math.log(c)) * idf[i]
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        matrix.append([v / norm for v in vec])
    _index = {"chunk_ids": chunk_ids, "vocab": vocab, "idf": idf,
              "matrix": matrix, "texts": texts}

def search(query, top_k=TOP_K):
    """코사인 유사도 상위 청크 반환: [(chunk_id, score)]"""
    if not _index["matrix"]:
        return []
    toks = tokenize(query)
    if not toks:
        return []
    vec = [0.0] * len(_index["vocab"])
    tf = {}
    for tok in toks:
        tf[tok] = tf.get(tok, 0) + 1
    for tok, c in tf.items():
        if tok in _index["vocab"]:
            i = _index["vocab"][tok]
            vec[i] = (1.0 + math.log(c)) * _index["idf"][i]
    norm = math.sqrt(sum(v * v for v in vec))
    if norm == 0:
        return []
    vec = [v / norm for v in vec]
    scored = []
    for cid, cvec in zip(_index["chunk_ids"], _index["matrix"]):
        s = sum(a * b for a, b in zip(vec, cvec))
        scored.append((cid, s))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]

# ---------------- 답변 생성 ----------------
_hl_re_cache = {}

def highlight(text, query):
    """질문 핵심어를 <mark>로 하이라이트 (HTML 이스케이프 후)"""
    import html
    esc = html.escape(text)
    keys = set()
    for w in _word_re.findall(query):
        # 단어의 모든 부분문자열(길이 2 이상) → 조사·어미 변형에도 매칭
        for i in range(len(w)):
            for j in range(i + 2, len(w) + 1):
                keys.add(w[i:j])
    # 긴 키워드 우선, 단일 패스로 치환 (중첩 mark 방지)
    keys = sorted((k for k in keys if len(k) >= 2), key=len, reverse=True)
    if keys:
        pat = re.compile("|".join(re.escape(html.escape(k)) for k in keys))
        esc = pat.sub(lambda m: f"<mark>{m.group(0)}</mark>", esc)
    return esc

def pick_sentences(chunk_text, query, n=2):
    sents = [s.strip() for s in re.split(r"(?<=[.!?。\n])\s*", chunk_text) if s.strip()]
    if not sents:
        return []
    qtoks = set(tokenize(query))
    scored = []
    for s in sents:
        st = set(tokenize(s))
        scored.append((len(qtoks & st), -len(s), s))
    scored.sort(reverse=True)
    return [s for _, _, s in scored[:n] if scored[0][0] > 0] or [s for _, _, s in scored[:n]]

def build_answer(question, hits):
    """발췌 기반 답변 + 출처 목록"""
    db = get_db()
    evidences = []
    for cid, score in hits:
        row = db.execute(
            """SELECT c.text, c.idx, d.filename,
                      (SELECT COUNT(*) FROM docs d2 WHERE d2.id <= d.id) AS doc_no
               FROM chunks c JOIN docs d ON d.id = c.doc_id
               WHERE c.id = ?""", (cid,)).fetchone()
        sents = pick_sentences(row["text"], question, n=2)
        excerpt = " ".join(sents)[:400]
        evidences.append({
            "doc_no": row["doc_no"],
            "chunk_no": row["idx"] + 1,
            "filename": row["filename"],
            "score": round(score, 3),
            "excerpt": excerpt,
            "excerpt_html": highlight(excerpt, question),
        })
    # 답변 본문: 최상위 근거 2개 발췌로 구성
    top = evidences[:2]
    lines = [f"문서에서 관련 내용을 찾았습니다. ({len(evidences)}개 근거)"]
    for e in top:
        lines.append(f"• {e['excerpt']}")
    return {
        "answer": "\n".join(lines),
        "found": True,
        "evidences": evidences,
    }

# ---------------- 라우트 ----------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/stats")
def api_stats():
    db = get_db()
    n_docs = db.execute("SELECT COUNT(*) c FROM docs").fetchone()["c"]
    n_chunks = db.execute("SELECT COUNT(*) c FROM chunks").fetchone()["c"]
    return jsonify({"docs": n_docs, "chunks": n_chunks})

@app.route("/api/docs")
def api_docs():
    db = get_db()
    rows = db.execute(
        "SELECT id, filename, n_chunks, uploaded_at FROM docs ORDER BY id").fetchall()
    return jsonify([{"id": r["id"], "filename": r["filename"],
                     "chunks": r["n_chunks"], "uploaded_at": r["uploaded_at"]}
                    for r in rows])

@app.route("/api/upload", methods=["POST"])
def api_upload():
    files = request.files.getlist("files")
    if not files or all(f.filename == "" for f in files):
        return jsonify({"ok": False, "error": "업로드할 파일을 선택해 주세요."}), 400
    db = get_db()
    added, errors = [], []
    for f in files:
        name = f.filename or ""
        ext = os.path.splitext(name)[1].lower()
        if ext not in ALLOWED_EXT:
            errors.append(f"‘{name}’: txt/md 파일만 업로드할 수 있습니다.")
            continue
        try:
            raw = f.read()
        except Exception:
            errors.append(f"‘{name}’: 파일을 읽지 못했습니다.")
            continue
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            try:
                text = raw.decode("cp949")
            except Exception:
                errors.append(f"‘{name}’: 텍스트 인코딩을 확인할 수 없습니다.")
                continue
        if not text.strip():
            errors.append(f"‘{name}’: 내용이 비어 있는 파일입니다.")
            continue
        if len(text) > 500_000:
            errors.append(f"‘{name}’: 파일이 너무 큽니다. (50만자 이하)")
            continue
        chunks = chunk_text(text)
        if not chunks:
            errors.append(f"‘{name}’: 청크로 나눌 수 있는 내용이 없습니다.")
            continue
        cur = db.execute(
            "INSERT INTO docs (filename, content, n_chunks, uploaded_at) VALUES (?,?,?,?)",
            (name, text, len(chunks), datetime.now().strftime("%Y-%m-%d %H:%M")))
        doc_id = cur.lastrowid
        db.executemany(
            "INSERT INTO chunks (doc_id, idx, text) VALUES (?,?,?)",
            [(doc_id, i, ch) for i, ch in enumerate(chunks)])
        added.append({"filename": name, "chunks": len(chunks)})
    db.commit()
    rebuild_index()
    ok = bool(added)
    msg = f"{len(added)}개 문서가 등록되었습니다." if added else "등록된 문서가 없습니다."
    return jsonify({"ok": ok, "added": added, "errors": errors, "message": msg})

@app.route("/api/docs/<int:doc_id>", methods=["DELETE"])
def api_delete_doc(doc_id):
    db = get_db()
    row = db.execute("SELECT filename FROM docs WHERE id=?", (doc_id,)).fetchone()
    if not row:
        return jsonify({"ok": False, "error": "이미 삭제된 문서입니다."}), 404
    db.execute("DELETE FROM chunks WHERE doc_id=?", (doc_id,))
    db.execute("DELETE FROM docs WHERE id=?", (doc_id,))
    db.commit()
    rebuild_index()
    return jsonify({"ok": True, "message": f"‘{row['filename']}’을(를) 삭제했습니다."})

@app.route("/api/seed", methods=["POST"])
def api_seed():
    """sample_docs의 샘플 문서를 DB에 적재"""
    db = get_db()
    if not os.path.isdir(SAMPLE_DIR):
        return jsonify({"ok": False, "error": "샘플 문서 폴더를 찾을 수 없습니다."}), 500
    added = []
    for name in sorted(os.listdir(SAMPLE_DIR)):
        ext = os.path.splitext(name)[1].lower()
        if ext not in ALLOWED_EXT:
            continue
        path = os.path.join(SAMPLE_DIR, name)
        with open(path, encoding="utf-8") as fh:
            text = fh.read()
        if not text.strip():
            continue
        exists = db.execute("SELECT 1 FROM docs WHERE filename=?", (name,)).fetchone()
        if exists:
            continue
        chunks = chunk_text(text)
        cur = db.execute(
            "INSERT INTO docs (filename, content, n_chunks, uploaded_at) VALUES (?,?,?,?)",
            (name, text, len(chunks), datetime.now().strftime("%Y-%m-%d %H:%M")))
        doc_id = cur.lastrowid
        db.executemany(
            "INSERT INTO chunks (doc_id, idx, text) VALUES (?,?,?)",
            [(doc_id, i, ch) for i, ch in enumerate(chunks)])
        added.append(name)
    db.commit()
    rebuild_index()
    return jsonify({"ok": True, "added": added,
                    "message": f"샘플 문서 {len(added)}개를 불러왔습니다." if added
                    else "샘플 문서가 이미 모두 등록되어 있습니다."})

@app.route("/api/chat", methods=["POST"])
def api_chat():
    data = request.get_json(silent=True) or {}
    question = (data.get("question") or "").strip()
    if not question:
        return jsonify({"ok": False, "error": "질문을 입력해 주세요."}), 400
    if len(question) > 500:
        return jsonify({"ok": False, "error": "질문은 500자 이내로 입력해 주세요."}), 400
    db = get_db()
    n_docs = db.execute("SELECT COUNT(*) c FROM docs").fetchone()["c"]
    if n_docs == 0:
        return jsonify({"ok": True, "found": False, "answer":
            "등록된 문서가 없습니다. 왼쪽 ‘문서’ 패널에서 샘플 문서를 불러오거나, "
            "txt/md 파일을 업로드한 뒤 질문해 주세요.",
            "evidences": [], "reason": "no_docs"})
    t0 = time.time()
    hits = [h for h in search(question) if h[1] >= MIN_SCORE]
    elapsed = round(time.time() - t0, 3)
    if not hits:
        return jsonify({"ok": True, "found": False,
            "answer": "문서에서 찾을 수 없습니다. 다른 표현으로 질문해 보시거나, "
                      "관련 문서를 추가로 업로드해 주세요.",
            "evidences": [], "reason": "no_match", "elapsed": elapsed})
    result = build_answer(question, hits)
    result["ok"] = True
    result["elapsed"] = elapsed
    return jsonify(result)

@app.errorhandler(413)
def too_large(e):
    return jsonify({"ok": False,
                    "error": f"파일이 너무 큽니다. {MAX_UPLOAD_MB}MB 이하의 파일만 업로드할 수 있습니다."}), 413

@app.errorhandler(Exception)
def handle_exc(e):
    app.logger.exception("Unhandled error")
    return jsonify({"ok": False,
                    "error": "일시적인 오류가 발생했습니다. 잠시 후 다시 시도해 주세요."}), 500

# ---------------- 시작 ----------------
def seed_if_empty():
    db = sqlite3.connect(DB_PATH)
    n = db.execute("SELECT COUNT(*) FROM docs").fetchone()[0]
    db.close()
    if n == 0 and os.path.isdir(SAMPLE_DIR):
        with app.test_request_context():
            g.db = sqlite3.connect(DB_PATH)
            g.db.row_factory = sqlite3.Row
            try:
                api_seed()
            finally:
                g.db.close()
                g.pop("db", None)

if __name__ == "__main__":
    init_db()
    seed_if_empty()
    rebuild_index()
    app.run(host="127.0.0.1", port=5001, debug=False)
